import copy
import functools
import json
import os
from pathlib import Path
from pdb import set_trace as st

import blobfile as bf
import imageio
import numpy as np
import torch as th
import torch.distributed as dist
import torchvision
from PIL import Image
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.nn import update_ema
from guided_diffusion.resample import LossAwareSampler, UniformSampler
# from .train_util import TrainLoop3DRec
from guided_diffusion.train_util import (TrainLoop, calc_average_loss,
                                         find_ema_checkpoint,
                                         find_resume_checkpoint,
                                         get_blob_logdir, log_loss_dict,
                                         log_rec3d_loss_dict,
                                         parse_resume_step_from_filename)
from guided_diffusion.gaussian_diffusion import ModelMeanType

import dnnlib

from .train_util_diffusion_single_stage import TrainLoop3DDiffusionSingleStage

from .cvD.nvsD_canoD import TrainLoop3DcvD_nvsD_canoD

from torch.cuda.amp import custom_bwd, custom_fwd

class SpecifyGradient(th.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return th.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None



class TrainLoop3DDiffusionSingleStageSDS(TrainLoop3DcvD_nvsD_canoD):
    """merge the reconstruction and ddpm parameters into a single optimizer.
    """
    # def __init__(self, *, rec_model, denoise_model, diffusion, loss_class, data, eval_data, batch_size, microbatch, lr, ema_rate, log_interval, eval_interval, save_interval, resume_checkpoint, use_fp16=False, fp16_scale_growth=0.001, schedule_sampler=None, weight_decay=0, lr_anneal_steps=0, iterations=10001, ignore_resume_opt=False, freeze_ae=False, denoised_ae=True, triplane_scaling_divider=10, use_amp=False, **kwargs):
    #     super().__init__(rec_model=rec_model, denoise_model=denoise_model, diffusion=diffusion, loss_class=loss_class, data=data, eval_data=eval_data, batch_size=batch_size, microbatch=microbatch, lr=lr, ema_rate=ema_rate, log_interval=log_interval, eval_interval=eval_interval, save_interval=save_interval, resume_checkpoint=resume_checkpoint, use_fp16=use_fp16, fp16_scale_growth=fp16_scale_growth, schedule_sampler=schedule_sampler, weight_decay=weight_decay, lr_anneal_steps=lr_anneal_steps, iterations=iterations, ignore_resume_opt=ignore_resume_opt, freeze_ae=freeze_ae, denoised_ae=denoised_ae, triplane_scaling_divider=triplane_scaling_divider, use_amp=use_amp, **kwargs)
    def __init__(self, *, model, denoise_model, diffusion, loss_class, data, eval_data, batch_size, microbatch, lr, ema_rate, log_interval, eval_interval, save_interval, resume_checkpoint, use_fp16=False, fp16_scale_growth=0.001, weight_decay=0, lr_anneal_steps=0, iterations=10001, load_submodule_name='', ignore_resume_opt=False, use_amp=False, **kwargs):
        super().__init__(model=model, loss_class=loss_class, data=data, eval_data=eval_data, batch_size=batch_size, microbatch=microbatch, lr=lr, ema_rate=ema_rate, log_interval=log_interval, eval_interval=eval_interval, save_interval=save_interval, resume_checkpoint=resume_checkpoint, use_fp16=use_fp16, fp16_scale_growth=fp16_scale_growth, weight_decay=weight_decay, lr_anneal_steps=lr_anneal_steps, iterations=iterations, load_submodule_name=load_submodule_name, ignore_resume_opt=ignore_resume_opt, use_amp=use_amp, **kwargs)

        # * diffusion flags
        self.diffusion = diffusion
        self.denoise_model = denoise_model

        self._load_and_sync_parameters()
        self.mp_ddpm_trainer = MixedPrecisionTrainer(
            model=self.denoise_model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            use_amp=use_amp,
        )

        self.opt = AdamW(self.mp_ddpm_trainer.master_params,
                         lr=self.lr,
                         weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_ddpm_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        # print('creating DDP')
        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.rec_model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn("Distributed training requires CUDA. "
                            "Gradients will not be synchronized properly!")
            self.use_ddp = False
            self.ddp_model = self.rec_model
        # print('creating DDP done')



    def forward_backward(self, batch, *args, **kwargs):
        self.mp_trainer_rec.zero_grad()
        batch_size = batch['img'].shape[0]

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev())
                for k, v in batch.items()
            }

            last_batch = (i + self.microbatch) >= batch_size

            # =================================== ae part ===================================
            with th.cuda.amp.autocast(dtype=th.float16,
                                      enabled=self.mp_trainer_rec.use_amp
                                      and not self.freeze_ae):

                pred = self.ddp_rec_model(img=micro['img_to_encoder'],
                                          c=micro['c'])  # pred: (B, 3, 64, 64)
                # ! AE step? not individually.
                # ! TODO, add cano - nvs here
                if not self.freeze_ae:
                # if False:
                    target = micro

                    if last_batch or not self.use_ddp:
                        ae_loss, loss_dict = self.loss_class(pred,
                                                             target,
                                                             test_mode=False)
                    else:
                        with self.ddp_ddpm_model.no_sync():  # type: ignore
                            ae_loss, loss_dict = self.loss_class(
                                pred, target, test_mode=False)

                    log_rec3d_loss_dict(loss_dict)
                else:
                    ae_loss = th.tensor(0.0).to(dist_util.dev())

                micro_to_denoise = pred[
                    self.
                    latent_name] / self.triplane_scaling_divider  # normalize std to 1

                t, weights = self.schedule_sampler.sample(
                    micro_to_denoise.shape[0], dist_util.dev())

                model_kwargs = {}

                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_ddpm_model,
                    micro_to_denoise,  # x_start
                    t,
                    model_kwargs=model_kwargs,
                    return_detail=True
                )

                # ! DDPM step
                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                    # denoised_out = denoised_fn()
                else:
                    with self.ddp_ddpm_model.no_sync():  # type: ignore
                        losses = compute_losses()

                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach())

                denoise_loss = (losses["loss"] * weights).mean()

                x_t = losses['x_t']
                model_output = losses.pop('model_output')
                diffusion_target = losses.pop('diffusion_target')
                alpha = losses.pop('alpha')

                log_loss_dict(self.diffusion, t,
                              {k: v * weights
                               for k, v in losses.items()})


                loss = ae_loss + denoise_loss

                diffusion_residual = 2 * weights.reshape(-1,1,1,1) * (diffusion_target-model_output) * alpha # TODO, add alpha weights
                sds_loss = SpecifyGradient.apply(micro_to_denoise, micro_to_denoise)

            # exit AMP before backward
            self.mp_trainer_rec.backward(loss)

            # TODO, merge visualization with original AE
            # =================================== denoised AE log part ===================================

            if dist_util.get_rank() == 0 and self.step % 500 == 0:
            # if dist_util.get_rank() == 1 and self.step % 500 == 0:
                with th.no_grad():
                    # gt_vis = th.cat([batch['img'], batch['depth']], dim=-1)

                    # st()

                    gt_depth = micro['depth']
                    if gt_depth.ndim == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() -
                                                              gt_depth.min())
                    # if True:
                    pred_depth = pred['image_depth']
                    pred_depth = (pred_depth - pred_depth.min()) / (
                        pred_depth.max() - pred_depth.min())
                    pred_img = pred['image_raw']
                    gt_img = micro['img']

                    # if 'image_sr' in pred:  # TODO
                    #     pred_img = th.cat(
                    #         [self.pool_512(pred_img), pred['image_sr']],
                    #         dim=-1)
                    #     gt_img = th.cat(
                    #         [self.pool_512(micro['img']), micro['img_sr']],
                    #         dim=-1)
                    #     pred_depth = self.pool_512(pred_depth)
                    #     gt_depth = self.pool_512(gt_depth)

                    gt_vis = th.cat(
                        [
                            gt_img, micro['img'], micro['img'],
                            gt_depth.repeat_interleave(3, dim=1)
                        ],
                        dim=-1)[0:1]  # TODO, fail to load depth. range [0, 1]

                    noised_ae_pred = self.ddp_rec_model(
                        img=None,
                        c=micro['c'][0:1],
                        latent=x_t[0:1] * self.
                        triplane_scaling_divider,  # TODO, how to define the scale automatically
                        behaviour=self.render_latent_behaviour)

                    if self.diffusion.model_mean_type == ModelMeanType.START_X:
                        pred_xstart = model_output
                    else:  # * used here
                        pred_xstart = self.diffusion._predict_xstart_from_eps(
                            x_t=x_t, t=t, eps=model_output)

                    denoised_ae_pred = self.ddp_rec_model(
                        img=None,
                        c=micro['c'][0:1],
                        latent=pred_xstart[0:1] * self.
                        triplane_scaling_divider,  # TODO, how to define the scale automatically?
                        behaviour=self.render_latent_behaviour)

                    # denoised_out = denoised_ae_pred


                    pred_vis = th.cat([
                        pred_img[0:1], noised_ae_pred['image_raw'][0:1],
                        denoised_ae_pred['image_raw'][0:1],
                        pred_depth[0:1].repeat_interleave(3, dim=1)
                    ],
                                      dim=-1)  # B, 3, H, W
                    # s

                    vis = th.cat([gt_vis, pred_vis], dim=-2)[0].permute(
                        1, 2, 0).cpu()  # ! pred in range[-1, 1]

                    # vis = th.cat([
                    #     self.pool_128(micro['img']), x_t[:, :3, ...],
                    #     denoised_out['pred_xstart'][:, :3, ...]
                    # ],
                    #              dim=-1)[0].permute(
                    #                  1, 2, 0).cpu()  # ! pred in range[-1, 1]

                    # vis_grid = torchvision.utils.make_grid(vis) # HWC
                    vis = vis.numpy() * 127.5 + 127.5
                    vis = vis.clip(0, 255).astype(np.uint8)
                    Image.fromarray(vis).save(
                        f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t[0].item()}.jpg'
                    )
                    print(
                        'log denoised vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t[0].item()}.jpg'
                    )

                    th.cuda.empty_cache()


    # def forward_backward(self, batch, *args, **kwargs):
    #     self.mp_trainer.zero_grad()
    #     batch_size = batch['img'].shape[0]

    #     for i in range(0, batch_size, self.microbatch):

    #         micro = {
    #             k: v[i:i + self.microbatch].to(dist_util.dev())
    #             for k, v in batch.items()
    #         }

    #         last_batch = (i + self.microbatch) >= batch_size

    #         # =================================== ae part ===================================
    #         with th.cuda.amp.autocast(dtype=th.float16,
    #                                   enabled=self.mp_trainer.use_amp
    #                                   and not self.freeze_ae):

    #             pred = self.ddp_rec_model(img=micro['img_to_encoder'],
    #                                       c=micro['c'])  # pred: (B, 3, 64, 64)
    #             # ! AE step? not individually.
    #             # ! TODO, add cano - nvs here
    #             if not self.freeze_ae:
    #             # if False:
    #                 target = micro

    #                 if last_batch or not self.use_ddp:
    #                     ae_loss, loss_dict = self.loss_class(pred,
    #                                                          target,
    #                                                          test_mode=False)
    #                 else:
    #                     with self.ddp_ddpm_model.no_sync():  # type: ignore
    #                         ae_loss, loss_dict = self.loss_class(
    #                             pred, target, test_mode=False)

    #                 log_rec3d_loss_dict(loss_dict)
    #             else:
    #                 ae_loss = th.tensor(0.0).to(dist_util.dev())

    #             micro_to_denoise = pred[
    #                 self.
    #                 latent_name] / self.triplane_scaling_divider  # normalize std to 1

    #             t, weights = self.schedule_sampler.sample(
    #                 micro_to_denoise.shape[0], dist_util.dev())

    #             model_kwargs = {}

    #             compute_losses = functools.partial(
    #                 self.diffusion.training_losses,
    #                 self.ddp_ddpm_model,
    #                 micro_to_denoise,  # x_start
    #                 t,
    #                 model_kwargs=model_kwargs,
    #                 return_detail=True
    #             )

    #             # ! DDPM step
    #             if last_batch or not self.use_ddp:
    #                 losses = compute_losses()
    #                 # denoised_out = denoised_fn()
    #             else:
    #                 with self.ddp_ddpm_model.no_sync():  # type: ignore
    #                     losses = compute_losses()

    #             if isinstance(self.schedule_sampler, LossAwareSampler):
    #                 self.schedule_sampler.update_with_local_losses(
    #                     t, losses["loss"].detach())

    #             denoise_loss = (losses["loss"] * weights).mean()

    #             x_t = losses['x_t']
    #             model_output = losses.pop('model_output')
    #             diffusion_target = losses.pop('diffusion_target')
    #             alpha = losses.pop('alpha')

    #             log_loss_dict(self.diffusion, t,
    #                           {k: v * weights
    #                            for k, v in losses.items()})


    #             loss = ae_loss + denoise_loss

    #             diffusion_residual = 2 * weights.reshape(-1,1,1,1) * (diffusion_target-model_output) * alpha # TODO, add alpha weights
    #             sds_loss = SpecifyGradient.apply(micro_to_denoise, micro_to_denoise)

    #         # exit AMP before backward
    #         self.mp_trainer.backward(loss)

    #         # TODO, merge visualization with original AE
    #         # =================================== denoised AE log part ===================================

    #         if dist_util.get_rank() == 0 and self.step % 500 == 0:
    #         # if dist_util.get_rank() == 1 and self.step % 500 == 0:
    #             with th.no_grad():
    #                 # gt_vis = th.cat([batch['img'], batch['depth']], dim=-1)

    #                 # st()

    #                 gt_depth = micro['depth']
    #                 if gt_depth.ndim == 3:
    #                     gt_depth = gt_depth.unsqueeze(1)
    #                 gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() -
    #                                                           gt_depth.min())
    #                 # if True:
    #                 pred_depth = pred['image_depth']
    #                 pred_depth = (pred_depth - pred_depth.min()) / (
    #                     pred_depth.max() - pred_depth.min())
    #                 pred_img = pred['image_raw']
    #                 gt_img = micro['img']

    #                 # if 'image_sr' in pred:  # TODO
    #                 #     pred_img = th.cat(
    #                 #         [self.pool_512(pred_img), pred['image_sr']],
    #                 #         dim=-1)
    #                 #     gt_img = th.cat(
    #                 #         [self.pool_512(micro['img']), micro['img_sr']],
    #                 #         dim=-1)
    #                 #     pred_depth = self.pool_512(pred_depth)
    #                 #     gt_depth = self.pool_512(gt_depth)

    #                 gt_vis = th.cat(
    #                     [
    #                         gt_img, micro['img'], micro['img'],
    #                         gt_depth.repeat_interleave(3, dim=1)
    #                     ],
    #                     dim=-1)[0:1]  # TODO, fail to load depth. range [0, 1]

    #                 noised_ae_pred = self.ddp_rec_model(
    #                     img=None,
    #                     c=micro['c'][0:1],
    #                     latent=x_t[0:1] * self.
    #                     triplane_scaling_divider,  # TODO, how to define the scale automatically
    #                     behaviour=self.render_latent_behaviour)

    #                 if self.diffusion.model_mean_type == ModelMeanType.START_X:
    #                     pred_xstart = model_output
    #                 else:  # * used here
    #                     pred_xstart = self.diffusion._predict_xstart_from_eps(
    #                         x_t=x_t, t=t, eps=model_output)

    #                 denoised_ae_pred = self.ddp_rec_model(
    #                     img=None,
    #                     c=micro['c'][0:1],
    #                     latent=pred_xstart[0:1] * self.
    #                     triplane_scaling_divider,  # TODO, how to define the scale automatically?
    #                     behaviour=self.render_latent_behaviour)

    #                 # denoised_out = denoised_ae_pred


    #                 pred_vis = th.cat([
    #                     pred_img[0:1], noised_ae_pred['image_raw'][0:1],
    #                     denoised_ae_pred['image_raw'][0:1],
    #                     pred_depth[0:1].repeat_interleave(3, dim=1)
    #                 ],
    #                                   dim=-1)  # B, 3, H, W
    #                 # s

    #                 vis = th.cat([gt_vis, pred_vis], dim=-2)[0].permute(
    #                     1, 2, 0).cpu()  # ! pred in range[-1, 1]

    #                 # vis = th.cat([
    #                 #     self.pool_128(micro['img']), x_t[:, :3, ...],
    #                 #     denoised_out['pred_xstart'][:, :3, ...]
    #                 # ],
    #                 #              dim=-1)[0].permute(
    #                 #                  1, 2, 0).cpu()  # ! pred in range[-1, 1]

    #                 # vis_grid = torchvision.utils.make_grid(vis) # HWC
    #                 vis = vis.numpy() * 127.5 + 127.5
    #                 vis = vis.clip(0, 255).astype(np.uint8)
    #                 Image.fromarray(vis).save(
    #                     f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t[0].item()}.jpg'
    #                 )
    #                 print(
    #                     'log denoised vis to: ',
    #                     f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t[0].item()}.jpg'
    #                 )

    #                 th.cuda.empty_cache()
