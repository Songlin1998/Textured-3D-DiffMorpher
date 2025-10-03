import copy
import functools
import json
import os
from pathlib import Path
from pdb import set_trace as st
from random import betavariate

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
import vision_aided_loss

from torch.cuda.amp import custom_bwd, custom_fwd


class SpecifyGradient(th.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return th.ones([1],
                       device=input_tensor.device,
                       dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


class AddGradient(th.autograd.Function):
    """
    the forward remains the same; in backward add the grad_output with sds_grad for BP
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, sds_grad):
        ctx.save_for_backward(sds_grad)
        # ctx.save_for_backward(input_tensor, sds_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        # return th.ones([1],
        #                device=input_tensor.device,
        #                dtype=input_tensor.dtype)
        return input_tensor  # since reconstruction loss still needed

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        sds_grad, = ctx.saved_tensors
        ae_grad_with_sds = (grad_output + sds_grad)
        return ae_grad_with_sds, None


class AE_and_diffusion(th.nn.Module):

    def __init__(self, rec_model, denoise_model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rec_model = rec_model
        self.denoise_model = denoise_model


class TrainLoop3DDiffusionSingleStagecvDSDS():
    """merge the reconstruction and ddpm parameters into a single optimizer.
    """

    def __init__(
            self,
            *,
            # model,
            rec_model,
            denoise_model,
            diffusion,
            loss_class,
            data,
            eval_data,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            log_interval,
            eval_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=0.001,
            schedule_sampler=None,
            weight_decay=0,
            lr_anneal_steps=0,
            iterations=10001,
            ignore_resume_opt=False,
            # freeze_ae=False,
            freeze_ae=False,
            denoised_ae=True,
            triplane_scaling_divider=10,
            use_amp=False,
            **kwargs):

        # ! copied original diffusiont trainer init attributes here===
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = ([ema_rate] if isinstance(ema_rate, float) else
                         [float(x) for x in ema_rate.split(",")])
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        # self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        # ! load and sync diffusion model
        # ! replace with single stage model

        self.model = AE_and_diffusion(rec_model, denoise_model)
        # self.model = denoise_model

        self._load_and_sync_parameters(model=getattr(self.model,
                                                     'denoise_model'),
                                       model_name='ddpm')

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            use_amp=use_amp,
            model_name='single-stage-model',  # TODO, 
        )

        self.opt = AdamW(self._init_optim_groups(kwargs))

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

            self.ema_params = [  # only ddpm using ema
                self._load_ema_parameters(
                    rate,
                    # model=self.model.denoise_model,
                    # model_name='ddpm'
                )  # only load ema for ddpm model
                for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_ddpm_model = DDP(
                self.model.denoise_model,
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
            self.ddp_ddpm_model = self.model.denoise_model

        # ! Reconstruction model here

        self.latent_name = 'latent_normalized'  # normalized triplane latent
        self.render_latent_behaviour = 'triplane_dec'  # directly render using triplane operations

        self.pool_512 = th.nn.AdaptiveAvgPool2d((512, 512))
        self.pool_256 = th.nn.AdaptiveAvgPool2d((256, 256))
        self.pool_128 = th.nn.AdaptiveAvgPool2d((128, 128))
        self.loss_class = loss_class
        # self.ddp_rec_model = rec_model
        self.eval_interval = eval_interval
        self.eval_data = eval_data
        self.iterations = iterations
        self.triplane_scaling_divider = triplane_scaling_divider

        self._load_and_sync_parameters(model=self.model.rec_model,
                                       model_name='rec')

        self.denoised_ae = denoised_ae

        if dist_util.get_rank() == 0:
            self.writer = SummaryWriter(log_dir=f'{logger.get_dir()}/runs')
            print(self.opt)

        if self.use_ddp is True:
            self.model.rec_model = th.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model.rec_model)
            self.ddp_rec_model = DDP(
                getattr(self.model, 'rec_model'),
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
                # find_unused_parameters=True,
            )
        else:
            self.ddp_rec_model = getattr(self.model, 'rec_model')
            # self.model.rec_model

        # ! no fixing vit for all the experiments
        # if freeze_ae:
        #     self.ddp_rec_model.eval()
        #     self.ddp_rec_model.requires_grad_(False)

        self.freeze_ae = freeze_ae

        # ! cvD

        device = dist_util.dev()
        self.canonical_cvD = vision_aided_loss.Discriminator(
            cv_type='clip', loss_type='multilevel_sigmoid_s',
            device=device).to(device)
        self.canonical_cvD.cv_ensemble.requires_grad_(
            False)  # Freeze feature extractor

        self._load_and_sync_parameters(model=self.canonical_cvD,
                                       model_name='cano_cvD')

        self.mp_trainer_canonical_cvD = MixedPrecisionTrainer(
            model=self.canonical_cvD,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            model_name='cano_cvD',
            use_amp=use_amp)

        self.opt_cano_cvD = AdamW(
            self.mp_trainer_canonical_cvD.master_params,
            lr=1e-5,  # same as the G
            betas=(0, 0.99),
            eps=1e-8)  # dlr in biggan cfg

        if self.use_ddp:
            self.ddp_canonical_cvD = DDP(
                self.canonical_cvD,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            self.ddp_canonical_cvD = self.canonical_cvD

        self.nvs_cvD = vision_aided_loss.Discriminator(
            cv_type='clip', loss_type='multilevel_sigmoid_s',
            device=device).to(device)
        self.nvs_cvD.cv_ensemble.requires_grad_(
            False)  # Freeze feature extractor

        self._load_and_sync_parameters(model=self.nvs_cvD, model_name='nvs_cvD')

        self.mp_trainer_nvs_cvD = MixedPrecisionTrainer(
            model=self.nvs_cvD,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            model_name='nvs_cvD',
            use_amp=use_amp)

        self.opt_nvs_cvD = AdamW(
            self.mp_trainer_nvs_cvD.master_params,
            lr=1e-5,  # same as the G
            betas=(0, 0.99),
            eps=1e-8)  # dlr in biggan cfg

        if self.use_ddp:
            self.ddp_nvs_cvD = DDP(  # ddp_nvs_cvD
                self.nvs_cvD,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            self.ddp_nvs_cvD = self.nvs_cvD

        th.cuda.empty_cache()

        # ! add nvs D? input value from both

    def run_D_Diter(self, real, fake, D=None):
        # Dmain: Minimize logits for generated images and maximize logits for real images.
        if D is None:
            D = self.ddp_canonical_cvD

        lossD = D(real, for_real=True).mean() + D(fake, for_real=False).mean()
        return lossD

    def _init_optim_groups(self, kwargs):
        optim_groups = [
            # ! diffusion part
            {
                'name': 'ddpm',
                'params': self.mp_trainer.model.denoise_model.parameters(),
                'lr': self.lr,
                'weight_decay': self.weight_decay,
            },
            # ! AE part following: vit encoder
            {
                'name': 'vit_encoder',
                'params': self.mp_trainer.model.rec_model.encoder.parameters(),
                'lr': kwargs['encoder_lr'],
                'weight_decay': kwargs['encoder_weight_decay']
            },
            # vit decoder
            {
                'name':
                'vit_decoder',
                'params':
                self.mp_trainer.model.rec_model.decoder.vit_decoder.parameters(
                ),
                'lr':
                kwargs['vit_decoder_lr'],
                'weight_decay':
                kwargs['vit_decoder_wd']
            },
            {
                'name':
                'vit_decoder_pred',
                'params':
                self.mp_trainer.model.rec_model.decoder.decoder_pred.
                parameters(),
                'lr':
                kwargs['vit_decoder_lr'],
                # 'weight_decay': 0
                'weight_decay':
                kwargs['vit_decoder_wd']
            },

            # triplane decoder
            {
                'name':
                'triplane_decoder',
                'params':
                self.mp_trainer.model.rec_model.decoder.triplane_decoder.
                parameters(),
                'lr':
                kwargs['triplane_decoder_lr'],
                # 'weight_decay': self.weight_decay
            },
        ]

        if self.mp_trainer.model.rec_model.decoder.superresolution is not None:
            optim_groups.append({
                'name':
                'triplane_decoder_superresolution',
                'params':
                self.mp_trainer.model.rec_model.decoder.superresolution.
                parameters(),
                'lr':
                kwargs['super_resolution_lr'],
            })

        return optim_groups

    def run_loop(self, batch=None):
        th.cuda.empty_cache()
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            # let all processes sync up before starting with a new epoch of training
            dist_util.synchronize()

            # if self.step % self.eval_interval == 0 and self.step != 0:
            if self.step % self.eval_interval == 0:
                if dist_util.get_rank() == 0:
                    self.eval_ddpm_sample()
                    self.eval_loop()
                    self.eval_novelview_loop()

                # let all processes sync up before starting with a new epoch of training
                dist_util.synchronize()
                th.cuda.empty_cache()

            # ! diffusion step
            batch = next(self.data)
            self.run_step(batch, step='diffusion_step_rec')

            if self.step % 2 == 0:
                batch = next(self.data)
                self.run_step(batch, step='d_step_rec')

            batch = next(self.data)
            self.run_step(batch, step='diffusion_step_nvs')

            if self.step % 2 == 1: # 
                batch = next(self.data)
                self.run_step(batch, step='d_step_nvs')

            batch = next(self.data)
            self.run_step(batch, step='sds')

            if self.step % self.log_interval == 0 and dist_util.get_rank(
            ) == 0:
                out = logger.dumpkvs()
                # * log to tensorboard
                for k, v in out.items():
                    self.writer.add_scalar(f'Loss/{k}', v,
                                           self.step + self.resume_step)

            if self.step % self.save_interval == 0 and self.step != 0:
                self.save(self.mp_trainer, )
                if not self.freeze_ae:
                    self.save(self.mp_trainer, 'rec')
                

                self.save(self.mp_trainer_nvs_cvD, self.mp_trainer_nvs_cvD.model_name)
                self.save(self.mp_trainer_canonical_cvD, self.mp_trainer_canonical_cvD.model_name)

                dist_util.synchronize()

                th.cuda.empty_cache()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST",
                                  "") and self.step > 0:
                    return

            self.step += 1

            if self.step > self.iterations:
                print('reached maximum iterations, exiting')

                # Save the last checkpoint if it wasn't already saved.
                if (self.step - 1) % self.save_interval != 0:
                    self.save()
                    if not self.freeze_ae:
                        self.save(self.mp_trainer, 'rec')
                    self.save(self.mp_trainer_nvs_cvD, self.mp_trainer_nvs_cvD.model_name)
                    self.save(self.mp_trainer_canonical_cvD, self.mp_trainer_canonical_cvD.model_name)

                exit()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            if not self.freeze_ae:
                self.save(self.mp_trainer, 'rec')

    def run_step(self, batch, step='diffusion_step_rec'):
        # self.forward_backward(batch)

        if step == 'diffusion_step':
            self.forward_diffusion(batch, behaviour='diff')
            took_step_g_rec = self.mp_trainer.optimize(self.opt)

            if took_step_g_rec:
                self._update_ema()  # g_ema

        elif step == 'diffusion_step_rec':
            self.forward_diffusion(batch, behaviour='rec')
            took_step_g_rec = self.mp_trainer.optimize(self.opt)

            if took_step_g_rec:
                self._update_ema()  # g_ema

        elif step == 'd_step_rec':
            self.forward_D(batch, behaviour='rec')
            _ = self.mp_trainer_canonical_cvD.optimize(self.opt_cano_cvD)

        elif step == 'diffusion_step_nvs':
            self.forward_diffusion(batch, behaviour='nvs')
            took_step_g_rec = self.mp_trainer.optimize(self.opt)
            # print('3')

            if took_step_g_rec:
                self._update_ema()  # g_ema

        elif step == 'd_step_nvs':
            self.forward_D(batch, behaviour='nvs')
            _ = self.mp_trainer_nvs_cvD.optimize(self.opt_nvs_cvD)

        elif step == 'sds':
            # print('run sds step next')
            self.forward_diffusion(batch, behaviour='sds')
            _ = self.mp_trainer.optimize(self.opt)
            # print('run sds step done')

        else:
            raise NotImplementedError(f'{step} not implemented')

        self._anneal_lr()
        self.log_step()

    # def forward_diffusion(self, batch, behaviour='rec', *args, **kwargs):
    #     self.ddp_canonical_cvD.requires_grad_(False)
    #     self.ddp_nvs_cvD.requires_grad_(False)
    #     self.ddp_rec_model.requires_grad_(True)
    #     self.mp_trainer.zero_grad()

    #     if behaviour == 'sds':
    #         self.ddp_ddpm_model.requires_grad_(False)
    #     else:
    #         self.ddp_ddpm_model.requires_grad_(True)

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

    #             latent = self.ddp_rec_model(
    #                 img=micro['img_to_encoder'],
    #                 c=micro['c'],
    #                 behaviour='enc_dec_wo_triplane')  # pred: (B, 3, 64, 64)

    #             ae_loss = th.tensor(0.0).to(dist_util.dev())
    #             vision_aided_loss = th.tensor(0.0).to(dist_util.dev())
    #             # denoise_loss = th.tensor(0.0).to(dist_util.dev())

    #             if behaviour == 'rec':
    #                 target = micro
    #                 pred = self.ddp_rec_model(latent=latent,
    #                                           c=micro['c'],
    #                                           behaviour='triplane_dec')

    #                 if last_batch or not self.use_ddp:
    #                     ae_loss, loss_dict = self.loss_class(pred,
    #                                                          target,
    #                                                          test_mode=False)
    #                 else:
    #                     with self.ddp_model.no_sync():  # type: ignore
    #                         ae_loss, loss_dict = self.loss_class(
    #                             pred, target, test_mode=False)

    #                 if 'image_sr' in pred:
    #                     vision_aided_loss = self.ddp_canonical_cvD(
    #                         0.5 * pred['image_sr'] +
    #                         0.5 * th.nn.functional.interpolate(
    #                             pred['image_raw'],
    #                             size=pred['image_sr'].shape[2:],
    #                             mode='bilinear'),
    #                         for_G=True).mean()  # [B, 1] shape
    #                 else:
    #                     vision_aided_loss = self.ddp_canonical_cvD(
    #                         pred['image_raw'], for_G=True
    #                     ).mean(
    #                     ) * self.loss_class.opt.rec_cvD_lambda  # [B, 1] shape

    #                 # d_weight = self.loss_class.opt.rec_cvD_lambda
    #                 loss_dict.update({
    #                     'vision_aided_loss/G_rec':
    #                     vision_aided_loss,
    #                 })

    #                 log_rec3d_loss_dict(loss_dict)

    #             elif behaviour == 'nvs':

    #                 novel_view_c = th.cat([micro['c'][1:], micro['c'][:1]])

    #                 pred = self.ddp_rec_model(latent=latent,
    #                                           c=novel_view_c,
    #                                           behaviour='triplane_dec')

    #                 if 'image_sr' in pred:
    #                     vision_aided_loss = self.ddp_nvs_cvD(
    #                         # pred_for_rec['image_sr'],
    #                         0.5 * pred['image_sr'] +
    #                         0.5 * th.nn.functional.interpolate(
    #                             pred['image_raw'],
    #                             size=pred['image_sr'].shape[2:],
    #                             mode='bilinear'),
    #                         for_G=True).mean()  # [B, 1] shape
    #                 else:
    #                     vision_aided_loss = self.ddp_nvs_cvD(
    #                         pred['image_raw'], for_G=True
    #                     ).mean(
    #                     ) * self.loss_class.opt.nvs_cvD_lambda  # [B, 1] shape

    #                 # d_weight = th.tensor(0.1, device=dist_util.dev())

    #                 log_rec3d_loss_dict({
    #                     'vision_aided_loss/G_nvs':
    #                     vision_aided_loss,
    #                 })

    #                 ae_loss = th.tensor(0.0).to(dist_util.dev())

    #             else:
    #                 assert behaviour == 'sds'

    #                 pred = None

    #             # if behaviour != 'sds': # also train diffusion
    #             # assert pred is not None

    #             micro_to_denoise = latent[
    #                 self.
    #                 latent_name] / self.triplane_scaling_divider  # ! detach() from vit computational graph

    #             # if behaviour != 'sds':
    #             # ! micro_to_denoise just for diffusion loss computation
    #             micro_to_denoise.detach_()

    #             # TODO, now no diffusion loss in sds step
    #             t, weights = self.schedule_sampler.sample(
    #                 micro_to_denoise.shape[0], dist_util.dev())

    #             model_kwargs = {}

    #             # print(micro_to_denoise.min(), micro_to_denoise.max())
    #             compute_losses = functools.partial(
    #                 self.diffusion.training_losses,
    #                 self.ddp_ddpm_model,
    #                 micro_to_denoise,  # x_start
    #                 t,
    #                 model_kwargs=model_kwargs,
    #                 return_detail=True)

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

    #             x_t = losses.pop('x_t')
    #             model_output = losses.pop('model_output')
    #             diffusion_target = losses.pop('diffusion_target')
    #             alpha_bar = losses.pop('alpha_bar')

    #             # ! prepare for later single-stage training
    #             # denoised_fn = functools.partial(
    #             #     self.diffusion.p_mean_variance,
    #             #     self.ddp_ddpm_model,
    #             #     x_t,  # x_start
    #             #     t,
    #             #     model_kwargs=model_kwargs)

    #             log_loss_dict(self.diffusion, t,
    #                           {k: v * weights
    #                            for k, v in losses.items()})

    #             if behaviour == 'sds':

    #                 # w = alpha_bar * (1 - alpha_bar**2) / self.triplane_scaling_divider # https://github.com/eladrich/latent-nerf/blob/f49ecefcd48972e69a28e3116fe95edf0fac4dc8/src/stable_diffusion.py#L144
    #                 w = (
    #                     1 - alpha_bar**2
    #                 ) / self.triplane_scaling_divider  # https://github.com/ashawkey/stable-dreamfusion/issues/106
    #                 # sds_grad = (model_output-diffusion_target) * w # * 2
    #                 sds_grad = denoise_loss.clone().detach() * w  # * 2

    #                 ae_loss = SpecifyGradient.apply(
    #                     latent[self.latent_name],
    #                     sds_grad)  # dummy '1' to be scaled by amp.

    #             loss = ae_loss + denoise_loss + vision_aided_loss  # caluclate loss within AMP

    #         # ! cvD loss

    #         # exit AMP before backward
    #         self.mp_trainer.backward(loss)

    #         # TODO, merge visualization with original AE
    #         # =================================== denoised AE log part ===================================

    #         # if behaviour != 'sds' and dist_util.get_rank() == 0 and self.step % 500 == 0:
    #         if dist_util.get_rank() == 0 and self.step % 500 == 0:
    #             with th.no_grad():
    #                 # gt_vis = th.cat([batch['img'], batch['depth']], dim=-1)

    #                 # st()

    #                 if behaviour == 'sds':  # visualize its performance of novel view reconstructions
    #                     novel_view_c = th.cat([micro['c'][1:], micro['c'][:1]])
    #                     pred = self.ddp_rec_model(latent=latent,
    #                                               c=novel_view_c,
    #                                               behaviour='triplane_dec')

    #                 assert pred is not None

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

    #                 # if denoised_out is None:
    #                 # if not self.denoised_ae:
    #                 # denoised_out = denoised_fn()

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

    #                 # if not self.denoised_ae:
    #                 #     denoised_ae_pred = self.ddp_rec_model(
    #                 #         img=None,
    #                 #         c=micro['c'][0:1],
    #                 #         latent=denoised_out['pred_xstart'][0:1] * self.
    #                 #         triplane_scaling_divider,  # TODO, how to define the scale automatically
    #                 #         behaviour=self.render_latent_behaviour)
    #                 # else:
    #                 #     assert denoised_ae_pred is not None
    #                 #     denoised_ae_pred['image_raw'] = denoised_ae_pred[
    #                 #         'image_raw'][0:1]

    #                 # print(pred_img.shape)
    #                 # print('denoised_ae:', self.denoised_ae)

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
    #                     f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t[0].item()}_{behaviour}.jpg'
    #                 )
    #                 print(
    #                     'log denoised vis to: ',
    #                     f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t[0].item()}_{behaviour}.jpg'
    #                 )

    #                 th.cuda.empty_cache()

    @th.no_grad()
    # def eval_loop(self, c_list:list):
    def eval_novelview_loop(self):
        # novel view synthesis given evaluation camera trajectory
        video_out = imageio.get_writer(
            f'{logger.get_dir()}/video_novelview_{self.step+self.resume_step}.mp4',
            mode='I',
            fps=60,
            codec='libx264')

        all_loss_dict = []
        novel_view_micro = {}

        # for i in range(0, len(c_list), 1): # TODO, larger batch size for eval
        for i, batch in enumerate(tqdm(self.eval_data)):
            # for i in range(0, 8, self.microbatch):
            # c = c_list[i].to(dist_util.dev()).reshape(1, -1)
            micro = {k: v.to(dist_util.dev()) for k, v in batch.items()}

            if i == 0:
                novel_view_micro = {
                    k: v[0:1].to(dist_util.dev()).repeat_interleave(
                        micro['img'].shape[0], 0)
                    for k, v in batch.items()
                }
            else:
                # if novel_view_micro['c'].shape[0] < micro['img'].shape[0]:
                novel_view_micro = {
                    k: v[0:1].to(dist_util.dev()).repeat_interleave(
                        micro['img'].shape[0], 0)
                    for k, v in novel_view_micro.items()
                }

            pred = self.ddp_rec_model(img=novel_view_micro['img_to_encoder'],
                                      c=micro['c'])  # pred: (B, 3, 64, 64)
            # target = {
            #     'img': micro['img'],
            #     'depth': micro['depth'],
            #     'depth_mask': micro['depth_mask']
            # }
            # targe

            _, loss_dict = self.loss_class(pred, micro, test_mode=True)
            all_loss_dict.append(loss_dict)

            # ! move to other places, add tensorboard

            # pred_vis = th.cat([
            #     pred['image_raw'],
            #     -pred['image_depth'].repeat_interleave(3, dim=1)
            # ],
            #                   dim=-1)

            # normalize depth
            # if True:
            pred_depth = pred['image_depth']
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())
            if 'image_sr' in pred:
                # pred_vis = th.cat([
                #     micro['img_sr'],
                #     self.pool_512(pred['image_raw']), pred['image_sr'],
                #     self.pool_512(pred_depth).repeat_interleave(3, dim=1)
                # ],
                #                   dim=-1)

                if pred['image_sr'].shape[-1] == 512:
                    pred_vis = th.cat([
                        micro['img_sr'],
                        self.pool_512(pred['image_raw']), pred['image_sr'],
                        self.pool_512(pred_depth).repeat_interleave(3, dim=1)
                    ],
                                    dim=-1)

                elif pred['image_sr'].shape[-1] == 256:
                    pred_vis = th.cat([
                        micro['img_sr'],
                        self.pool_256(pred['image_raw']), pred['image_sr'],
                        self.pool_256(pred_depth).repeat_interleave(3, dim=1)
                    ],
                                    dim=-1)
                else:
                    pred_vis = th.cat([
                        micro['img_sr'], (pred['image_raw']), pred['image_sr'],
                        (pred_depth).repeat_interleave(3, dim=1)
                    ],
                                      dim=-1)

            else:
                pred_vis = th.cat([
                    self.pool_128(micro['img']), pred['image_raw'],
                    pred_depth.repeat_interleave(3, dim=1)
                ],
                                  dim=-1)  # B, 3, H, W

            vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
            vis = vis * 127.5 + 127.5
            vis = vis.clip(0, 255).astype(np.uint8)

            for j in range(vis.shape[0]):
                video_out.append_data(vis[j])

        video_out.close()

        val_scores_for_logging = calc_average_loss(all_loss_dict)
        with open(os.path.join(logger.get_dir(), 'scores_novelview.json'),
                  'a') as f:
            json.dump({'step': self.step, **val_scores_for_logging}, f)

        # * log to tensorboard
        for k, v in val_scores_for_logging.items():
            self.writer.add_scalar(f'Eval/NovelView/{k}', v,
                                   self.step + self.resume_step)

    @th.no_grad()
    # def eval_loop(self, c_list:list):
    def eval_loop(self):
        # novel view synthesis given evaluation camera trajectory
        video_out = imageio.get_writer(
            f'{logger.get_dir()}/video_{self.step+self.resume_step}.mp4',
            mode='I',
            fps=60,
            codec='libx264')
        all_loss_dict = []

        # for i in range(0, len(c_list), 1): # TODO, larger batch size for eval
        for i, batch in enumerate(tqdm(self.eval_data)):
            # for i in range(0, 8, self.microbatch):
            # c = c_list[i].to(dist_util.dev()).reshape(1, -1)
            micro = {k: v.to(dist_util.dev()) for k, v in batch.items()}

            # pred = self.model(img=micro['img_to_encoder'],
            #                   c=micro['c'])  # pred: (B, 3, 64, 64)

            # pred of rec model
            pred = self.ddp_rec_model(img=micro['img_to_encoder'],
                                      c=micro['c'])  # pred: (B, 3, 64, 64)

            pred_depth = pred['image_depth']
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())

            if pred.get('image_sr', None) is not None:

                if pred['image_sr'].shape[-1] == 512:

                    pred_vis = th.cat([
                        micro['img_sr'],
                        self.pool_512(pred['image_raw']), pred['image_sr'],
                        self.pool_512(pred_depth).repeat_interleave(3, dim=1)
                    ],
                                      dim=-1)

                elif pred['image_sr'].shape[-1] == 256:
                    pred_vis = th.cat([
                        micro['img_sr'],
                        self.pool_256(pred['image_raw']), pred['image_sr'],
                        self.pool_256(pred_depth).repeat_interleave(3, dim=1)
                    ],
                                      dim=-1)

                else:
                    pred_vis = th.cat([
                        micro['img_sr'], (pred['image_raw']), pred['image_sr'],
                        (pred_depth).repeat_interleave(3, dim=1)
                    ],
                                      dim=-1)
            else:
                pred_vis = th.cat([
                    self.pool_128(micro['img']), pred['image_raw'],
                    pred_depth.repeat_interleave(3, dim=1)
                ],
                                  dim=-1)  # B, 3, H, W

            vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
            vis = vis * 127.5 + 127.5
            vis = vis.clip(0, 255).astype(np.uint8)

            for j in range(vis.shape[0]):
                video_out.append_data(vis[j])

        video_out.close()

        val_scores_for_logging = calc_average_loss(all_loss_dict)
        with open(os.path.join(logger.get_dir(), 'scores.json'), 'a') as f:
            json.dump({'step': self.step, **val_scores_for_logging}, f)

        # * log to tensorboard
        for k, v in val_scores_for_logging.items():
            self.writer.add_scalar(f'Eval/Rec/{k}', v,
                                   self.step + self.resume_step)

        self.eval_novelview_loop()

    def _load_and_sync_parameters(self, model=None, model_name='ddpm'):
        resume_checkpoint, self.resume_step = find_resume_checkpoint(
            self.resume_checkpoint, model_name) or self.resume_checkpoint

        if model is None:
            model = self.model
        # print(resume_checkpoint)

        if resume_checkpoint and Path(resume_checkpoint).exists():
            if dist_util.get_rank() == 0:

                # ! rank 0 return will cause all other ranks to hang
                # if not Path(resume_checkpoint).exists():
                #     logger.log(
                #         f"failed to load model from checkpoint: {resume_checkpoint}, not exist"
                #     )
                #     return

                logger.log(
                    f"loading model from checkpoint: {resume_checkpoint}...")
                map_location = {
                    'cuda:%d' % 0: 'cuda:%d' % dist_util.get_rank()
                }  # configure map_location properly

                # print(f'mark {model_name} loading ', flush=True)
                resume_state_dict = dist_util.load_state_dict(
                    resume_checkpoint, map_location=map_location)
                # print(f'mark {model_name} loading finished', flush=True)

                model_state_dict = model.state_dict()

                for k, v in resume_state_dict.items():
                    if k in model_state_dict.keys() and v.size(
                    ) == model_state_dict[k].size():
                        model_state_dict[k] = v

                    # elif 'IN' in k and model_name == 'rec' and getattr(model.decoder, 'decomposed_IN', False):
                    #     model_state_dict[k.replace('IN', 'superresolution.norm.norm_layer')] = v # decomposed IN

                    else:
                        print('!!!! ignore key: ', k, ": ", v.size(),
                              'shape in model: ', model_state_dict[k].size())

                model.load_state_dict(model_state_dict, strict=True)
                del model_state_dict

        if dist_util.get_world_size() > 1:
            dist_util.sync_params(model.parameters())
            print(f'synced {model_name} params')

    def eval_ddpm_sample(self):

        args = dnnlib.EasyDict(
            dict(
                batch_size=1,
                image_size=224,
                denoise_in_channels=self.ddp_rec_model.module.decoder.
                triplane_decoder.out_chans,  # type: ignore
                clip_denoised=False,
                class_cond=False,
                use_ddim=False))

        model_kwargs = {}

        if args.class_cond:
            classes = th.randint(low=0,
                                 high=NUM_CLASSES,
                                 size=(args.batch_size, ),
                                 device=dist_util.dev())
            model_kwargs["y"] = classes

        diffusion = self.diffusion
        sample_fn = (diffusion.p_sample_loop
                     if not args.use_ddim else diffusion.ddim_sample_loop)

        for i in range(1):
            triplane_sample = sample_fn(
                self.ddp_ddpm_model,
                (args.batch_size, args.denoise_in_channels, args.image_size,
                 args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            th.cuda.empty_cache()
            self.render_video_given_triplane(
                triplane_sample,
                name_prefix=f'{self.step + self.resume_step}_{i}')
            th.cuda.empty_cache()

    @th.inference_mode()
    def render_video_given_triplane(self, planes, name_prefix='0'):

        planes *= self.triplane_scaling_divider  # if setting clip_denoised=True, the sampled planes will lie in [-1,1]. Thus, values beyond [+- std] will be abandoned in this version. Move to IN for later experiments.

        # st()

        # print(planes.min(), planes.max())

        # used during diffusion sampling inference
        video_out = imageio.get_writer(
            f'{logger.get_dir()}/triplane_{name_prefix}.mp4',
            mode='I',
            fps=60,
            codec='libx264')

        # for i in range(0, len(c_list), 1): # TODO, larger batch size for eval
        for i, batch in enumerate(tqdm(self.eval_data)):
            micro = {k: v.to(dist_util.dev()) for k, v in batch.items()}

            pred = self.ddp_rec_model(img=None,
                                      c=micro['c'],
                                      latent=planes,
                                      behaviour=self.render_latent_behaviour)

            # if True:
            pred_depth = pred['image_depth']
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())

            # if 'image_sr' in pred:
            #     pred_vis = th.cat([
            #         micro['img_sr'],
            #         self.pool_512(pred['image_raw']), pred['image_sr'],
            #         self.pool_512(pred_depth).repeat_interleave(3, dim=1)
            #     ],
            #                       dim=-1)
            # else:
            # ? why error

            pred_vis = th.cat([
                self.pool_128(micro['img']), pred['image_raw'],
                pred_depth.repeat_interleave(3, dim=1)
            ],
                              dim=-1)  # B, 3, H, W

            vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
            vis = vis * 127.5 + 127.5
            vis = vis.clip(0, 255).astype(np.uint8)

            for j in range(vis.shape[0]):
                video_out.append_data(vis[j])

        video_out.close()
        print('logged video to: ',
              f'{logger.get_dir()}/triplane_{name_prefix}.mp4')

    @th.inference_mode()
    def render_video_noise_schedule(self, name_prefix='0'):

        # planes *= self.triplane_std # denormalize for rendering

        video_out = imageio.get_writer(
            f'{logger.get_dir()}/triplane_visnoise_{name_prefix}.mp4',
            mode='I',
            fps=30,
            codec='libx264')

        for i, batch in enumerate(tqdm(self.eval_data)):
            micro = {k: v.to(dist_util.dev()) for k, v in batch.items()}

            if i % 10 != 0:
                continue

            # ========= novel view plane settings ====
            if i == 0:
                novel_view_micro = {
                    k: v[0:1].to(dist_util.dev()).repeat_interleave(
                        micro['img'].shape[0], 0)
                    for k, v in batch.items()
                }
            else:
                # if novel_view_micro['c'].shape[0] < micro['img'].shape[0]:
                novel_view_micro = {
                    k: v[0:1].to(dist_util.dev()).repeat_interleave(
                        micro['img'].shape[0], 0)
                    for k, v in novel_view_micro.items()
                }

            latent = self.ddp_rec_model(
                img=novel_view_micro['img_to_encoder'],
                c=micro['c'])[self.latent_name]  # pred: (B, 3, 64, 64)

            x_start = latent / self.triplane_scaling_divider  # normalize std to 1
            # x_start = latent

            all_pred_vis = []
            # for t in th.range(0,
            #                   4001,
            #                   500,
            #                   dtype=th.long,
            #                   device=dist_util.dev()):  # cosine 4k steps
            for t in th.range(0,
                              1001,
                              125,
                              dtype=th.long,
                              device=dist_util.dev()):  # cosine 4k steps

                # ========= add noise according to t
                noise = th.randn_like(x_start)  # x_start is the x0 image
                x_t = self.diffusion.q_sample(
                    x_start, t, noise=noise
                )  # * add noise according to predefined schedule
                planes_x_t = (x_t * self.triplane_scaling_divider).clamp(
                    -50, 50)  # de-scaling noised x_t

                # planes_x_t = (x_t * 1).clamp(
                #     -50, 50)  # de-scaling noised x_t

                # ===== visualize
                pred = self.ddp_rec_model(
                    img=None,
                    c=micro['c'],
                    latent=planes_x_t,
                    behaviour=self.render_latent_behaviour
                )  # pred: (B, 3, 64, 64)

                # pred_depth = pred['image_depth']
                # pred_depth = (pred_depth - pred_depth.min()) / (
                #     pred_depth.max() - pred_depth.min())
                # pred_vis = th.cat([
                #     # self.pool_128(micro['img']),
                #     pred['image_raw'],
                # ],
                #                   dim=-1)  # B, 3, H, W
                pred_vis = pred['image_raw']

                all_pred_vis.append(pred_vis)
                # TODO, make grid

            all_pred_vis = torchvision.utils.make_grid(
                th.cat(all_pred_vis, 0),
                nrow=len(all_pred_vis),
                normalize=True,
                value_range=(-1, 1),
                scale_each=True)  # normalized to [-1,1]

            vis = all_pred_vis.permute(1, 2, 0).cpu().numpy()  # H W 3

            vis = (vis * 255).clip(0, 255).astype(np.uint8)

            video_out.append_data(vis)

        video_out.close()
        print('logged video to: ',
              f'{logger.get_dir()}/triplane_visnoise_{name_prefix}.mp4')

        th.cuda.empty_cache()

    # def _update_ema(self):
    #     for rate, params in zip(self.ema_rate, self.ema_params):
    #         update_ema(params, self.mp_trainer.master_params, rate=rate)
    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples",
                     (self.step + self.resume_step + 1) * self.global_batch)

    def save(self, mp_trainer=None, model_name='ddpm'):
        if mp_trainer is None:
            mp_trainer = self.mp_trainer

        def save_checkpoint(rate, model, params, model_name):
            state_dict = mp_trainer.master_params_to_state_dict(params, model)
            if dist_util.get_rank() == 0:
                logger.log(f"saving model {model_name} {rate}...")

                if not rate:
                    filename = f"model_{model_name}{(self.step+self.resume_step):07d}.pt"
                else:
                    filename = f"ema_{model_name}_{rate}_{(self.step+self.resume_step):07d}.pt"

                with bf.BlobFile(bf.join(get_blob_logdir(), filename),
                                 "wb") as f:
                    th.save(state_dict, f)

        # ! save ddpm and rec individually
        save_checkpoint(0, self.mp_trainer.model.denoise_model,
                        list(self.mp_trainer.model.denoise_model.parameters()),
                        'ddpm')
        save_checkpoint(0, self.mp_trainer.model.rec_model,
                        list(self.mp_trainer.model.rec_model.parameters()),
                        'rec')

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(
                rate, self.mp_trainer.model, params,
                self.mp_trainer.model_name)  # save ema only for DDPM

        dist.barrier()

    def _load_optimizer_state(self):
        main_checkpoint, _ = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(bf.dirname(main_checkpoint),
                                 f"opt{self.resume_step:06}.pt")
        if bf.exists(opt_checkpoint):
            logger.log(
                f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)

    def _load_ema_parameters(self,
                             rate,
                             model=None,
                             mp_trainer=None,
                             model_name='ddpm'):

        if mp_trainer is None:
            mp_trainer = self.mp_trainer
        if model is None:
            model = self.model

        ema_params = copy.deepcopy(mp_trainer.master_params)
        # ema_params = copy.deepcopy(list(mp_trainer.model.denoise_model.parameters()))

        main_checkpoint, _ = find_resume_checkpoint(
            self.resume_checkpoint, model_name) or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step,
                                             rate, model_name)
        # logger.log('ema model to load: {}'.format(ema_checkpoint))
        if ema_checkpoint:

            if dist_util.get_rank() == 0:

                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")

                map_location = {
                    'cuda:%d' % 0: 'cuda:%d' % dist_util.get_rank()
                }  # configure map_location properly

                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=map_location)

                model_ema_state_dict = model.state_dict()

                for k, v in state_dict.items():
                    if k in model_ema_state_dict.keys() and v.size(
                    ) == model_ema_state_dict[k].size():
                        model_ema_state_dict[k] = v

                    else:
                        print('ignore key: ', k, ": ", v.size())

                ema_params = mp_trainer.state_dict_to_master_params(
                    model_ema_state_dict, model=model)

                del state_dict
                logger.log(
                    f"loading EMA from checkpoint finished: {ema_checkpoint}..."
                )

        if dist_util.get_world_size() > 1:
            dist_util.sync_params(ema_params)
            print(f'synced {model_name} ema_params')
        return ema_params

    # def _sample_nvs_pose(self, batch_size):

    #     device = dist_util.dev()

    #     fov_deg = 18.837  # for ffhq/afhq
    #     intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    #     all_nvs_params = []

    #     pitch_range = 0.25
    #     yaw_range = 0.35
    #     # num_keyframes = batch_size  # how many nv poses to sample from
    #     w_frames = 1

    #     cam_pivot = th.Tensor(
    #         self.G.rendering_kwargs.get('avg_camera_pivot')).to(device)
    #     cam_radius = self.G.rendering_kwargs.get('avg_camera_radius')

    #     for _ in range(batch_size):

    #         cam2world_pose = LookAtPoseSampler.sample(
    #             np.pi / 2,
    #             np.pi / 2,
    #             cam_pivot,
    #             horizontal_stddev=yaw_range,
    #             vertical_stddev=pitch_range,
    #             radius=cam_radius,
    #             device=device)

    #         camera_params = th.cat(
    #             [cam2world_pose.reshape(-1, 16),
    #              intrinsics.reshape(-1, 9)], 1)

    #         all_nvs_params.append(camera_params)

    #     return th.cat(all_nvs_params, dim=0)

    def forward_D(self, batch, behaviour):  # update D
        self.mp_trainer_canonical_cvD.zero_grad()
        self.mp_trainer_nvs_cvD.zero_grad()
        self.ddp_rec_model.requires_grad_(False)
        self.ddp_ddpm_model.requires_grad_(False)

        # update two D
        if behaviour == 'nvs':
            self.ddp_nvs_cvD.requires_grad_(True)
            self.ddp_canonical_cvD.requires_grad_(False)
        else:  # update rec canonical D
            self.ddp_nvs_cvD.requires_grad_(False)
            self.ddp_canonical_cvD.requires_grad_(True)

        batch_size = batch['img'].shape[0]

        # * sample a new batch for D training
        for i in range(0, batch_size, self.microbatch):
            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev()).contiguous()
                for k, v in batch.items()
            }

            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_canonical_cvD.use_amp):

                novel_view_c = th.cat([micro['c'][1:], micro['c'][:1]])

                latent = self.ddp_rec_model(img=micro['img_to_encoder'],
                                            behaviour='enc_dec_wo_triplane')

                cano_pred = self.ddp_rec_model(latent=latent,
                                               c=micro['c'],
                                               behaviour='triplane_dec')

                # TODO, optimize with one encoder, and two triplane decoder
                if behaviour == 'rec':

                    if 'image_sr' in cano_pred:
                        d_loss_cano = self.run_D_Diter(
                            # real=micro['img_sr'],
                            # fake=cano_pred['image_sr'],
                            real=0.5 * micro['img_sr'] +
                            0.5 * th.nn.functional.interpolate(
                                micro['img'],
                                size=micro['img_sr'].shape[2:],
                                mode='bilinear'),
                            fake=0.5 * cano_pred['image_sr'] +
                            0.5 * th.nn.functional.interpolate(
                                cano_pred['image_raw'],
                                size=cano_pred['image_sr'].shape[2:],
                                mode='bilinear'),
                            D=self.ddp_canonical_cvD)  # TODO, add SR for FFHQ
                    else:
                        d_loss_cano = self.run_D_Diter(
                            real=micro['img'],
                            fake=cano_pred['image_raw'],
                            D=self.ddp_canonical_cvD)  # TODO, add SR for FFHQ

                    log_rec3d_loss_dict(
                        {'vision_aided_loss/D_cano': d_loss_cano})
                    self.mp_trainer_canonical_cvD.backward(d_loss_cano)
                else:
                    assert behaviour == 'nvs'

                    nvs_pred = self.ddp_rec_model(latent=latent,
                                                  c=novel_view_c,
                                                  behaviour='triplane_dec')

                    if 'image_sr' in nvs_pred:
                        d_loss_nvs = self.run_D_Diter(
                            # real=cano_pred['image_sr'],
                            # fake=nvs_pred['image_sr'],
                            real=0.5 * cano_pred['image_sr'] +
                            0.5 * th.nn.functional.interpolate(
                                cano_pred['image_raw'],
                                size=cano_pred['image_sr'].shape[2:],
                                mode='bilinear'),
                            fake=0.5 * nvs_pred['image_sr'] +
                            0.5 * th.nn.functional.interpolate(
                                nvs_pred['image_raw'],
                                size=nvs_pred['image_sr'].shape[2:],
                                mode='bilinear'),
                            D=self.ddp_nvs_cvD)  # TODO, add SR for FFHQ
                    else:
                        d_loss_nvs = self.run_D_Diter(
                            real=cano_pred['image_raw'],
                            fake=nvs_pred['image_raw'],
                            D=self.ddp_nvs_cvD)  # TODO, add SR for FFHQ

                    log_rec3d_loss_dict(
                        {'vision_aided_loss/D_nvs': d_loss_nvs})
                    self.mp_trainer_nvs_cvD.backward(d_loss_nvs)


class TrainLoop3DDiffusionSingleStagecvDSDS_sdswithrec(
        TrainLoop3DDiffusionSingleStagecvDSDS):

    def __init__(self,
                 *,
                 rec_model,
                 denoise_model,
                 diffusion,
                 loss_class,
                 data,
                 eval_data,
                 batch_size,
                 microbatch,
                 lr,
                 ema_rate,
                 log_interval,
                 eval_interval,
                 save_interval,
                 resume_checkpoint,
                 use_fp16=False,
                 fp16_scale_growth=0.001,
                 schedule_sampler=None,
                 weight_decay=0,
                 lr_anneal_steps=0,
                 iterations=10001,
                 ignore_resume_opt=False,
                 freeze_ae=False,
                 denoised_ae=True,
                 triplane_scaling_divider=10,
                 use_amp=False,
                 **kwargs):
        super().__init__(rec_model=rec_model,
                         denoise_model=denoise_model,
                         diffusion=diffusion,
                         loss_class=loss_class,
                         data=data,
                         eval_data=eval_data,
                         batch_size=batch_size,
                         microbatch=microbatch,
                         lr=lr,
                         ema_rate=ema_rate,
                         log_interval=log_interval,
                         eval_interval=eval_interval,
                         save_interval=save_interval,
                         resume_checkpoint=resume_checkpoint,
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         schedule_sampler=schedule_sampler,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         ignore_resume_opt=ignore_resume_opt,
                         freeze_ae=freeze_ae,
                         denoised_ae=denoised_ae,
                         triplane_scaling_divider=triplane_scaling_divider,
                         use_amp=use_amp,
                         **kwargs)

    def run_loop(self, batch=None):
        th.cuda.empty_cache()
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            # let all processes sync up before starting with a new epoch of training
            dist_util.synchronize()

            # if self.step % self.eval_interval == 0 and self.step != 0:
            if self.step % self.eval_interval == 0:
                if dist_util.get_rank() == 0:
                    self.eval_ddpm_sample()
                    self.eval_loop()
                    self.eval_novelview_loop()

                # let all processes sync up before starting with a new epoch of training
                dist_util.synchronize()
                th.cuda.empty_cache()
            
            # ! pure diffusion step (to accelerate training, encoder fixed)

            batch1 = next(self.data)
            batch2 = next(self.data)
            batch = {} # double bs for pure diffusion training
            for k, v in batch1.items():
                batch[k] = th.cat([batch1[k], batch2[k]], dim=0)
            self.run_step(batch, step='diffusion_step')

            # ! diffusion + rec step
            batch = next(self.data)
            self.run_step(batch, step='diffusion_step_rec')

            if self.step % 2 == 0:
                batch = next(self.data)
                self.run_step(batch, step='d_step_rec')

            batch = next(self.data)
            self.run_step(batch, step='diffusion_step_nvs')

            if self.step % 2 == 1: # 
                batch = next(self.data)
                self.run_step(batch, step='d_step_nvs')

            # batch = next(self.data)
            # self.run_step(batch, step='sds')

            if self.step % self.log_interval == 0 and dist_util.get_rank(
            ) == 0:
                out = logger.dumpkvs()
                # * log to tensorboard
                for k, v in out.items():
                    self.writer.add_scalar(f'Loss/{k}', v,
                                           self.step + self.resume_step)

            if self.step % self.save_interval == 0 and self.step != 0:
                self.save(self.mp_trainer, )
                if not self.freeze_ae:
                    self.save(self.mp_trainer, 'rec')

                self.save(self.mp_trainer_nvs_cvD, self.mp_trainer_nvs_cvD.model_name)
                self.save(self.mp_trainer_canonical_cvD, self.mp_trainer_canonical_cvD.model_name)

                dist_util.synchronize()

                th.cuda.empty_cache()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST",
                                  "") and self.step > 0:
                    return

            self.step += 1

            if self.step > self.iterations:
                print('reached maximum iterations, exiting')

                # Save the last checkpoint if it wasn't already saved.
                if (self.step - 1) % self.save_interval != 0:
                    self.save()

                    self.save(self.mp_trainer_nvs_cvD, self.mp_trainer_nvs_cvD.model_name)
                    self.save(self.mp_trainer_canonical_cvD, self.mp_trainer_canonical_cvD.model_name)

                    if not self.freeze_ae:
                        self.save(self.mp_trainer, 'rec')

                exit()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            if not self.freeze_ae:
                self.save(self.mp_trainer, 'rec')

    def forward_diffusion(self, batch, behaviour='rec', *args, **kwargs):
        """
        add sds grad to all ae predicted x_0 
        """

        self.ddp_canonical_cvD.requires_grad_(False)
        self.ddp_nvs_cvD.requires_grad_(False)

        self.ddp_ddpm_model.requires_grad_(True)
        self.ddp_rec_model.requires_grad_(True)

        # if behaviour != 'diff' and 'rec' in behaviour:
        # if behaviour != 'diff' and 'rec' in behaviour: # pure diffusion step
        #     self.ddp_rec_model.requires_grad_(True)
        for param in self.ddp_rec_model.module.decoder.triplane_decoder.parameters(
        ):  # type: ignore
            param.requires_grad_(False) # ! disable triplane_decoder grad in each iteration indepenently; 
        # else:

        self.mp_trainer.zero_grad()

        # ! no 'sds' step now, both add sds grad back to ViT

        assert behaviour != 'sds'
        # if behaviour == 'sds':
        # else:
        #     self.ddp_ddpm_model.requires_grad_(True)

        batch_size = batch['img'].shape[0]

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev())
                for k, v in batch.items()
            }

            last_batch = (i + self.microbatch) >= batch_size

            ae_loss = th.tensor(0.0).to(dist_util.dev())
            vision_aided_loss = th.tensor(0.0).to(dist_util.dev())
            denoise_loss = th.tensor(0.0).to(dist_util.dev())
            d_weight = th.tensor(0.0).to(dist_util.dev())

            # =================================== ae part ===================================
            with th.cuda.amp.autocast(dtype=th.float16,
                                      enabled=self.mp_trainer.use_amp
                                      and not self.freeze_ae):

                latent = self.ddp_rec_model(
                    img=micro['img_to_encoder'],
                    c=micro['c'],
                    behaviour='enc_dec_wo_triplane')  # pred: (B, 3, 64, 64)

                if behaviour == 'rec':
                    target = micro
                    pred = self.ddp_rec_model(latent=latent,
                                              c=micro['c'],
                                              behaviour='triplane_dec')

                    if last_batch or not self.use_ddp:
                        ae_loss, loss_dict = self.loss_class(pred,
                                                             target,
                                                             test_mode=False)
                    else:
                        with self.ddp_model.no_sync():  # type: ignore
                            ae_loss, loss_dict = self.loss_class(
                                pred, target, test_mode=False)

                    # last_layer = self.ddp_rec_model.module.decoder.triplane_decoder.decoder.net[  # type: ignore
                    #     -1].weight  # type: ignore

                    if 'image_sr' in pred:
                        vision_aided_loss = self.ddp_canonical_cvD(
                            0.5 * pred['image_sr'] +
                            0.5 * th.nn.functional.interpolate(
                                pred['image_raw'],
                                size=pred['image_sr'].shape[2:],
                                mode='bilinear'),
                            for_G=True).mean()  # [B, 1] shape
                    else:
                        vision_aided_loss = self.ddp_canonical_cvD(
                            pred['image_raw'], for_G=True
                        ).mean(
                        )   # [B, 1] shape

                    # d_weight = calculate_adaptive_weight(
                    #     ae_loss,
                    #     vision_aided_loss,
                    #     last_layer,
                    #     # disc_weight_max=1) * 1
                    #     disc_weight_max=1) * self.loss_class.opt.rec_cvD_lambda
                    d_weight = self.loss_class.opt.rec_cvD_lambda # since decoder is fixed here. set to 0.001
                    
                    vision_aided_loss *= d_weight

                    # d_weight = self.loss_class.opt.rec_cvD_lambda
                    loss_dict.update({
                        'vision_aided_loss/G_rec':
                        vision_aided_loss,
                        # 'd_weight_G_rec':
                        # d_weight,
                    })

                    log_rec3d_loss_dict(loss_dict)

                elif behaviour == 'nvs':

                    novel_view_c = th.cat([micro['c'][1:], micro['c'][:1]])

                    pred = self.ddp_rec_model(latent=latent,
                                              c=novel_view_c,
                                              behaviour='triplane_dec')

                    if 'image_sr' in pred:
                        vision_aided_loss = self.ddp_nvs_cvD(
                            # pred_for_rec['image_sr'],
                            0.5 * pred['image_sr'] +
                            0.5 * th.nn.functional.interpolate(
                                pred['image_raw'],
                                size=pred['image_sr'].shape[2:],
                                mode='bilinear'),
                            for_G=True).mean()  # [B, 1] shape
                    else:
                        vision_aided_loss = self.ddp_nvs_cvD(
                            pred['image_raw'], for_G=True
                        ).mean(
                        )  # [B, 1] shape

                    d_weight = self.loss_class.opt.nvs_cvD_lambda
                    vision_aided_loss *= d_weight

                    log_rec3d_loss_dict({
                        'vision_aided_loss/G_nvs':
                        vision_aided_loss,
                    })

                    # ae_loss = th.tensor(0.0).to(dist_util.dev())

                elif behaviour == 'diff':
                    self.ddp_rec_model.requires_grad_(False)
                    # assert self.ddp_rec_model.module.requires_grad == False, 'freeze ddpm_rec for pure diff step'
                else:
                    raise NotImplementedError(behaviour)
                #     assert behaviour == 'sds'

                # pred = None

                # if behaviour != 'sds': # also train diffusion
                # assert pred is not None

                # TODO, train diff and sds together, available?
                micro_to_denoise = latent[
                    self.
                    latent_name] / self.triplane_scaling_divider  # ! detach() from vit computational graph

                # if behaviour != 'sds':
                micro_to_denoise.detach_()

                t, weights = self.schedule_sampler.sample(
                    micro_to_denoise.shape[0], dist_util.dev())

                model_kwargs = {}

                # print(micro_to_denoise.min(), micro_to_denoise.max())
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_ddpm_model,
                    micro_to_denoise,  # x_start
                    t,
                    model_kwargs=model_kwargs,
                    return_detail=True)

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

                x_t = losses.pop('x_t')
                model_output = losses.pop('model_output')
                diffusion_target = losses.pop('diffusion_target')
                alpha_bar = losses.pop('alpha_bar')

                # ! prepare for later single-stage training
                # denoised_fn = functools.partial(
                #     self.diffusion.p_mean_variance,
                #     self.ddp_ddpm_model,
                #     x_t,  # x_start
                #     t,
                #     model_kwargs=model_kwargs)

                log_loss_dict(self.diffusion, t,
                              {k: v * weights
                               for k, v in losses.items()})

                # if behaviour == 'sds':
                # ! calculate sds grad, and add to the grad of

                if 'rec' in behaviour and self.loss_class.opt.sds_lamdba > 0:  # only enable sds along with rec step
                    w = (
                        1 - alpha_bar**2
                    ) / self.triplane_scaling_divider * self.loss_class.opt.sds_lamdba  # https://github.com/ashawkey/stable-dreamfusion/issues/106
                    sds_grad = denoise_loss.clone().detach(
                    ) * w  # * https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html. detach() returned Tensor share the same storage with previous one. add clone() here.

                    # ae_loss = AddGradient.apply(latent[self.latent_name], sds_grad) # add sds_grad during backward

                    def sds_hook(grad_to_add):

                        def modify_grad(grad):
                            return grad + grad_to_add  # add the sds grad to the original grad for BP

                        return modify_grad

                    latent[self.latent_name].register_hook(
                        sds_hook(sds_grad))  # merge sds grad with rec/nvs ae step

                loss = ae_loss + denoise_loss + vision_aided_loss  # caluclate loss within AMP

            # ! cvD loss

            # exit AMP before backward
            self.mp_trainer.backward(loss)

            # TODO, merge visualization with original AE
            # =================================== denoised AE log part ===================================

            if dist_util.get_rank() == 0 and self.step % 500 == 0 and behaviour != 'diff':
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

                    # if denoised_out is None:
                    # if not self.denoised_ae:
                    # denoised_out = denoised_fn()

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

                    # if not self.denoised_ae:
                    #     denoised_ae_pred = self.ddp_rec_model(
                    #         img=None,
                    #         c=micro['c'][0:1],
                    #         latent=denoised_out['pred_xstart'][0:1] * self.
                    #         triplane_scaling_divider,  # TODO, how to define the scale automatically
                    #         behaviour=self.render_latent_behaviour)
                    # else:
                    #     assert denoised_ae_pred is not None
                    #     denoised_ae_pred['image_raw'] = denoised_ae_pred[
                    #         'image_raw'][0:1]

                    # print(pred_img.shape)
                    # print('denoised_ae:', self.denoised_ae)

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
                        f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t[0].item()}_{behaviour}.jpg'
                    )
                    print(
                        'log denoised vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t[0].item()}_{behaviour}.jpg'
                    )

                    th.cuda.empty_cache()

    @th.inference_mode()
    def render_video_given_triplane(self, planes, name_prefix='0', save_img=False):

        planes *= self.triplane_scaling_divider  # if setting clip_denoised=True, the sampled planes will lie in [-1,1]. Thus, values beyond [+- std] will be abandoned in this version. Move to IN for later experiments.

        sr_w_code = getattr(self.ddp_rec_model.module.decoder,'w_avg', None)
        batch_size = planes.shape[0]

        if sr_w_code is not None:
            sr_w_code= sr_w_code.reshape(1,1,-1).repeat_interleave(batch_size, 0)

        # used during diffusion sampling inference
        video_out = imageio.get_writer(
            f'{logger.get_dir()}/triplane_{name_prefix}.mp4',
            mode='I',
            fps=60,
            codec='libx264')

        # for i in range(0, len(c_list), 1): # TODO, larger batch size for eval
        for i, batch in enumerate(tqdm(self.eval_data)):
            micro = {k: v.to(dist_util.dev()).repeat_interleave(batch_size, 0) for k, v in batch.items()}
            # micro = {'c': batch['c'].to(dist_util.dev()).repeat_interleave(batch_size, 0)}

            pred = self.ddp_rec_model(img=None,
                                      c=micro['c'],
                                      latent={
                                          'latent_normalized': planes,
                                          'sr_w_code': sr_w_code
                                      },
                                      behaviour=self.render_latent_behaviour)

            # if True:
            pred_depth = pred['image_depth']
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())

            if 'image_sr' in pred:

                gen_img = pred['image_sr']

                if pred['image_sr'].shape[-1] == 512:

                    pred_vis = th.cat([
                        micro['img_sr'],
                        self.pool_512(pred['image_raw']), gen_img,
                        self.pool_512(pred_depth).repeat_interleave(3, dim=1)
                    ],
                                      dim=-1)

                elif pred['image_sr'].shape[-1] == 256:

                    pred_vis = th.cat([
                        micro['img_sr'],
                        self.pool_256(pred['image_raw']), pred['image_sr'],
                        self.pool_256(pred_depth).repeat_interleave(3, dim=1)
                    ],
                                      dim=-1)

            else:
                gen_img = pred['image_raw']

                pred_vis = th.cat([
                    # self.pool_128(micro['img']), 
                    gen_img,
                    pred_depth.repeat_interleave(3, dim=1)
                ],
                                dim=-1)  # B, 3, H, W


            if save_img:
                for batch_idx in range(gen_img.shape[0]):
                    sampled_img = Image.fromarray((gen_img[batch_idx].permute(1,2,0).cpu().numpy() * 127.5 + 127.5).clip(0, 255).astype(np.uint8))
                    if sampled_img.size != (512, 512):
                        sampled_img = sampled_img.resize((128, 128), Image.HAMMING) # for shapenet
                    sampled_img.save(logger.get_dir() + '/FID_Cals/{}_{}.png'.format(int(name_prefix)*batch_size+batch_idx, i))
                    # print('FID_Cals/{}_{}.png'.format(int(name_prefix)*batch_size+batch_idx, i))

            vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
            vis = vis * 127.5 + 127.5
            vis = vis.clip(0, 255).astype(np.uint8)
            if vis.shape[0] > 1:
                vis = np.concatenate(np.split(vis, vis.shape[0], axis=0), axis=-3)


            for j in range(vis.shape[0]):
                video_out.append_data(vis[j])

        video_out.close()
        print('logged video to: ',
              f'{logger.get_dir()}/triplane_{name_prefix}.mp4')
