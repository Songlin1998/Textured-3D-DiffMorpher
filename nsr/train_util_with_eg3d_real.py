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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.nn import update_ema
from guided_diffusion.resample import LossAwareSampler, UniformSampler
from guided_diffusion.train_util import (calc_average_loss,
                                         find_ema_checkpoint,
                                         find_resume_checkpoint,
                                         get_blob_logdir, log_rec3d_loss_dict,
                                         parse_resume_step_from_filename)

import vision_aided_loss
from .train_util_with_eg3d import TrainLoop3DRecEG3D
from .train_util import TrainLoop3DRec
from .cvD.nvsD_canoD import TrainLoop3DcvD_nvsD_canoD
# from .train_util_cvD import TrainLoop3DcvD
from dnnlib.util import calculate_adaptive_weight


# class TrainLoop3DRecEG3DHybrid(TrainLoop3DRecEG3D, TrainLoop3DcvD):
# class TrainLoop3DRecEG3DHybrid(TrainLoop3DRecEG3D, TrainLoop3DcvD_nvsD_canoD):
class TrainLoop3DRecEG3DReal(TrainLoop3DRecEG3D, TrainLoop3DcvD_nvsD_canoD):

    def __init__(self,
                 *,
                 G,
                 model,
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
                 weight_decay=0,
                 lr_anneal_steps=0,
                 iterations=10001,
                 load_submodule_name='',
                 ignore_resume_opt=False,
                 model_name='rec',
                 use_amp=False,
                 hybrid_training=False,
                 **kwargs):

        super().__init__(G=G,
                         model=model,
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
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         load_submodule_name=load_submodule_name,
                         ignore_resume_opt=ignore_resume_opt,
                         model_name=model_name,
                         use_amp=use_amp,
                         hybrid_training=hybrid_training,
                         **kwargs)

        th.cuda.empty_cache()

    def run_loop(self):
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            # let all processes sync up before starting with a new epoch of training
            dist_util.synchronize()

            # batch, cond = next(self.data)
            # if batch is None:

            # batch = next(self.data)
            # self.run_step(batch, 'g_step_rec_eg3d')

            batch = next(self.data)
            self.run_step(batch, 'g_step_rec')

            # batch = next(self.data)
            # self.run_step(batch, 'd_step_rec')

            # batch = next(self.data)
            # self.run_step(batch, 'g_step_nvs')

            # batch = next(self.data)
            # self.run_step(batch, 'd_step_nvs')

            if self.step % self.log_interval == 0 and dist_util.get_rank(
            ) == 0:
                out = logger.dumpkvs()
                # * log to tensorboard
                for k, v in out.items():
                    self.writer.add_scalar(f'Loss/{k}', v,
                                           self.step + self.resume_step)

            # if self.step % self.eval_interval == 0 and self.step != 0:
            if self.step % self.eval_interval == 0:
                if dist_util.get_rank() == 0:
                    self.eval_loop()
                    # self.eval_novelview_loop()
                    # let all processes sync up before starting with a new epoch of training
                th.cuda.empty_cache()
                dist_util.synchronize()

            if self.step % self.save_interval == 0:
                self.save()
                self.save(self.mp_trainer_cvD, 'cvD')
                self.save(self.mp_trainer_canonical_cvD, 'cano_cvD')

                dist_util.synchronize()
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
                    self.save(self.mp_trainer_cvD, 'cvD')
                    self.save(self.mp_trainer_canonical_cvD, 'cano_cvD')

                exit()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.save(self.mp_trainer_canonical_cvD, 'cvD')

    def run_step(self, batch, step='g_step'):

        # if step == 'g_step_rec_eg3d':
        #     self.forward_G_rec_eg3d(batch)
        #     took_step_g_rec = self.mp_trainer.optimize(self.opt)

        #     if took_step_g_rec:
        #         self._update_ema()  # g_ema

        if step == 'g_step_rec':
            self.forward_G_rec(batch)
            took_step_g_rec = self.mp_trainer.optimize(self.opt)

            if took_step_g_rec:
                self._update_ema()  # g_ema

        # elif step == 'd_step_rec':
        #     self.forward_D(batch, behaviour='rec')
        #     _ = self.mp_trainer_canonical_cvD.optimize(self.opt_cano_cvD)

        if step == 'g_step_nvs':
            self.forward_G_nvs(batch)
            took_step_g_nvs = self.mp_trainer_rec.optimize(self.opt)

            if took_step_g_nvs:
                self._update_ema()  # g_ema

        # elif step == 'd_step_nvs':
        #     self.forward_D(batch, behaviour='nvs')
        #     _ = self.mp_trainer_cvD.optimize(self.opt_cvD)

        self._anneal_lr()
        self.log_step()

    def forward_G_nvs(self, batch):  # update G
        """send geometry into the D
        """

        self.mp_trainer_rec.zero_grad()
        self.rec_model.requires_grad_(True)

        self.ddp_cano_cvD.requires_grad_(False)
        self.ddp_nvs_cvD.requires_grad_(False)  # only use novel view D

        batch_size = batch['img'].shape[0]

        for i in range(0, batch_size, self.microbatch):
            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev()).contiguous()
                for k, v in batch.items()
            }

            st()

            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_rec.use_amp):

                pred_nv = self.rec_model(
                    img=micro['img_to_encoder'],
                    c=th.cat([
                        micro['c'][1:],
                        micro['c'][:1],
                    ]), return_raw_only=True)  # ! render novel views only for D loss

                # add cvD supervision

                # if 'image_sr' in pred_nv:
                #     vision_aided_loss = self.ddp_cvD(
                #         # pred_nv['image_sr'], 
                #         0.5 * pred_nv['image_sr'] + 0.5 * th.nn.functional.interpolate(pred_nv['image_raw'], size=pred_nv['image_sr'].shape[2:], mode='bilinear'),
                #         for_G=True).mean()  # ! for debugging
                #         # pred_nv['image_raw'], for_G=True).mean()  # [B, 1] shape
                # else:
                vision_aided_loss = self.ddp_nvs_cvD(
                    pred_nv['image_raw'], for_G=True).mean()  # [B, 1] shape
                    # pred_nv['image_depth'].repeat_interleave(3, dim=1), for_G=True).mean()  # [B, 1] shape

                loss = vision_aided_loss * self.loss_class.opt.nvs_cvD_lambda

                log_rec3d_loss_dict({
                    'vision_aided_loss/G_nvs':
                    vision_aided_loss,
                })

            self.mp_trainer_rec.backward(loss)

            for name, p in self.rec_model.named_parameters():
                if p.grad is None:
                    print(f"found rec unused param: {name}")

            # ! move to other places, add tensorboard

            if dist_util.get_rank() == 0 and self.step % 500 == 0:
                with th.no_grad():
                    # gt_vis = th.cat([batch['img'], batch['depth']], dim=-1)

                    gt_depth = micro['depth']
                    if gt_depth.ndim == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() -
                                                              gt_depth.min())
                    # if True:
                    pred_depth = pred_nv['image_depth']
                    pred_depth = (pred_depth - pred_depth.min()) / (
                        pred_depth.max() - pred_depth.min())
                    pred_img = pred_nv['image_raw']
                    gt_img = micro['img']

                    if 'image_sr' in pred_nv:

                        if pred_nv['image_sr'].shape[-1] == 512:
                            pred_img = th.cat(
                                [self.pool_512(pred_img), pred_nv['image_sr']],
                                dim=-1)
                            gt_img = th.cat(
                                [self.pool_512(micro['img']), micro['img_sr']],
                                dim=-1)
                            pred_depth = self.pool_512(pred_depth)
                            gt_depth = self.pool_512(gt_depth)

                        elif pred_nv['image_sr'].shape[-1] == 256:
                            pred_img = th.cat(
                                [self.pool_256(pred_img), pred_nv['image_sr']],
                                dim=-1)
                            gt_img = th.cat(
                                [self.pool_256(micro['img']), micro['img_sr']],
                                dim=-1)
                            pred_depth = self.pool_256(pred_depth)
                            gt_depth = self.pool_256(gt_depth)

                        else:
                            pred_img = th.cat(
                                [pred_img, pred_nv['image_sr']],
                                dim=-1)
                            gt_img = th.cat(
                                [micro['img'], micro['img_sr']],
                                dim=-1)

                    gt_vis = th.cat(
                        [gt_img, gt_depth.repeat_interleave(3, dim=1)],
                        dim=-1)  # TODO, fail to load depth. range [0, 1]

                    pred_vis = th.cat(
                        [pred_img,
                         pred_depth.repeat_interleave(3, dim=1)],
                        dim=-1)  # B, 3, H, W

                    # vis = th.cat([gt_vis, pred_vis], dim=-2)[0].permute(
                    #     1, 2, 0).cpu()  # ! pred in range[-1, 1]
                    vis = th.cat([gt_vis, pred_vis], dim=-2)

                    vis = torchvision.utils.make_grid(
                        vis,
                        normalize=True,
                        scale_each=True,
                        value_range=(-1, 1)).cpu().permute(1, 2, 0)  # H W 3
                    vis = vis.numpy() * 255
                    vis = vis.clip(0, 255).astype(np.uint8)

                    # print(vis.shape)

                    Image.fromarray(vis).save(
                        f'{logger.get_dir()}/{self.step+self.resume_step}_nvs.jpg'
                    )
                    print(
                        'log vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}_nvs.jpg'
                    )

    def forward_D(self, batch, behaviour):  # update D
        self.mp_trainer_canonical_cvD.zero_grad()
        self.mp_trainer_cvD.zero_grad()

        self.rec_model.requires_grad_(False)
        # self.ddp_model.requires_grad_(False)

        # update two D
        if behaviour == 'nvs':
            self.ddp_nvs_cvD.requires_grad_(True)
            self.ddp_cano_cvD.requires_grad_(False)
        else:  # update rec canonical D
            self.ddp_nvs_cvD.requires_grad_(False)
            self.ddp_cano_cvD.requires_grad_(True)

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

                latent = self.rec_model(img=micro['img_to_encoder'],
                                        behaviour='enc_dec_wo_triplane')



                # TODO, optimize with one encoder, and two triplane decoder
                if behaviour == 'rec':

                    cano_pred = self.rec_model(latent=latent,
                                            c=micro['c'],
                                            behaviour='triplane_dec')

                    if 'image_sr' in cano_pred:
                        d_loss_cano = self.run_D_Diter(
                            # real=micro['img_sr'],
                            # fake=cano_pred['image_sr'],
                            real=0.5 * micro['img_sr'] + 0.5 * th.nn.functional.interpolate(micro['img'], size=micro['img_sr'].shape[2:], mode='bilinear'),
                            fake=0.5 * cano_pred['image_sr'] + 0.5 * th.nn.functional.interpolate(cano_pred['image_raw'], size=cano_pred['image_sr'].shape[2:], mode='bilinear'),
                            D=self.ddp_cano_cvD)  # TODO, add SR for FFHQ
                    else:
                        d_loss_cano = self.run_D_Diter(
                            real=micro['img'],
                            fake=cano_pred['image_raw'],
                            D=self.ddp_cano_cvD)  # TODO, add SR for FFHQ

                    log_rec3d_loss_dict(
                        {'vision_aided_loss/D_cano': d_loss_cano})
                    self.mp_trainer_canonical_cvD.backward(d_loss_cano)

                    for name, p in self.rec_model.named_parameters():
                        if p.grad is None:
                            print(f"found rec unused param: {name}")


                else:
                    assert behaviour == 'nvs'

                    nvs_pred = self.rec_model(latent=latent,
                                              c=novel_view_c,
                                              behaviour='triplane_dec')

                    # if 'image_sr' in nvs_pred:
                    #     d_loss_nvs = self.run_D_Diter(
                    #         # real=cano_pred['image_sr'],
                    #         # fake=nvs_pred['image_sr'],
                    #         real=0.5 * cano_pred['image_sr'] + 0.5 * th.nn.functional.interpolate(cano_pred['image_raw'], size=cano_pred['image_sr'].shape[2:], mode='bilinear'),
                    #         fake=0.5 * nvs_pred['image_sr'] + 0.5 * th.nn.functional.interpolate(nvs_pred['image_raw'], size=nvs_pred['image_sr'].shape[2:], mode='bilinear'),
                    #         D=self.ddp_cvD)  # TODO, add SR for FFHQ
                    # else:

                    micro = {'c': batch['c'].to(dist_util.dev())}

                    with th.no_grad():  # * infer gt
                        eg3d_batch, ws = self.run_G(
                            z=th.randn(micro['c'].shape[0], 512).to(dist_util.dev()),
                            c=micro['c'].to(dist_util.dev(
                            )),  # use real img pose here? or synthesized pose.
                            swapping_prob=0,
                            neural_rendering_resolution=128)

                    # micro.update({
                    #     'img': eg3d_batch['image_raw'],  # gt
                    #     'img_to_encoder': self.pool_224(eg3d_batch['image']),
                    #     'depth': eg3d_batch['image_depth'],
                    #     'img_sr': eg3d_batch['image'],
                    # })


                    d_loss_nvs = self.run_D_Diter(
                        # real=cano_pred['image_raw'],
                        # fake=nvs_pred['image_raw'],
                        real=eg3d_batch['image_depth'].repeat_interleave(3, dim=1),
                        fake=nvs_pred['image_depth'].repeat_interleave(3, dim=1),
                        D=self.ddp_nvs_cvD)  # TODO, add SR for FFHQ

                    log_rec3d_loss_dict(
                        {'vision_aided_loss/D_nvs': d_loss_nvs})
                    self.mp_trainer_cvD.backward(d_loss_nvs)

                    for name, p in self.rec_model.named_parameters():
                        if p.grad is None:
                            print(f"found rec unused param: {name}")

    def forward_G_rec_eg3d(self, batch, *args, **kwargs):

        self.mp_trainer_rec.zero_grad()
        self.rec_model.requires_grad_(True)

        self.ddp_cano_cvD.requires_grad_(False)
        self.ddp_nvs_cvD.requires_grad_(False)

        batch_size = batch['c'].shape[0]

        for i in range(0, batch_size, self.microbatch):

            micro = {'c': batch['c'].to(dist_util.dev())}

            with th.no_grad():  # * infer gt
                eg3d_batch, ws = self.run_G(
                    z=th.randn(micro['c'].shape[0], 512).to(dist_util.dev()),
                    c=micro['c'].to(dist_util.dev(
                    )),  # use real img pose here? or synthesized pose.
                    swapping_prob=0,
                    neural_rendering_resolution=128)

            micro.update({
                'img': eg3d_batch['image_raw'],  # gt
                'img_to_encoder': self.pool_224(eg3d_batch['image']),
                'depth': eg3d_batch['image_depth'],
                'img_sr': eg3d_batch['image'],
            })

            last_batch = (i + self.microbatch) >= batch_size

            # wrap forward within amp
            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=False):
                            #  enabled=self.mp_trainer.use_amp):

                pred_gen_output = self.rec_model(
                    img=micro['img_to_encoder'],  # pool from 512
                    c=micro['c'])  # pred: (B, 3, 64, 64)

                # target = micro
                target = dict(
                    img=eg3d_batch['image_raw'],
                    shape_synthesized=eg3d_batch['shape_synthesized'],
                    img_sr=eg3d_batch['image'],
                )

                pred_gen_output['shape_synthesized_query'] = {
                    'coarse_densities':
                    pred_gen_output['shape_synthesized']['coarse_densities'],
                    'image_depth':
                    pred_gen_output['image_depth'],
                }

                eg3d_batch['shape_synthesized']['image_depth'] = eg3d_batch[
                    'image_depth']

                batch_size, num_rays, _, _ = pred_gen_output[
                    'shape_synthesized']['coarse_densities'].shape

                for coord_key in ['fine_coords']:  # TODO add surface points

                    sigma = self.rec_model(
                        latent=pred_gen_output['latent_denormalized'],
                        coordinates=eg3d_batch['shape_synthesized'][coord_key],
                        directions=th.randn_like(
                            eg3d_batch['shape_synthesized'][coord_key]),
                        behaviour='triplane_renderer',
                    )['sigma']

                    rendering_kwargs = self.rec_model(
                        behaviour='get_rendering_kwargs')

                    sigma = sigma.reshape(
                        batch_size, num_rays,
                        rendering_kwargs['depth_resolution_importance'], 1)

                    pred_gen_output['shape_synthesized_query'][
                        f"{coord_key.split('_')[0]}_densities"] = sigma

                # * 2D reconstruction loss
                if last_batch or not self.use_ddp:
                    loss, loss_dict = self.loss_class(pred_gen_output,
                                                      target,
                                                      test_mode=False)
                else:
                    with self.rec_model.no_sync():  # type: ignore
                        loss, loss_dict = self.loss_class(pred_gen_output,
                                                          target,
                                                          test_mode=False)

                # * fully mimic 3D geometry output

                loss_shape = self.calc_shape_rec_loss(
                    pred_gen_output['shape_synthesized_query'],
                    eg3d_batch['shape_synthesized'])

                loss += loss_shape.mean()

                # * add feature loss on feature_image
                loss_feature_volume = th.nn.functional.mse_loss(
                    eg3d_batch['feature_volume'],
                    pred_gen_output['feature_volume'])
                loss += loss_feature_volume * 0.1

                # * add ws prediction loss
                loss_ws = th.nn.functional.mse_loss(
                    ws[:, -1:, :],
                    pred_gen_output['sr_w_code'])
                loss += loss_ws * 0.1

                loss_dict.update(
                    dict(loss_feature_volume=loss_feature_volume,
                         loss=loss,
                         loss_shape=loss_shape, 
                         loss_ws=loss_ws))

                log_rec3d_loss_dict(loss_dict)

            self.mp_trainer_rec.backward(loss, disable_amp=True)

            for name, p in self.rec_model.named_parameters():
                if p.grad is None:
                    print(f"found rec unused param: {name}")

            # for name, p in self.ddp_model.named_parameters():
            #     if p.grad is None:
            #         print(f"(in eg3d)found rec unused param: {name}")

            # for name, p in self.ddp_model.named_parameters():
            #     if p.grad is None:
            #         print(f"found rec unused param: {name}")

            if dist_util.get_rank() == 0 and self.step % 500 == 0:
                with th.no_grad():
                    # gt_vis = th.cat([batch['img'], batch['depth']], dim=-1)

                    pred_img = pred_gen_output['image_raw']
                    gt_img = micro['img']

                    if 'depth' in micro:
                        gt_depth = micro['depth']
                        if gt_depth.ndim == 3:
                            gt_depth = gt_depth.unsqueeze(1)
                        gt_depth = (gt_depth - gt_depth.min()) / (
                            gt_depth.max() - gt_depth.min())

                        pred_depth = pred_gen_output['image_depth']
                        pred_depth = (pred_depth - pred_depth.min()) / (
                            pred_depth.max() - pred_depth.min())

                        gt_vis = th.cat(
                            [gt_img,
                             gt_depth.repeat_interleave(3, dim=1)],
                            dim=-1)  # TODO, fail to load depth. range [0, 1]
                    else:

                        gt_vis = th.cat(
                            [gt_img],
                            dim=-1)  # TODO, fail to load depth. range [0, 1]

                    if 'image_sr' in pred_gen_output:
                        pred_img = th.cat([
                            self.pool_512(pred_img),
                            pred_gen_output['image_sr']
                        ],
                                          dim=-1)
                        pred_depth = self.pool_512(pred_depth)
                        gt_depth = self.pool_512(gt_depth)

                        gt_vis = th.cat(
                            [self.pool_512(micro['img']), micro['img_sr'], gt_depth.repeat_interleave(3, dim=1)],
                            dim=-1)

                    pred_vis = th.cat(
                        [pred_img,
                         pred_depth.repeat_interleave(3, dim=1)],
                        dim=-1)  # B, 3, H, W

                    # st()
                    vis = th.cat([gt_vis, pred_vis], dim=-2)[0].permute(
                        1, 2, 0).cpu()  # ! pred in range[-1, 1]
                    # vis_grid = torchvision.utils.make_grid(vis) # HWC
                    vis = vis.numpy() * 127.5 + 127.5
                    vis = vis.clip(0, 255).astype(np.uint8)
                    Image.fromarray(vis).save(
                        f'{logger.get_dir()}/{self.step+self.resume_step}.jpg')
                    print(
                        'log vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}.jpg')

                    # self.writer.add_image(f'images',
                    #                       vis,
                    #                       self.step + self.resume_step,
                    #                       dataformats='HWC')
            return pred_gen_output


class TrainLoop3DRecEG3DRealOnly(TrainLoop3DRecEG3D):

    def __init__(self,
                 *,
                 G,
                 rec_model,
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
                 weight_decay=0,
                 lr_anneal_steps=0,
                 iterations=10001,
                 load_submodule_name='',
                 ignore_resume_opt=False,
                 model_name='rec',
                 use_amp=False,
                 hybrid_training=False,
                 **kwargs):

        super().__init__(G=G,
                         rec_model=rec_model,
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
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         load_submodule_name=load_submodule_name,
                         ignore_resume_opt=ignore_resume_opt,
                         model_name=model_name,
                         use_amp=use_amp,
                         hybrid_training=hybrid_training,
                         **kwargs)

        th.cuda.empty_cache()

    def run_loop(self):
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            # let all processes sync up before starting with a new epoch of training
            dist_util.synchronize()

            # batch, cond = next(self.data)
            # if batch is None:

            # batch = next(self.data)
            # self.run_step(batch, 'g_step_rec_eg3d')

            batch = next(self.data)
            self.run_step(batch, 'g_step_rec')

            # batch = next(self.data)
            # self.run_step(batch, 'd_step_rec')

            # batch = next(self.data)
            # self.run_step(batch, 'g_step_nvs')

            # batch = next(self.data)
            # self.run_step(batch, 'd_step_nvs')

            if self.step % self.log_interval == 0 and dist_util.get_rank(
            ) == 0:
                out = logger.dumpkvs()
                # * log to tensorboard
                for k, v in out.items():
                    self.writer.add_scalar(f'Loss/{k}', v,
                                           self.step + self.resume_step)

            # if self.step % self.eval_interval == 0 and self.step != 0:
            if self.step % self.eval_interval == 0:
                if dist_util.get_rank() == 0:
                    self.eval_loop()
                    # self.eval_novelview_loop()
                    # let all processes sync up before starting with a new epoch of training
                th.cuda.empty_cache()
                dist_util.synchronize()

            if self.step % self.save_interval == 0:
                self.save()

                dist_util.synchronize()
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

                exit()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, step='g_step_rec'):

        if step == 'g_step_rec':
            # self.forward_G_rec(batch)
            self.forward_backward(batch) # raw reconstruction
            took_step_g_rec = self.mp_trainer_rec.optimize(self.opt)

            if took_step_g_rec:
                self._update_ema()  # g_ema

        # if step == 'g_step_nvs':
        #     self.forward_G_nvs(batch)
        #     took_step_g_nvs = self.mp_trainer_rec.optimize(self.opt)

        #     if took_step_g_nvs:
        #         self._update_ema()  # g_ema


        self._anneal_lr()
        self.log_step()

    # basic forward_backward in reconstruction + mean SR code?
    def forward_backward(self, batch, *args, **kwargs):
        # th.cuda.empty_cache()
        self.mp_trainer_rec.zero_grad()
        batch_size = batch['img_to_encoder'].shape[0]

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev())
                for k, v in batch.items()
            }

            last_batch = (i + self.microbatch) >= batch_size

            # wrap forward within amp
            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_rec.use_amp):

                pred = self.rec_model(img=micro['img_to_encoder'],
                                      c=micro['c'])  # pred: (B, 3, 64, 64)
                target = micro

                if last_batch or not self.use_ddp:
                    loss, loss_dict = self.loss_class(pred,
                                                      target,
                                                      test_mode=False)
                else:
                    with self.rec_model.no_sync():  # type: ignore
                        loss, loss_dict = self.loss_class(pred,
                                                          target,
                                                          test_mode=False)

                # ! add density-reg in eg3d, tv-loss

                if self.loss_class.opt.density_reg > 0 and self.step % self.loss_class.opt.density_reg_every == 0:

                    initial_coordinates = th.rand(
                        (batch_size, 1000, 3),
                        device=dist_util.dev()) * 2 - 1  # [-1, 1]
                    perturbed_coordinates = initial_coordinates + th.randn_like(
                        initial_coordinates
                    ) * self.loss_class.opt.density_reg_p_dist
                    all_coordinates = th.cat(
                        [initial_coordinates, perturbed_coordinates], dim=1)

                    sigma = self.rec_model(
                        latent=pred['latent'],
                        coordinates=all_coordinates,
                        directions=th.randn_like(all_coordinates),
                        behaviour='triplane_renderer',
                    )['sigma']

                    sigma_initial = sigma[:, :sigma.shape[1] // 2]
                    sigma_perturbed = sigma[:, sigma.shape[1] // 2:]

                    TVloss = th.nn.functional.l1_loss(
                        sigma_initial,
                        sigma_perturbed) * self.loss_class.opt.density_reg

                    loss_dict.update(dict(tv_loss=TVloss))
                    loss += TVloss

            self.mp_trainer_rec.backward(loss)
            log_rec3d_loss_dict(loss_dict)

            for name, p in self.rec_model.named_parameters():	
                if p.grad is None:
                    logger.log(f"found rec unused param: {name}")

            # for name, p in self.ddp_rec_model.named_parameters():
            #     if p.grad is None:
            #         logger.log(f"found rec unused param: {name}")

            if dist_util.get_rank() == 0 and self.step % 500 == 0:
                with th.no_grad():
                    # gt_vis = th.cat([batch['img'], batch['depth']], dim=-1)

                    pred_img = pred['image_raw']
                    gt_img = micro['img']

                    # if 'depth' in micro:
                    gt_depth = micro['depth']
                    if gt_depth.ndim == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    gt_depth = (gt_depth - gt_depth.min()) / (
                        gt_depth.max() - gt_depth.min())
                    # if True:
                    pred_depth = pred['image_depth']
                    pred_depth = (pred_depth - pred_depth.min()) / (
                        pred_depth.max() - pred_depth.min())

                    # else:

                        # gt_vis = th.cat(
                        #     [gt_img],
                        #     dim=-1)  # TODO, fail to load depth. range [0, 1]

                    if 'image_sr' in pred:
                        if pred['image_sr'].shape[-1] == 512:
                            pred_img = th.cat(
                                [self.pool_512(pred_img), pred['image_sr']],
                                dim=-1)
                            gt_img = th.cat(
                                [self.pool_512(micro['img']), micro['img_sr']],
                                dim=-1)
                            pred_depth = self.pool_512(pred_depth)
                            gt_depth = self.pool_512(gt_depth)

                        elif pred['image_sr'].shape[-1] == 256:
                            pred_img = th.cat(
                                [self.pool_256(pred_img), pred['image_sr']],
                                dim=-1)
                            gt_img = th.cat(
                                [self.pool_256(micro['img']), micro['img_sr']],
                                dim=-1)
                            pred_depth = self.pool_256(pred_depth)
                            gt_depth = self.pool_256(gt_depth)

                        else:
                            pred_img = th.cat(
                                [self.pool_128(pred_img), pred['image_sr']],
                                dim=-1)
                            gt_img = th.cat(
                                [self.pool_128(micro['img']), micro['img_sr']],
                                dim=-1)
                            gt_depth = self.pool_128(gt_depth)
                            pred_depth = self.pool_128(pred_depth)

                    pred_vis = th.cat(
                        [pred_img,
                         pred_depth.repeat_interleave(3, dim=1)],
                        dim=-1)  # B, 3, H, W

                    gt_vis = th.cat(
                        [gt_img,
                            gt_depth.repeat_interleave(3, dim=1)],
                        dim=-1)  # TODO, fail to load depth. range [0, 1]

                    vis = th.cat([gt_vis, pred_vis], dim=-2)[0].permute(
                        1, 2, 0).cpu()  # ! pred in range[-1, 1]
                    # vis_grid = thvision.utils.make_grid(vis) # HWC
                    vis = vis.numpy() * 127.5 + 127.5
                    vis = vis.clip(0, 255).astype(np.uint8)
                    Image.fromarray(vis).save(
                        f'{logger.get_dir()}/{self.step+self.resume_step}.jpg')
                    logger.log(
                        'log vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}.jpg')

                    # self.writer.add_image(f'images',
                    #                       vis,
                    #                       self.step + self.resume_step,
                    #                       dataformats='HWC')
            return pred
