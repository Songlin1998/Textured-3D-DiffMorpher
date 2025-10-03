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

# from .train_util import TrainLoop3DRec
import vision_aided_loss
from .train_util_with_eg3d import TrainLoop3DRecEG3D
from .cvD.nvsD_canoD import TrainLoop3DcvD_nvsD_canoD
# from .train_util_cvD import TrainLoop3DcvD
from dnnlib.util import calculate_adaptive_weight
from .camera_utils import LookAtPoseSampler, FOV_to_intrinsics


# class TrainLoop3DRecEG3DHybrid(TrainLoop3DRecEG3D, TrainLoop3DcvD):
class TrainLoop3DRecEG3DHybrid(TrainLoop3DRecEG3D, TrainLoop3DcvD_nvsD_canoD):

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

        self._prepare_nvs_pose()
        th.cuda.empty_cache()
    
    def _prepare_nvs_pose(self):

        device = dist_util.dev()
        
        fov_deg = 18.837 # for ffhq/afhq
        intrinsics = FOV_to_intrinsics(fov_deg, device=device)

        all_nvs_params = []

        pitch_range = 0.25
        yaw_range = 0.35
        num_keyframes = 100 # how many nv poses to sample from
        w_frames = 1

        cam_pivot = th.Tensor(self.G.rendering_kwargs.get('avg_camera_pivot')).to(device)
        cam_radius = self.G.rendering_kwargs.get('avg_camera_radius')

        for frame_idx in range(num_keyframes):

            cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                    3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                    cam_pivot, radius=cam_radius, device=device)

            camera_params = th.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            all_nvs_params.append(camera_params)
        
        self.all_nvs_params = th.cat(all_nvs_params, 0)

    def run_loop(self):
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            # let all processes sync up before starting with a new epoch of training
            dist_util.synchronize()

            # batch, cond = next(self.data)
            # if batch is None:

            batch = next(self.data)
            self.run_step(batch, 'g_step_rec_eg3d')

            batch = next(self.data)
            self.run_step(batch, 'g_step_rec')

            batch = next(self.data)
            self.run_step(batch, 'd_step_rec')

            batch = next(self.data)
            self.run_step(batch, 'g_step_nvs')

            batch = next(self.data)
            self.run_step(batch, 'd_step_nvs')

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

        if step == 'g_step_rec_eg3d':
            self.forward_G_rec_eg3d(batch)
            took_step_g_rec = self.mp_trainer_rec.optimize(self.opt)

            if took_step_g_rec:
                self._update_ema()  # g_ema

        elif step == 'g_step_rec':
            self.forward_G_rec(batch)
            took_step_g_rec = self.mp_trainer_rec.optimize(self.opt)

            if took_step_g_rec:
                self._update_ema()  # g_ema

        elif step == 'd_step_rec':
            self.forward_D(batch, behaviour='rec')
            _ = self.mp_trainer_canonical_cvD.optimize(self.opt_cano_cvD)

        elif step == 'g_step_nvs':
            self.forward_G_nvs(batch)
            took_step_g_nvs = self.mp_trainer_rec.optimize(self.opt)

            if took_step_g_nvs:
                self._update_ema()  # g_ema

        elif step == 'd_step_nvs':
            self.forward_D(batch, behaviour='nvs')
            _ = self.mp_trainer_cvD.optimize(self.opt_cvD)

        self._anneal_lr()
        self.log_step()

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
                    depth_mask=eg3d_batch['image_depth']>0,
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
                if pred_gen_output.get('sr_w_code', None) is not None:
                    loss_ws = th.nn.functional.mse_loss(
                        ws[:, -1:, :],
                        pred_gen_output['sr_w_code'])
                    loss += loss_ws * 0.1
                else:
                    loss_ws = th.tensor(0.0, device=dist_util.dev())

                loss_dict.update(
                    dict(loss_feature_volume=loss_feature_volume,
                         loss=loss,
                         loss_shape=loss_shape, 
                         loss_ws=loss_ws))

                log_rec3d_loss_dict(loss_dict)

            self.mp_trainer_rec.backward(loss, disable_amp=True)
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

    def forward_G_nvs(self, batch):  # update G

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

            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_rec.use_amp):
                
                random_indices = th.randperm(self.all_nvs_params.shape[0])[:micro['img_to_encoder'].shape[0]].to(dist_util.dev())
                random_c = th.index_select(self.all_nvs_params, 0, random_indices) # randomly selected nvs c

                pred_nvs = self.rec_model(
                    img=micro['img_to_encoder'],
                    c=random_c,
                    # c=th.cat([
                    #     micro['c'][1:],
                    #     micro['c'][:1],
                    # ])
                    )  # ! render novel views only for D loss

                # add cvD supervision

                if 'image_sr' in pred_nvs:
                    vision_aided_loss = self.ddp_nvs_cvD(
                        # pred_nv['image_sr'], 
                        0.5 * pred_nvs['image_sr'] + 0.5 * th.nn.functional.interpolate(pred_nvs['image_raw'], size=pred_nvs['image_sr'].shape[2:], mode='bilinear'),
                        for_G=True).mean()  # ! for debugging
                        # pred_nv['image_raw'], for_G=True).mean()  # [B, 1] shape
                else:
                    vision_aided_loss = self.ddp_nvs_cvD(
                        pred_nvs['image_raw'], for_G=True).mean()  # [B, 1] shape

                loss = vision_aided_loss * self.loss_class.opt.nvs_cvD_lambda

                log_rec3d_loss_dict({
                    'vision_aided_loss/G_nvs':
                    vision_aided_loss,
                })

            self.mp_trainer_rec.backward(loss)

            # ! move to other places, add tensorboard

            if dist_util.get_rank() == 0 and self.step % 500 == 0:
                with th.no_grad():
                    # gt_vis = th.cat([batch['img'], batch['depth']], dim=-1)

                    # st()

                    gt_depth = micro['depth']
                    if gt_depth.ndim == 3:
                        gt_depth = gt_depth.unsqueeze(1)
                    gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() -
                                                              gt_depth.min())
                    # if True:
                    pred_depth = pred_nvs['image_depth']
                    pred_depth = (pred_depth - pred_depth.min()) / (
                        pred_depth.max() - pred_depth.min())
                    pred_img = pred_nvs['image_raw']
                    gt_img = micro['img']

                    if 'image_sr' in pred_nvs:

                        if pred_nvs['image_sr'].shape[-1] == 512:
                            pred_img = th.cat(
                                [self.pool_512(pred_img), pred_nvs['image_sr']],
                                dim=-1)
                            gt_img = th.cat(
                                [self.pool_512(micro['img']), micro['img_sr']],
                                dim=-1)
                            pred_depth = self.pool_512(pred_depth)
                            gt_depth = self.pool_512(gt_depth)

                        elif pred_nvs['image_sr'].shape[-1] == 256:
                            pred_img = th.cat(
                                [self.pool_256(pred_img), pred_nvs['image_sr']],
                                dim=-1)
                            gt_img = th.cat(
                                [self.pool_256(micro['img']), micro['img_sr']],
                                dim=-1)
                            pred_depth = self.pool_256(pred_depth)
                            gt_depth = self.pool_256(gt_depth)

                        else:
                            pred_img = th.cat(
                                [pred_img, pred_nvs['image_sr']],
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

                # novel_view_c = th.cat([micro['c'][1:], micro['c'][:1]])

                random_indices = th.randperm(self.all_nvs_params.shape[0])[:micro['img_to_encoder'].shape[0]].to(dist_util.dev())
                novel_view_c = th.index_select(self.all_nvs_params, 0, random_indices) # randomly selected nvs c

                latent = self.rec_model(img=micro['img_to_encoder'],
                                        behaviour='enc_dec_wo_triplane')

                cano_pred = self.rec_model(latent=latent,
                                           c=micro['c'],
                                           behaviour='triplane_dec')

                # TODO, optimize with one encoder, and two triplane decoder
                if behaviour == 'rec':

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
                else:
                    assert behaviour == 'nvs'

                    nvs_pred = self.rec_model(latent=latent,
                                              c=novel_view_c,
                                              behaviour='triplane_dec')

                    if 'image_sr' in nvs_pred:
                        d_loss_nvs = self.run_D_Diter(
                            # real=cano_pred['image_sr'],
                            # fake=nvs_pred['image_sr'],
                            real=0.5 * cano_pred['image_sr'] + 0.5 * th.nn.functional.interpolate(cano_pred['image_raw'], size=cano_pred['image_sr'].shape[2:], mode='bilinear'),
                            fake=0.5 * nvs_pred['image_sr'] + 0.5 * th.nn.functional.interpolate(nvs_pred['image_raw'], size=nvs_pred['image_sr'].shape[2:], mode='bilinear'),
                            D=self.ddp_nvs_cvD)  # TODO, add SR for FFHQ
                    else:
                        d_loss_nvs = self.run_D_Diter(
                            real=cano_pred['image_raw'],
                            fake=nvs_pred['image_raw'],
                            D=self.ddp_nvs_cvD)  # TODO, add SR for FFHQ

                    log_rec3d_loss_dict(
                        {'vision_aided_loss/D_nvs': d_loss_nvs})
                    self.mp_trainer_cvD.backward(d_loss_nvs)