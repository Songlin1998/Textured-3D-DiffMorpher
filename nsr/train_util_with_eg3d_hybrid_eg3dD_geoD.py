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
from .train_util_cvD import TrainLoop3DcvD
from dnnlib.util import calculate_adaptive_weight
from .camera_utils import LookAtPoseSampler, FOV_to_intrinsics

from torch_utils.ops import conv2d_gradfix, upfirdn2d
from nsr.dual_discriminator import filtered_resizing


# class TrainLoop3DRecEG3DHybrid(TrainLoop3DRecEG3D, TrainLoop3DcvD):
class TrainLoop3DRecEG3DHybridEG3DD(TrainLoop3DRecEG3D):

    def __init__(self,
                 *,
                 G,
                 D,
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
                 augment_pipe=None,
                #  num_of_cano_img_for_sup=0,
                 num_of_cano_img_for_sup=2,
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
        self.num_of_cano_img_for_sup = num_of_cano_img_for_sup  # TODO, add schedule

        self.eg3d_D = D# .requires_grad_(True)
        for name, param in self.eg3d_D.named_parameters():
            if any([res in name for res in ('b32', 'b16', 'b8', 'b4')]):
                param.requires_grad_(True)
                logger.log(name)

        self.augment_pipe = augment_pipe

        self.mp_trainer_eg3d_D = MixedPrecisionTrainer(
            model=self.eg3d_D,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            model_name='eg3d_D',
            use_amp=use_amp)

        self.opt_eg3d_D = AdamW(
            self.mp_trainer_eg3d_D.master_params,
            lr=1e-5,  # same as the G
            betas=(0, 0.99),
            eps=1e-8)  # dlr in biggan cfg

        if self.use_ddp:
            self.ddp_eg3d_D = DDP(
                self.eg3d_D,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            self.ddp_eg3d_D = self.eg3d_D
        
        # ! add geometry D

        device = dist_util.dev()

        cvD_name = 'geoD'

        self.cvD = vision_aided_loss.Discriminator(
            cv_type='clip', loss_type='multilevel_sigmoid_s',
            device=device).to(device)
        self.cvD.cv_ensemble.requires_grad_(False)  # Freeze feature extractor

        self._load_and_sync_parameters(model=self.cvD, model_name=cvD_name)

        self.mp_trainer_cvD = MixedPrecisionTrainer(
            model=self.cvD,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            model_name=cvD_name,
            use_amp=use_amp)

        self.opt_cvD = AdamW(
            self.mp_trainer_cvD.master_params,
            lr=1e-5, # same as the G
            betas=(0, 0.99),
            eps=1e-8)  # dlr in biggan cfg

        if self.use_ddp:
            self.ddp_cvD = DDP(
                self.cvD,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            self.ddp_cvD = self.cvD


        th.cuda.empty_cache()

    def _prepare_nvs_pose(self):

        device = dist_util.dev()

        fov_deg = 18.837  # for ffhq/afhq
        intrinsics = FOV_to_intrinsics(fov_deg, device=device)

        all_nvs_params = []

        pitch_range = 0.25
        yaw_range = 0.35
        num_keyframes = 100  # how many nv poses to sample from
        w_frames = 1

        cam_pivot = th.Tensor(
            self.G.rendering_kwargs.get('avg_camera_pivot')).to(device)
        cam_radius = self.G.rendering_kwargs.get('avg_camera_radius')

        for frame_idx in range(num_keyframes):

            cam2world_pose = LookAtPoseSampler.sample(
                3.14 / 2 + yaw_range * np.sin(2 * 3.14 * frame_idx /
                                              (num_keyframes * w_frames)),
                3.14 / 2 - 0.05 +
                pitch_range * np.cos(2 * 3.14 * frame_idx /
                                     (num_keyframes * w_frames)),
                cam_pivot,
                radius=cam_radius,
                device=device)

            camera_params = th.cat(
                [cam2world_pose.reshape(-1, 16),
                 intrinsics.reshape(-1, 9)], 1)

            all_nvs_params.append(camera_params)

        self.all_nvs_params = th.cat(all_nvs_params, 0)

    def _sample_nvs_pose(self, batch_size):

        device = dist_util.dev()

        fov_deg = 18.837  # for ffhq/afhq
        intrinsics = FOV_to_intrinsics(fov_deg, device=device)

        all_nvs_params = []

        pitch_range = 0.25
        yaw_range = 0.35
        # num_keyframes = batch_size  # how many nv poses to sample from
        w_frames = 1

        cam_pivot = th.Tensor(
            self.G.rendering_kwargs.get('avg_camera_pivot')).to(device)
        cam_radius = self.G.rendering_kwargs.get('avg_camera_radius')

        for _ in range(batch_size):

            cam2world_pose = LookAtPoseSampler.sample(
                np.pi / 2,
                np.pi / 2,
                cam_pivot,
                horizontal_stddev=yaw_range,
                vertical_stddev=pitch_range,
                radius=cam_radius,
                device=device)

            camera_params = th.cat(
                [cam2world_pose.reshape(-1, 16),
                 intrinsics.reshape(-1, 9)], 1)

            all_nvs_params.append(camera_params)

        return th.cat(all_nvs_params, dim=0)

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
            self.run_step(batch, 'g_step_nvs')

            batch = next(self.data)
            self.run_step(batch, 'd_step_hybrid')

            # batch = next(self.data)
            # self.run_step(batch, 'forward_G_hybrid')

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
                # self.save(self.mp_trainer_cvD, 'cvD')
                # self.save(self.mp_trainer_canonical_cvD, 'cano_cvD')

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
                    # self.save(self.mp_trainer_cvD, 'cvD')
                    # self.save(self.mp_trainer_canonical_cvD, 'cano_cvD')

                exit()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            # self.save(self.mp_trainer_canonical_cvD, 'cvD')

    def run_step(self, batch, step='g_step'):

        if step == 'g_step_rec_eg3d':
            self.forward_G_rec_eg3d(batch)
            took_step_g_rec = self.mp_trainer.optimize(self.opt)

            if took_step_g_rec:
                self._update_ema()  # g_ema

        if step == 'g_step_rec':
            self.forward_G_rec(batch)
            took_step_g_rec = self.mp_trainer.optimize(self.opt)

            if took_step_g_rec:
                self._update_ema()  # g_ema

        elif step == 'forward_G_hybrid':
            self.forward_G_hybrid(batch)
            took_step_g_rec = self.mp_trainer.optimize(self.opt)

            if took_step_g_rec:
                self._update_ema()  # g_ema

        elif step == 'd_step_hybrid':
            # self.forward_D(batch, behaviour='rec')
            self.forward_D(batch)
            _ = self.mp_trainer_eg3d_D.optimize(self.opt_eg3d_D)

        elif step == 'g_step_nvs':
            self.forward_G_nvs(batch)
            took_step_g_nvs = self.mp_trainer.optimize(self.opt)

            if took_step_g_nvs:
                self._update_ema()  # g_ema

        self._anneal_lr()
        self.log_step()

    def forward_G_rec_eg3d(self, batch, *args, **kwargs):

        self.mp_trainer.zero_grad()
        self.ddp_model.requires_grad_(True)
        self.ddp_eg3d_D.requires_grad_(False)
        # ! geoD also in here
        self.ddp_cvD.requires_grad_(True)

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
                    depth_mask=eg3d_batch['image_depth'] > 0,
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
                    with self.ddp_model.no_sync():  # type: ignore
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
                        ws[:, -1:, :], pred_gen_output['sr_w_code'])
                    loss += loss_ws * 0.1
                else:
                    loss_ws = th.tensor(0.0, device=dist_util.dev())

                loss_dict.update(
                    dict(loss_feature_volume=loss_feature_volume,
                         loss=loss,
                         loss_shape=loss_shape,
                         loss_ws=loss_ws))

                log_rec3d_loss_dict(loss_dict)
                log_rec3d_loss_dict(loss_dict)
            
                # ! geoD loss
                loss_cvD = self.run_cvD_Diter(
                    real=eg3d_batch['shape_synthesized']['image_depth'].unsqueeze(1).repeat_interleave(3, dim=1),
                    fake=pred_gen_output['image_depth'].unsqueeze(1).repeat_interleave(3, dim=1))  # TODO, add SR for FFHQ

                log_rec3d_loss_dict({'vision_aided_loss/geoD': loss_cvD})


            self.mp_trainer.backward(loss, disable_amp=True)
            self.mp_trainer_cvD.backward(loss_cvD, disable_amp=True)
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

                        gt_vis = th.cat([
                            self.pool_512(micro['img']), micro['img_sr'],
                            gt_depth.repeat_interleave(3, dim=1)
                        ],
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

        self.mp_trainer.zero_grad()
        self.ddp_model.requires_grad_(True)
        self.ddp_eg3d_D.requires_grad_(False)  # only use novel view D
        self.ddp_cvD.requires_grad_(False)

        # self.ddp_canonical_cvD.requires_grad_(False)

        batch_size = batch['img'].shape[0]

        for i in range(0, batch_size, self.microbatch):
            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev()).contiguous()
                for k, v in batch.items()
            }

            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer.use_amp):

                # random_indices = th.randperm(
                #     self.all_nvs_params.shape[0])[:micro['img_to_encoder'].
                #                                   shape[0]].to(dist_util.dev())
                # random_c = th.index_select(
                #     self.all_nvs_params, 0,
                #     random_indices)  # randomly selected nvs c
                random_c = self._sample_nvs_pose(batch_size)

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
                    gen_logits = self.ddp_eg3d_D(
                        pred_nvs, random_c).mean()  # ! for debugging
                    # pred_nv['image_raw'], micro['c']).mean()  # [B, 1] shape
                else:
                    gen_logits = self.ddp_eg3d_D(
                        pred_nvs, random_c).mean()  # [B, 1] shape

                eg3d_d_loss = th.nn.functional.softplus(-gen_logits)
                loss = eg3d_d_loss * self.loss_class.opt.nvs_cvD_lambda

                # ! vision aided loss
                vision_aided_loss = self.ddp_cvD(
                    pred_nvs['image_depth'].unsqueeze(1).repeat_interleave(3, dim=1),
                    for_G=True).mean()  # [B, 1] shape
                loss += vision_aided_loss * self.loss_class.opt.nvs_cvD_lambda

                log_rec3d_loss_dict({
                    'eg3d_d_loss/G_nvs': eg3d_d_loss,
                    'vision_aided_loss/G_nvs_geo': vision_aided_loss * self.loss_class.opt.nvs_cvD_lambda,
                })

            self.mp_trainer.backward(loss)

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
                    pred_depth = pred_nvs['image_depth']
                    pred_depth = (pred_depth - pred_depth.min()) / (
                        pred_depth.max() - pred_depth.min())
                    pred_img = pred_nvs['image_raw']
                    gt_img = micro['img']

                    if 'image_sr' in pred_nvs:

                        if pred_nvs['image_sr'].shape[-1] == 512:
                            pred_img = th.cat([
                                self.pool_512(pred_img), pred_nvs['image_sr']
                            ],
                                              dim=-1)
                            gt_img = th.cat(
                                [self.pool_512(micro['img']), micro['img_sr']],
                                dim=-1)
                            pred_depth = self.pool_512(pred_depth)
                            gt_depth = self.pool_512(gt_depth)

                        elif pred_nvs['image_sr'].shape[-1] == 256:
                            pred_img = th.cat([
                                self.pool_256(pred_img), pred_nvs['image_sr']
                            ],
                                              dim=-1)
                            gt_img = th.cat(
                                [self.pool_256(micro['img']), micro['img_sr']],
                                dim=-1)
                            pred_depth = self.pool_256(pred_depth)
                            gt_depth = self.pool_256(gt_depth)

                        else:
                            pred_img = th.cat([pred_img, pred_nvs['image_sr']],
                                              dim=-1)
                            gt_img = th.cat([micro['img'], micro['img_sr']],
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

    def forward_D(self,
                  batch,
                  behaviour='none'):  # update D, only applied to novel view ?
        self.mp_trainer_eg3d_D.zero_grad()
        self.rec_model.requires_grad_(False)
        self.eg3d_D.requires_grad_(True)
        # self.ddp_model.requires_grad_(False)

        batch_size = batch['img'].shape[0]
        blur_sigma = 0

        # * sample a new batch for D training
        for i in range(0, batch_size, self.microbatch):
            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev()).contiguous()
                for k, v in batch.items()
            }

            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_eg3d_D.use_amp):

                # novel_view_c = th.cat([micro['c'][1:], micro['c'][:1]])

                random_indices = th.randperm(
                    self.all_nvs_params.shape[0])[:micro['img_to_encoder'].
                                                  shape[0]].to(dist_util.dev())
                novel_view_c = th.index_select(
                    self.all_nvs_params, 0,
                    random_indices)  # randomly selected nvs c

                rec_input_img = micro['img_to_encoder']
                if self.augment_pipe is not None:
                    rec_input_img = self.augment_pipe(rec_input_img)

                latent = self.rec_model(img=rec_input_img,
                                        behaviour='enc_dec_wo_triplane')

                cano_pred = self.rec_model(
                    latent={k:v[:self.num_of_cano_img_for_sup] for k, v in latent.items()},
                    c=micro['c'][:self.num_of_cano_img_for_sup],
                    behaviour='triplane_dec')

                nvs_pred = self.rec_model(
                    # latent=latent[self.num_of_cano_img_for_sup:],
                    latent={k:v[self.num_of_cano_img_for_sup:] for k, v in latent.items()},
                    c=novel_view_c[self.num_of_cano_img_for_sup:],
                    behaviour='triplane_dec')

                all_pred = {}
                for k, cano_v in cano_pred.items(
                ):  # concat the novel view and canonical view predictiosn
                    # all_pred[k] = th.cat([cano_pred[k], nvs_pred[k]], dim=0)
                    if k == 'shape_synthesized':
                        continue
                    all_pred[k] = th.cat([cano_v, nvs_pred[k]], dim=0)

                all_c = th.cat([
                    micro['c'][:self.num_of_cano_img_for_sup],
                    novel_view_c[self.num_of_cano_img_for_sup:]
                ],
                               dim=0)

                # TODO, optimize with one encoder, and two triplane decoder

                # Dmain: Minimize logits for generated images.
                gen_logits = self.run_D(all_pred,
                                        all_c,
                                        blur_sigma=blur_sigma,
                                        update_emas=True)
                loss_Dgen = th.nn.functional.softplus(gen_logits)

                # Dmain: Maximize logits for real images.
                # Dr1: Apply R1 regularization.0

                real_img_tmp_image = micro['img_sr'].detach().requires_grad_(
                    True)
                real_img_tmp_image_raw = micro['img'].detach(
                ).requires_grad_(True)
                real_img_tmp = {
                    'image_sr': real_img_tmp_image,
                    'image_raw': real_img_tmp_image_raw
                }

                real_logits = self.run_D(real_img_tmp,
                                         micro['c'],
                                         blur_sigma=blur_sigma)

                loss_Dreal = th.nn.functional.softplus(-real_logits)

                loss_Dr1 = 0

                r1_grads = th.autograd.grad(
                    outputs=[real_logits.sum()],
                    inputs=[real_img_tmp['image_sr'], real_img_tmp['image_raw']],
                    create_graph=True,
                    only_inputs=True)
                r1_grads_image = r1_grads[0]
                r1_grads_image_raw = r1_grads[1]
                r1_penalty = r1_grads_image.square().sum(
                    [1, 2, 3]) + r1_grads_image_raw.square().sum([1, 2, 3])

                loss_Dr1 = r1_penalty * (self.loss_class.opt.r1_gamma / 2)

                d_loss = (loss_Dreal + loss_Dr1 + loss_Dgen).mean()

                log_rec3d_loss_dict({'eg3d_d_loss/D_hybrid': d_loss})
                self.mp_trainer_eg3d_D.backward(d_loss)

    def forward_G_rec(self, batch):  # update G

        self.mp_trainer.zero_grad()
        self.ddp_model.requires_grad_(True)
        self.ddp_cvD.requires_grad_(False)

        # self.ddp_canonical_cvD.requires_grad_(False)
        # self.ddp_cvD.requires_grad_(False)

        batch_size = batch['img'].shape[0]

        for i in range(0, batch_size, self.microbatch):
            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev()).contiguous()
                for k, v in batch.items()
            }

            last_batch = (i + self.microbatch) >= batch_size

            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer.use_amp):

                pred = self.rec_model(
                    img=micro['img_to_encoder'], c=micro['c']
                )  # render novel view for first half of the batch for D loss

                target_for_rec = micro
                pred_for_rec = pred

                if last_batch or not self.use_ddp:
                    loss, loss_dict = self.loss_class(pred_for_rec,
                                                      target_for_rec,
                                                      test_mode=False)
                else:
                    with self.ddp_model.no_sync():  # type: ignore
                        loss, loss_dict = self.loss_class(pred_for_rec,
                                                          target_for_rec,
                                                          test_mode=False)

                # add cvD supervision
                # ! TODO

                assert 'image_sr' in pred_for_rec
                gen_logits = self.eg3d_D(pred_for_rec,
                                         micro['c']).mean()  # [B, 1] shape

                eg3d_d_loss = th.nn.functional.softplus(-gen_logits)
                last_layer = self.rec_model.module.decoder.triplane_decoder.decoder.net[  # type: ignore
                    -1].weight  # type: ignore

                d_weight = calculate_adaptive_weight(
                    loss, eg3d_d_loss, last_layer,
                    disc_weight_max=1) * self.loss_class.opt.rec_cvD_lambda
                loss += eg3d_d_loss * d_weight

                # ! geoD

                vision_aided_loss = self.ddp_cvD(
                    pred_for_rec['image_depth'].unsqueeze(1).repeat_interleave(3, dim=1),
                    for_G=True).mean()  # [B, 1] shape
                loss += vision_aided_loss * d_weight

                loss_dict.update({
                    'eg3d_d_loss/G_rec': eg3d_d_loss * d_weight,
                    'vision_aided_loss/G_rec_geo': vision_aided_loss * d_weight,
                    'd_weight': d_weight
                })

                log_rec3d_loss_dict(loss_dict)

            self.mp_trainer.backward(loss)  # no nvs cvD loss, following VQ3D

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
                    pred_depth = pred['image_depth']
                    pred_depth = (pred_depth - pred_depth.min()) / (
                        pred_depth.max() - pred_depth.min())
                    pred_img = pred['image_raw']
                    gt_img = micro['img']

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
                            pred_img = th.cat([pred_img, pred['image_sr']],
                                              dim=-1)
                            gt_img = th.cat([micro['img'], micro['img_sr']],
                                            dim=-1)

                    gt_vis = th.cat(
                        [gt_img, gt_depth.repeat_interleave(3, dim=1)],
                        dim=-1)  # TODO, fail to load depth. range [0, 1]

                    pred_vis = th.cat(
                        [pred_img,
                         pred_depth.repeat_interleave(3, dim=1)],
                        dim=-1)  # B, 3, H, W

                    vis = th.cat([gt_vis, pred_vis], dim=-2)[0].permute(
                        1, 2, 0).cpu()  # ! pred in range[-1, 1]
                    # vis_grid = torchvision.utils.make_grid(vis) # HWC
                    vis = vis.numpy() * 127.5 + 127.5
                    vis = vis.clip(0, 255).astype(np.uint8)
                    Image.fromarray(vis).save(
                        f'{logger.get_dir()}/{self.step+self.resume_step}_rec.jpg'
                    )
                    print(
                        'log vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}_rec.jpg'
                    )

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with th.autograd.profiler.record_function('blur'):
                f = th.arange(-blur_size,
                              blur_size + 1,
                              device=img['image'].device).div(
                                  blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        logits = self.ddp_eg3d_D(img, c, update_emas=update_emas)
        return logits

    def forward_G_hybrid(self, batch):
        """combine the input reconstruction and nv D loss, dynamically increase the reconstruction part weight
        """

        self.mp_trainer.zero_grad()
        self.ddp_model.requires_grad_(True)

        # self.ddp_canonical_cvD.requires_grad_(False)
        # self.ddp_cvD.requires_grad_(False)

        batch_size = batch['img'].shape[0]

        for i in range(0, batch_size, self.microbatch):
            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev()).contiguous()
                for k, v in batch.items()
            }

            last_batch = (i + self.microbatch) >= batch_size

            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer.use_amp):

                rec_input_img = micro['img_to_encoder']
                if self.augment_pipe is not None:
                    rec_input_img = self.augment_pipe(rec_input_img)

                # latent = self.rec_model(img=micro['img_to_encoder'],
                latent = self.rec_model(img=rec_input_img,
                                        behaviour='enc_dec_wo_triplane')

                novel_view_c = self._sample_nvs_pose(
                    batch_size - self.num_of_cano_img_for_sup)

                nvs_pred = self.rec_model(
                    latent={
                        k: v[self.num_of_cano_img_for_sup:]
                        for k, v in latent.items()
                    },
                    c=novel_view_c[self.num_of_cano_img_for_sup:],
                    behaviour='triplane_dec')

                if self.num_of_cano_img_for_sup > 0:

                    cano_pred = self.rec_model(
                        latent={
                            k: v[:self.num_of_cano_img_for_sup]
                            for k, v in latent.items()
                        },
                        c=micro['c'][:self.num_of_cano_img_for_sup],
                        behaviour='triplane_dec')

                    target_for_rec = {
                        k:
                        v[:self.num_of_cano_img_for_sup]
                        if v is not None else None
                        for k, v in micro.items()
                    }

                    if last_batch or not self.use_ddp:
                        loss, loss_dict = self.loss_class(cano_pred,
                                                          target_for_rec,
                                                          test_mode=False)
                    else:
                        with self.ddp_model.no_sync():  # type: ignore
                            loss, loss_dict = self.loss_class(cano_pred,
                                                              target_for_rec,
                                                              test_mode=False)
                    all_pred = {}
                    for k, _ in cano_pred.keys(
                    ):  # concat the novel view and canonical view predictiosn
                        all_pred[k] = th.cat([cano_pred[k], nvs_pred[k]], dim=0)

                else:
                    loss = th.tensor(0., device=dist_util.dev())
                    loss_dict = {}
                    all_pred = nvs_pred

                all_c = th.cat([
                    micro['c'][:self.num_of_cano_img_for_sup],
                    novel_view_c[self.num_of_cano_img_for_sup:]
                ],
                               dim=0)

                # add cvD supervision
                # ! TODO

                assert 'image_sr' in all_pred
                gen_logits = self.eg3d_D(all_pred,
                                         all_c).mean()  # [B, 1] shape

                eg3d_d_loss = th.nn.functional.softplus(-gen_logits)
                last_layer = self.rec_model.module.decoder.triplane_decoder.decoder.net[  # type: ignore
                    -1].weight  # type: ignore

                d_weight = calculate_adaptive_weight(loss,
                                                     eg3d_d_loss,
                                                     last_layer,
                                                     disc_weight_max=1)
                # disc_weight_max=1) * self.loss_class.opt.rec_cvD_lambda
                loss += eg3d_d_loss * d_weight

                loss_dict.update({
                    'eg3d_d_loss/G_rec': eg3d_d_loss * d_weight,
                    'd_weight': d_weight
                })

                log_rec3d_loss_dict(loss_dict)

            self.mp_trainer.backward(loss)  # no nvs cvD loss, following VQ3D

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
                    pred_depth = pred['image_depth']
                    pred_depth = (pred_depth - pred_depth.min()) / (
                        pred_depth.max() - pred_depth.min())
                    pred_img = pred['image_raw']
                    gt_img = micro['img']

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
                            pred_img = th.cat([pred_img, pred['image_sr']],
                                              dim=-1)
                            gt_img = th.cat([micro['img'], micro['img_sr']],
                                            dim=-1)

                    gt_vis = th.cat(
                        [gt_img, gt_depth.repeat_interleave(3, dim=1)],
                        dim=-1)  # TODO, fail to load depth. range [0, 1]

                    pred_vis = th.cat(
                        [pred_img,
                         pred_depth.repeat_interleave(3, dim=1)],
                        dim=-1)  # B, 3, H, W

                    vis = th.cat([gt_vis, pred_vis], dim=-2)[0].permute(
                        1, 2, 0).cpu()  # ! pred in range[-1, 1]
                    # vis_grid = torchvision.utils.make_grid(vis) # HWC
                    vis = vis.numpy() * 127.5 + 127.5
                    vis = vis.clip(0, 255).astype(np.uint8)
                    Image.fromarray(vis).save(
                        f'{logger.get_dir()}/{self.step+self.resume_step}_rec.jpg'
                    )
                    print(
                        'log vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}_rec.jpg'
                    )

    def run_cvD_Diter(self, real, fake, D=None):
        # Dmain: Minimize logits for generated images and maximize logits for real images.
        if D is None:
            D = self.ddp_cvD

        lossD = D(real, for_real=True).mean() + D(
            fake, for_real=False).mean()
        return lossD

