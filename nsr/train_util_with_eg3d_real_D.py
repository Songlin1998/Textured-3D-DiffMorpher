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

from .train_util_with_eg3d_real import TrainLoop3DRecEG3DRealOnly
from .train_util_with_eg3d_hybrid_eg3dD import TrainLoop3DRecEG3DHybridEG3DD


class TrainLoop3DRecEG3DRealOnl_D(TrainLoop3DRecEG3DHybridEG3DD):
    """
    add input view and novel view D
    """

    def __init__(self,
                 *,
                 G,
                 D,
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
                         D=D,
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

        logger.log(f'self.adv_loss_start_iter: {self.adv_loss_start_iter}')

        th.cuda.empty_cache()

    def run_loop(self):
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            # let all processes sync up before starting with a new epoch of training
            dist_util.synchronize()

            batch = next(self.data)
            self.run_step(batch, 'g_step_rec')

            if self.step + self.resume_step >= self.adv_loss_start_iter:

                batch = next(self.data)
                self.run_step(batch, 'g_step_nvs')
                
                batch = next(self.data)
                # self.run_step(batch, 'd_step_rec')
                self.run_step(batch, 'd_step')

            # batch = next(self.data)
            # self.run_step(batch, 'd_step_nvs')

            # * log to tensorboard
            if self.step % self.log_interval == 0 and dist_util.get_rank(
            ) == 0:
                out = logger.dumpkvs()
                for k, v in out.items():
                    self.writer.add_scalar(f'Loss/{k}', v,
                                           self.step + self.resume_step)

            # if self.step % self.eval_interval == 0 and self.step != 0:
            if self.step % self.eval_interval == 0:
                if dist_util.get_rank() == 0:
                    self.eval_loop()
                    self.eval_novelview_loop()
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
            self.forward_G_rec(batch)
            took_step_g_rec = self.mp_trainer_rec.optimize(self.opt)

            if took_step_g_rec:
                self._update_ema()  # g_ema

        # elif step == 'd_step_rec':
        #     self.forward_D(batch, behaviour='rec')
        #     _ = self.mp_trainer_eg3d_D.optimize(self.opt_eg3d_D)

        elif step == 'g_step_nvs':
            self.forward_G_nvs(batch)
            took_step_g_nvs = self.mp_trainer_rec.optimize(self.opt)

            if took_step_g_nvs:
                self._update_ema()  # g_ema

        elif step == 'd_step':
            # self.forward_D(batch, behaviour='nvs')
            self.forward_D(batch)
            _ = self.mp_trainer_eg3d_D.optimize(self.opt_eg3d_D)


        self._anneal_lr()
        self.log_step()

    # # basic forward_backward in reconstruction + mean SR code?
    # def forward_backward(self, batch, *args, **kwargs):
    #     # th.cuda.empty_cache()
    #     self.mp_trainer_rec.zero_grad()
    #     batch_size = batch['img_to_encoder'].shape[0]

    #     for i in range(0, batch_size, self.microbatch):

    #         micro = {
    #             k: v[i:i + self.microbatch].to(dist_util.dev())
    #             for k, v in batch.items()
    #         }

    #         last_batch = (i + self.microbatch) >= batch_size

    #         # wrap forward within amp
    #         with th.autocast(device_type='cuda',
    #                          dtype=th.float16,
    #                          enabled=self.mp_trainer_rec.use_amp):

    #             pred = self.rec_model(img=micro['img_to_encoder'],
    #                                   c=micro['c'])  # pred: (B, 3, 64, 64)
    #             target = micro

    #             if last_batch or not self.use_ddp:
    #                 loss, loss_dict = self.loss_class(pred,
    #                                                   target,
    #                                                   test_mode=False)
    #             else:
    #                 with self.rec_model.no_sync():  # type: ignore
    #                     loss, loss_dict = self.loss_class(pred,
    #                                                       target,
    #                                                       test_mode=False)

    #             # ! add density-reg in eg3d, tv-loss

    #             if self.loss_class.opt.density_reg > 0 and self.step % self.loss_class.opt.density_reg_every == 0:

    #                 initial_coordinates = th.rand(
    #                     (batch_size, 1000, 3),
    #                     device=dist_util.dev()) * 2 - 1  # [-1, 1]
    #                 perturbed_coordinates = initial_coordinates + th.randn_like(
    #                     initial_coordinates
    #                 ) * self.loss_class.opt.density_reg_p_dist
    #                 all_coordinates = th.cat(
    #                     [initial_coordinates, perturbed_coordinates], dim=1)

    #                 sigma = self.rec_model(
    #                     latent=pred['latent'],
    #                     coordinates=all_coordinates,
    #                     directions=th.randn_like(all_coordinates),
    #                     behaviour='triplane_renderer',
    #                 )['sigma']

    #                 sigma_initial = sigma[:, :sigma.shape[1] // 2]
    #                 sigma_perturbed = sigma[:, sigma.shape[1] // 2:]

    #                 TVloss = th.nn.functional.l1_loss(
    #                     sigma_initial,
    #                     sigma_perturbed) * self.loss_class.opt.density_reg

    #                 loss_dict.update(dict(tv_loss=TVloss))
    #                 loss += TVloss

    #         self.mp_trainer_rec.backward(loss)
    #         log_rec3d_loss_dict(loss_dict)

    #         for name, p in self.rec_model.named_parameters():	
    #             if p.grad is None:
    #                 logger.log(f"found rec unused param: {name}")

    #         # for name, p in self.ddp_rec_model.named_parameters():
    #         #     if p.grad is None:
    #         #         logger.log(f"found rec unused param: {name}")

    #         if dist_util.get_rank() == 0 and self.step % 500 == 0:
    #             with th.no_grad():
    #                 # gt_vis = th.cat([batch['img'], batch['depth']], dim=-1)

    #                 pred_img = pred['image_raw']
    #                 gt_img = micro['img']

    #                 # if 'depth' in micro:
    #                 gt_depth = micro['depth']
    #                 if gt_depth.ndim == 3:
    #                     gt_depth = gt_depth.unsqueeze(1)
    #                 gt_depth = (gt_depth - gt_depth.min()) / (
    #                     gt_depth.max() - gt_depth.min())
    #                 # if True:
    #                 pred_depth = pred['image_depth']
    #                 pred_depth = (pred_depth - pred_depth.min()) / (
    #                     pred_depth.max() - pred_depth.min())

    #                 # else:

    #                     # gt_vis = th.cat(
    #                     #     [gt_img],
    #                     #     dim=-1)  # TODO, fail to load depth. range [0, 1]

    #                 if 'image_sr' in pred:
    #                     if pred['image_sr'].shape[-1] == 512:
    #                         pred_img = th.cat(
    #                             [self.pool_512(pred_img), pred['image_sr']],
    #                             dim=-1)
    #                         gt_img = th.cat(
    #                             [self.pool_512(micro['img']), micro['img_sr']],
    #                             dim=-1)
    #                         pred_depth = self.pool_512(pred_depth)
    #                         gt_depth = self.pool_512(gt_depth)

    #                     elif pred['image_sr'].shape[-1] == 256:
    #                         pred_img = th.cat(
    #                             [self.pool_256(pred_img), pred['image_sr']],
    #                             dim=-1)
    #                         gt_img = th.cat(
    #                             [self.pool_256(micro['img']), micro['img_sr']],
    #                             dim=-1)
    #                         pred_depth = self.pool_256(pred_depth)
    #                         gt_depth = self.pool_256(gt_depth)

    #                     else:
    #                         pred_img = th.cat(
    #                             [self.pool_128(pred_img), pred['image_sr']],
    #                             dim=-1)
    #                         gt_img = th.cat(
    #                             [self.pool_128(micro['img']), micro['img_sr']],
    #                             dim=-1)
    #                         gt_depth = self.pool_128(gt_depth)
    #                         pred_depth = self.pool_128(pred_depth)

    #                 pred_vis = th.cat(
    #                     [pred_img,
    #                      pred_depth.repeat_interleave(3, dim=1)],
    #                     dim=-1)  # B, 3, H, W

    #                 gt_vis = th.cat(
    #                     [gt_img,
    #                         gt_depth.repeat_interleave(3, dim=1)],
    #                     dim=-1)  # TODO, fail to load depth. range [0, 1]

    #                 vis = th.cat([gt_vis, pred_vis], dim=-2)[0].permute(
    #                     1, 2, 0).cpu()  # ! pred in range[-1, 1]
    #                 # vis_grid = thvision.utils.make_grid(vis) # HWC
    #                 vis = vis.numpy() * 127.5 + 127.5
    #                 vis = vis.clip(0, 255).astype(np.uint8)
    #                 Image.fromarray(vis).save(
    #                     f'{logger.get_dir()}/{self.step+self.resume_step}.jpg')
    #                 logger.log(
    #                     'log vis to: ',
    #                     f'{logger.get_dir()}/{self.step+self.resume_step}.jpg')

    #                 # self.writer.add_image(f'images',
    #                 #                       vis,
    #                 #                       self.step + self.resume_step,
    #                 #                       dataformats='HWC')
    #         return pred
