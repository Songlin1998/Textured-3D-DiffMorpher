"""
Train a diffusion model on images.
"""
import json
import sys
import os

sys.path.append('.')
import torch as th
import torch.multiprocessing as mp
import numpy as np

from pathlib import Path
import torch.distributed as dist
import dnnlib
from dnnlib.util import EasyDict, InfiniteSampler
import traceback

import argparse
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    args_to_dict,
    add_dict_to_argparser,
)
import nsr
# from nsr.train_util_with_eg3d import TrainLoop3DRecEG3D as TrainLoop
from nsr.script_util import create_3DAE_model, encoder_and_nsr_defaults, loss_defaults
from datasets.shapenet import load_data, load_eval_data, load_memory_data
from nsr.losses.builder import E3DGELossClass

from torch.utils.data import Subset

from pdb import set_trace as st

from torch_utils import legacy, misc

from datasets.eg3d_dataset import init_dataset_kwargs

# th.backends.cuda.matmul.allow_tf32 = True # https://huggingface.co/docs/diffusers/optimization/fp16

SEED = 0


def training_loop(args):
    # def training_loop(args):
    dist_util.setup_dist(args)

    th.cuda.set_device(args.local_rank)
    th.cuda.empty_cache()

    th.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    # logger.configure(dir=args.logdir, format_strs=["tensorboard", "csv"])
    logger.configure(dir=args.logdir)

    logger.log("creating eg3d G...")

    common_kwargs = dict(c_dim=25, img_resolution=512, img_channels=3)

    G_kwargs = EasyDict(class_name=None,
                        z_dim=512,
                        w_dim=512,
                        mapping_kwargs=EasyDict())
    opts = EasyDict(
        dict(
            cbase=32768,
            cmax=512,
            map_depth=2,
            g_class_name='nsr.triplane.TriPlaneGenerator',  # TODO
            d_class_name='nsr.dual_discriminator.DualDiscriminator',
            g_num_fp16_res=0,
            disc_c_noise=0,
            freezed=0, 
            # mbstd_group=4,
            mbstd_group=args.batch_size, # shall be divisible by batch size
        ))
    
    # ! D kwargs
    D_kwargs = dnnlib.EasyDict(
        class_name='nsr.networks_stylegan2.Discriminator',
        block_kwargs=dnnlib.EasyDict(),
        mapping_kwargs=dnnlib.EasyDict(),
        epilogue_kwargs=dnnlib.EasyDict())

    D_kwargs.channel_base = opts.cbase
    D_kwargs.channel_max = opts.cmax
    D_kwargs.block_kwargs.freeze_layers = opts.freezed
    D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    D_kwargs.class_name = opts.d_class_name
    D_kwargs.disc_c_noise = opts.disc_c_noise  # Regularization for discriminator pose conditioning

    D = dnnlib.util.construct_class_by_name(
        **D_kwargs, **common_kwargs).train().requires_grad_(False).to(
            dist_util.dev())  # subclass of torch.nn.Module


    G_kwargs.channel_base = opts.cbase
    G_kwargs.channel_max = opts.cmax
    G_kwargs.mapping_kwargs.num_layers = opts.map_depth
    G_kwargs.class_name = opts.g_class_name
    G_kwargs.fused_modconv_default = 'inference_only'  # Speed up training by using regular convolutions instead of grouped convolutions.
    G_kwargs.rendering_kwargs = args.rendering_kwargs
    G_kwargs.num_fp16_res = 0
    G_kwargs.sr_num_fp16_res = 4

    G_kwargs.sr_kwargs = EasyDict(channel_base=opts.cbase,
                                  channel_max=opts.cmax,
                                  fused_modconv_default='inference_only',
                                  use_noise=True) # ! close noise injection? since noise_mode='none' in eg3d

    G_kwargs.num_fp16_res = opts.g_num_fp16_res
    G_kwargs.conv_clamp = 256 if opts.g_num_fp16_res > 0 else None

    # creating G
    G_ema = dnnlib.util.construct_class_by_name(
        **G_kwargs, **common_kwargs).train().requires_grad_(False).to(
            dist_util.dev())  # subclass of th.nn.Module

    # * load pretrained G model
    # with dnnlib.util.open_url(args.resume_checkpoint_EG3D) as f:
    #     resume_data = legacy.load_network_pkl(f, device=dist_util.dev())
    
    # th.save(resume_data, Path(args.resume_checkpoint_EG3D).parent / 'ffhq.ckpt')
    resume_data = th.load(args.resume_checkpoint_EG3D, map_location='cuda:{}'.format(args.local_rank))

    # d_load_except = []

    # for name, param in D.named_parameters():
    #     # if any([res in name for res in ('b32', 'b16', 'b8', 'b4')]):
    #     if any([res in name for res in ('b8', 'b4')]):  # mimic cvD
    #     # if any([res in name for res in ['b4']]):  # mimic cvD
    #         d_load_except.append(name)



    for name, module in [
        ('G_ema', G_ema),
        # ('D', D),
    ]:
        misc.copy_params_and_buffers(
            resume_data[name],  # type: ignore
            module,
            require_all=True,
            # load_except=d_load_except if name == 'D' else [],
            )
    
    del resume_data

    th.cuda.empty_cache()

    G_ema.requires_grad_(False)
    G_ema.eval()

    logger.log("creating encoder and NSR decoder...")

    if args.sr_training:
        args.sr_kwargs = G_kwargs.sr_kwargs # uncomment if needs to train with SR module

    auto_encoder = create_3DAE_model(
        **args_to_dict(args,
                       encoder_and_nsr_defaults().keys()))

    # * clone G_ema.decoder to auto_encoder triplane
    logger.log("AE triplane decoder reuses G_ema decoder...")
    auto_encoder.decoder.register_buffer('w_avg', G_ema.backbone.mapping.w_avg)

    auto_encoder.decoder.triplane_decoder.decoder.load_state_dict(  # type: ignore
        G_ema.decoder.state_dict())  # type: ignore

    # set grad=False in this manner suppresses the DDP forward no grad error.
    for param in auto_encoder.decoder.triplane_decoder.decoder.parameters(): # type: ignore
        param.requires_grad_(False)
    
    if args.sr_training:

        logger.log("AE triplane decoder reuses G_ema SR module...")
        auto_encoder.decoder.triplane_decoder.superresolution.load_state_dict(  # type: ignore
            G_ema.superresolution.state_dict())  # type: ignore

        # set grad=False in this manner suppresses the DDP forward no grad error.
        for param in auto_encoder.decoder.triplane_decoder.superresolution.parameters(): # type: ignore
            param.requires_grad_(False)

    # auto_encoder.decoder.triplane_decoder.decoder.requires_grad_(  # type: ignore
    #     False)  # type: ignore

    auto_encoder.to(dist_util.dev())
    auto_encoder.train()

    logger.log("creating data loader...")

    # ! load FFHQ
    # Training set.
    # training_set_kwargs, dataset_name = init_dataset_kwargs(data=args.data_dir, class_name='datasets.eg3d_dataset.ImageFolderDatasetPose') # only load pose here
    training_set_kwargs, dataset_name = init_dataset_kwargs(data=args.data_dir, class_name='datasets.eg3d_dataset.ImageFolderDataset') # only load pose here
    # if args.cond and not training_set_kwargs.use_labels:
    # raise Exception('check here')

    # training_set_kwargs.use_labels = args.cond
    training_set_kwargs.use_labels = True
    training_set_kwargs.xflip = True
    training_set_kwargs.random_seed = SEED
    # desc = f'{args.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}'

    # * construct ffhq/afhq dataset
    training_set = dnnlib.util.construct_class_by_name(
        **training_set_kwargs)  # subclass of training.dataset.Dataset

    training_set = dnnlib.util.construct_class_by_name(
        **training_set_kwargs)  # subclass of training.dataset.Dataset

    training_set_sampler = InfiniteSampler(
        dataset=training_set,
        rank=dist_util.get_rank(),
        num_replicas=dist_util.get_world_size(),
        seed=SEED)

    training_set_iterator = iter(
        th.utils.data.DataLoader(dataset=training_set,
                                 sampler=training_set_sampler,
                                 batch_size=args.batch_size,
                                 pin_memory=True,
                                 num_workers=args.num_workers,))
                                #  prefetch_factor=2))

    eval_data = th.utils.data.DataLoader(dataset=Subset(training_set, np.arange(10)),
                                 batch_size=args.eval_batch_size,
                                 num_workers=1)

    args.img_size = [args.image_size_encoder]

    # let all processes sync up before starting with a new epoch of training
    dist_util.synchronize()

    opt = EasyDict(args_to_dict(args, loss_defaults().keys()))
    loss_class = E3DGELossClass(dist_util.dev(), opt).to(dist_util.dev())

    TrainLoop = {
        'eg3d_trainer': nsr.TrainLoop3DRecEG3D,
        'hybrid_trainer': nsr.TrainLoop3DRecEG3DHybrid,
        'real_trainer': nsr.TrainLoop3DRecEG3DReal,
        'eg3d_d_trainer': nsr.TrainLoop3DRecEG3DHybridEG3DD,
        'TrainLoop3DRecEG3DRealOnl_D': nsr.TrainLoop3DRecEG3DRealOnl_D
    }[args.trainer_name]

    logger.log("training...")
    TrainLoop(G=G_ema,
              D=D,
              rec_model=auto_encoder,
              loss_class=loss_class,
              data=training_set_iterator,
              eval_data=eval_data,
              **vars(args)).run_loop()  # ! overfitting


def create_argparser(**kwargs):
    # defaults.update(model_and_diffusion_defaults())

    defaults = dict(
        trainer_name='eg3d_trainer',
        use_amp=False,
        overfitting=False,
        num_workers=4,
        image_size=128,
        image_size_encoder=224,
        iterations=150000,
        anneal_lr=False,
        lr=5e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        eval_batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        eval_interval=2500,
        save_interval=10000,
        resume_checkpoint="",
        resume_checkpoint_EG3D="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        data_dir="",
        eval_data_dir="",
        # load_depth=False, # TODO
        logdir="/mnt/lustre/yslan/logs/nips23/",
        adv_loss_start_iter=20000,
    )

    defaults.update(encoder_and_nsr_defaults())  # type: ignore
    defaults.update(loss_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    os.environ[
        "th_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.
    os.environ["th_CPP_LOG_LEVEL"] = "INFO"

    args = create_argparser().parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.gpus = th.cuda.device_count()

    opts = args

    rendering_options = {
        # 'image_resolution': c.training_set_kwargs.resolution,
        'image_resolution': 256,
        'disparity_space_sampling': False,
        'clamp_mode': 'softplus',
        'c_gen_conditioning_zero':
        True,  # if true, fill generator pose conditioning label with dummy zero vector
        # 'gpc_reg_prob': opts.gpc_reg_prob if opts.gen_pose_cond else None,
        'c_scale':
        opts.c_scale,  # mutliplier for generator pose conditioning label
        'superresolution_noise_mode': 'random',
        'density_reg': opts.density_reg,  # strength of density regularization
        'density_reg_p_dist': opts.
        density_reg_p_dist,  # distance at which to sample perturbed points for density regularization
        'reg_type': opts.
        reg_type,  # for experimenting with variations on density regularization
        # 'decoder_lr_mul':
        # opts.decoder_lr_mul,  # learning rate multiplier for decoder
        'sr_antialias': True,
        'return_triplane_features': False,  # for DDF supervision
        'return_sampling_details_flag': False,
    }

    if opts.cfg == 'ffhq':
        rendering_options.update({
            'superresolution_module':
            'nsr.superresolution.SuperresolutionHybrid8XDC',
            'focal': 2985.29 / 700,
            'depth_resolution':
            48,  # number of uniform samples to take per ray.
            # 36,  # number of uniform samples to take per ray.
            'depth_resolution_importance':
            48,  # number of importance samples to take per ray.
            # 36,  # number of importance samples to take per ray.
            'ray_start':
            2.25,  # near point along each ray to start taking samples.
            'ray_end':
            3.3,  # far point along each ray to stop taking samples. 
            'box_warp':
            1,  # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
            'avg_camera_radius':
            2.7,  # used only in the visualizer to specify camera orbit radius.
            'avg_camera_pivot': [
                0, 0, 0.2
            ],  # used only in the visualizer to control center of camera rotation.
        })
    elif opts.cfg == 'afhq':
        rendering_options.update({
            'superresolution_module':
            'nsr.superresolution.SuperresolutionHybrid8X',
            'focal': 4.2647,
            'depth_resolution': 48,
            'depth_resolution_importance': 48,
            'ray_start': 2.25,
            'ray_end': 3.3,
            'box_warp': 1,
            'avg_camera_radius': 2.7,
            'avg_camera_pivot': [0, 0, -0.06],
        })
    elif opts.cfg == 'shapenet':  # TODO, lies in a sphere
        rendering_options.update({
            'depth_resolution': 64,
            'depth_resolution_importance': 64,
            # * radius 1.2 setting, newly rendered images
            'ray_start': 0.2,
            'ray_end': 2.2,
            # 'ray_start': opts.ray_start,
            # 'ray_end': opts.ray_end,
            'box_warp': 2,  # TODO, how to set this value?
            'white_back': True,
            'avg_camera_radius': 1.2,
            'avg_camera_pivot': [0, 0, 0],
        })

    elif opts.cfg == 'shapenet_tuneray':  # TODO, lies in a sphere
        rendering_options.update({
            'depth_resolution': 64,
            'depth_resolution_importance': 64,
            # * radius 1.2 setting, newly rendered images
            'ray_start': opts.ray_start,
            'ray_end': opts.ray_end,
            'box_warp':
            opts.ray_end - opts.ray_start,  # TODO, how to set this value?
            'white_back': True,
            'avg_camera_radius': 1.2,
            'avg_camera_pivot': [0, 0, 0],
        })

    elif opts.cfg == 'shapenet_tuneray_wraphalf':  # TODO, lies in a sphere
        rendering_options.update({
            'depth_resolution': 64,
            'depth_resolution_importance': 64,
            # * radius 1.2 setting, newly rendered images
            'ray_start': opts.ray_start,
            'ray_end': opts.ray_end,
            'box_warp': (opts.ray_end - opts.ray_start) /
            2,  # TODO, how to set this value?
            'white_back': True,
            'avg_camera_radius': 1.2,
            'avg_camera_pivot': [0, 0, 0],
        })

    # return synthesized shapes for loss
    rendering_options.update({'return_sampling_details_flag': True})

    args.rendering_kwargs = rendering_options

    # print(args)
    with open(os.path.join(args.logdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Launch processes.
    print('Launching processes...')

    try:
        training_loop(args)
    # except KeyboardInterrupt as e:
    except Exception as e:
        # print(e)
        traceback.print_exc()
        dist_util.cleanup()  # clean port and socket when ctrl+c
