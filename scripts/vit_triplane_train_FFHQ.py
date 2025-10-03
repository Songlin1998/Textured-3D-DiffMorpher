"""
Train a diffusion model on images.
"""
import json
import sys
import os

sys.path.append('.')
import torch.distributed as dist

import traceback

import torch as th
import torch.multiprocessing as mp
import numpy as np

import argparse
import dnnlib
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    args_to_dict,
    add_dict_to_argparser,
)
from nsr.train_util import TrainLoop3DRec as TrainLoop
from nsr.script_util import create_3DAE_model, encoder_and_nsr_defaults, loss_defaults
# from datasets.shapenet import load_data, load_eval_data, load_memory_data
from datasets.eg3d_dataset import init_dataset_kwargs
from nsr.losses.builder import E3DGELossClass

from torch.utils.data.distributed import DistributedSampler
from dnnlib.util import InfiniteSampler

from pdb import set_trace as st
from torch.utils.data import RandomSampler, DataLoader, Subset

# th.backends.cuda.matmul.allow_tf32 = True # https://huggingface.co/docs/diffusers/optimization/fp16

SEED = 0


def training_loop(args):
    # def training_loop(args):
    dist_util.setup_dist(args)

    print(f"{args.local_rank=} init complete")
    th.cuda.set_device(args.local_rank)
    th.cuda.empty_cache()

    th.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    logger.configure(dir=args.logdir)

    logger.log("creating encoder and NSR decoder...")
    device = th.device("cuda", args.local_rank)

    auto_encoder = create_3DAE_model(
        **args_to_dict(args,
                       encoder_and_nsr_defaults().keys()))
    auto_encoder.to(device)
    auto_encoder.train()

    logger.log("creating data loader...")

    # ! load FFHQ
    # Training set.
    training_set_kwargs, dataset_name = init_dataset_kwargs(data=args.data_dir)
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
                                 num_workers=args.num_workers,
                                 prefetch_factor=2))
    #  num_workers=0))
    # ! how to set eval_data?
    # eval_data = training_set_iterator
    eval_data = iter(
        th.utils.data.DataLoader(
            dataset=Subset(training_set, np.arange(50)),
            batch_size=args.eval_batch_size,
            num_workers=1))

    # eval_data = load_eval_data(
    #     file_path=args.eval_data_dir,
    #     batch_size=args.eval_batch_size,
    #     reso=args.image_size,
    #     reso_encoder=args.image_size_encoder,  # 224 -> 128
    #     num_workers=args.num_workers,
    #     load_depth=True,  # for evaluation
    #     preprocess=auto_encoder.preprocess
    # )

    if dist_util.get_rank() == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print(f'Dataset path:        {training_set_kwargs.path}')
        print(f'Dataset size:        {training_set_kwargs.max_size} images')
        print(f'Dataset resolution:  {training_set_kwargs.resolution}')
        print(f'Dataset labels:      {training_set_kwargs.use_labels}')
        print(f'Dataset x-flips:     {training_set_kwargs.xflip}')

    # dry run data loading
    assert training_set_iterator is not None

    # if dist_util.get_rank() == 0:
    #     batch = next(training_set_iterator)
    #     print('dry run shape:',  batch['img_to_encoder'].shape, batch['img_to_encoder'].min(), batch['img_to_encoder'].max())

    args.img_size = [args.image_size_encoder]

    # let all processes sync up before starting with a new epoch of training
    dist_util.synchronize()

    opt = dnnlib.EasyDict(args_to_dict(args, loss_defaults().keys()))
    loss_class = E3DGELossClass(device, opt).to(device)

    logger.log("training...")
    TrainLoop(model=auto_encoder,
              loss_class=loss_class,
              data=training_set_iterator,
              eval_data=eval_data,
              **vars(args)).run_loop()  # ! overfitting


def create_argparser(**kwargs):

    defaults = dict(
        # data=''
        # gpus=1,
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
        eval_batch_size=12,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        eval_interval=2500,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        data_dir="",
        eval_data_dir="",
        # load_depth=False, # TODO
        logdir="/mnt/lustre/yslan/logs/nips23/",
    )

    defaults.update(encoder_and_nsr_defaults())  # type: ignore
    defaults.update(loss_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"

    args = create_argparser().parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.gpus = th.cuda.device_count()

    opts = args

    rendering_options = {
        # 'image_resolution': c.training_set_kwargs.resolution,
        'image_resolution': 256,
        'disparity_space_sampling': False,
        'clamp_mode': 'softplus',
        # 'superresolution_module': sr_module,
        # 'c_gen_conditioning_zero': not opts.
        # gen_pose_cond,  # if true, fill generator pose conditioning label with dummy zero vector
        # 'gpc_reg_prob': opts.gpc_reg_prob if opts.gen_pose_cond else None,
        'c_scale':
        opts.c_scale,  # mutliplier for generator pose conditioning label
        # 'superresolution_noise_mode': opts.
        # sr_noise_mode,  # [random or none], whether to inject pixel noise into super-resolution layers
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
            'focal': 2985.29 / 700,
            'depth_resolution':
            48,  # number of uniform samples to take per ray.
            'depth_resolution_importance':
            48,  # number of importance samples to take per ray.
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

    args.rendering_kwargs = rendering_options

    args.num_gpus = dist_util.get_world_size()

    # print(args)
    with open(os.path.join(args.logdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Launch processes.
    print('Launching processes...')

    # try:
    # training_loop(args)

    # if dist_util.get_world_size() == 1:
    training_loop(args=args)
    # else:
        # th.multiprocessing.spawn(fn=training_loop,
                                #  args=(args, ),
                                #  nprocs=dist_util.get_world_size())

    # except KeyboardInterrupt as e:
    # except Exception as e:
    #     # print(e)
    #     traceback.print_exc()
    #     dist_util.cleanup()  # clean port and socket when ctrl+c
