"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import pickle
import argparse
import json
import sys
import os

sys.path.append('.')

from pdb import set_trace as st
import imageio
import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from tqdm import tqdm, trange

# from nsr.train_util_diffusion import TrainLoop3DDiffusion as TrainLoop
import nsr

from nsr.script_util import create_3DAE_model, encoder_and_nsr_defaults, loss_defaults, AE_with_Diffusion, rendering_options_defaults

from datasets.shapenet import load_data, load_eval_data, load_memory_data, load_data_dryrun
# from datasets.shapenet import load_eval_data


def main():
    args = create_argparser().parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.gpus = th.cuda.device_count()

    args.rendering_kwargs = rendering_options_defaults(args)

    dist_util.setup_dist(args)
    logger.configure(dir=args.logdir)

    # * set denoise model args
    logger.log("creating model and diffusion...")
    args.img_size = [args.image_size_encoder]
    args.denoise_in_channels = args.out_chans
    args.denoise_out_channels = args.out_chans
    args.image_size = args.image_size_encoder  # 224, follow the triplane size

    ddpm_model, diffusion = create_model_and_diffusion(
        **args_to_dict(args,
                       model_and_diffusion_defaults().keys()))

    ddpm_model.load_state_dict(
        dist_util.load_state_dict(args.ddpm_model_path, map_location="cpu"))
    ddpm_model.to(dist_util.dev())
    if args.use_fp16:
        ddpm_model.convert_to_fp16()
    ddpm_model.eval()

    # * auto-encoder reconstruction model
    logger.log("creating 3DAE...")
    auto_encoder = create_3DAE_model(
        **args_to_dict(args,
                       encoder_and_nsr_defaults().keys()))

    auto_encoder.load_state_dict(
        dist_util.load_state_dict(args.rec_model_path, map_location="cpu"))

    auto_encoder.to(dist_util.dev())
    auto_encoder.eval()

    # triplane_std = 50  # TODO
    # triplane_std = 10  # TODO

    # TODO, how to set the scale?

    logger.log("create dataset")

    data = load_data_dryrun(
        file_path=args.data_dir,
        # batch_size=args.batch_size,
        batch_size=128,
        # batch_size=1,
        reso=args.image_size,
        reso_encoder=args.image_size_encoder,  # 224 -> 128
        num_workers=args.num_workers,
        load_depth=True,
        preprocess=auto_encoder.preprocess # clip
        # load_depth=True # for evaluation
    )

    foreground_percent = []
    fnames = []
    # foreground_percent = {}
    for idx, batch in enumerate(tqdm(data)):
        resolution = batch['depth_mask'].shape[-1]
        batch_size = batch['depth_mask'].shape[0]
        for i in range(batch_size):
            foreground_percent.append(
            batch['depth_mask'][i].sum() / resolution ** 2
            )
            fnames.append(batch['rgb_fname'][i])
            # foreground_percent[batch['rgb_fname']] = {}
    
    with open(logger.get_dir() + '/fg_percent.pkl', 'wb') as f: # do this for both car and plane
        pickle.dump({
            'fg_percent': foreground_percent,
            'fnames': fnames
        }, f)
        print(logger.get_dir() + '/fg_percent.pkl')

def create_argparser():
    defaults = dict(
        trainer_name='adm',
        use_amp=False,
        triplane_scaling_divider=10, # divide by this value

        image_size_encoder=224,

        # * sampling flags
        clip_denoised=False,
        num_samples=10,
        use_ddim=False,
        ddpm_model_path="",
        rec_model_path="",

        # * eval logging flags
        logdir="/mnt/lustre/yslan/logs/nips23/",
        data_dir="",
        eval_data_dir="",
        eval_batch_size=1,
        num_workers=1,

        # * training flags for loading TrainingLoop class
        overfitting=False,
        image_size=128,
        iterations=150000,
        schedule_sampler="uniform",
        anneal_lr=False,
        lr=5e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        eval_interval=2500,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        load_submodule_name='',  # for loading pretrained auto_encoder model
        ignore_resume_opt=False,
        freeze_ae=False,
        denoised_ae=True,
    )

    defaults.update(model_and_diffusion_defaults())
    defaults.update(encoder_and_nsr_defaults())  # type: ignore
    defaults.update(loss_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    # os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    # os.environ["NCCL_DEBUG"] = "INFO"

    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.

    main()
