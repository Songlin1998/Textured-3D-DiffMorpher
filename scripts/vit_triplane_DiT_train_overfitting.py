"""
Train a diffusion model on images.
"""
import json
import sys
import os


sys.path.append('.')
import torch.distributed as dist
from guided_diffusion.train_util import TrainLoop

import traceback

import torch as th
import torch.multiprocessing as mp
import numpy as np

from copy import deepcopy

import argparse
import dnnlib
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    args_to_dict,
    add_dict_to_argparser,
    diffusion_defaults,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_gaussian_diffusion
)
# from nsr.train_util_DiT import TrainLoop3DDiffusionDiT as TrainLoop
from nsr.train_util_DiT import TrainLoop3DDiffusionDiTOverfit as TrainLoop
# from nsr.train_util_diffusion import TrainLoop3DDiffusion as TrainLoop # all the same except the model
from nsr.script_util import create_3DAE_model, encoder_and_nsr_defaults, loss_defaults, rendering_options_defaults, DiT_defaults
from datasets.shapenet import load_data, load_eval_data, load_memory_data
from nsr.losses.builder import E3DGELossClass

from pdb import set_trace as st

from dit.dit_models import DiT_models

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

# th.backends.cuda.matmul.allow_tf32 = True # https://huggingface.co/docs/diffusers/optimization/fp16

# from torch.utils.tensorboard import SummaryWriter


def training_loop(args):

    SEED = args.global_seed

    # def training_loop(args):
    dist_util.setup_dist(args)

    # dist.init_process_group(backend='nccl', init_method='env://', rank=args.local_rank, world_size=th.cuda.device_count())
    print(f"{args.local_rank=} init complete")
    th.cuda.set_device(args.local_rank)
    th.cuda.empty_cache()

    th.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    # logger.configure(dir=args.logdir, format_strs=["tensorboard", "csv"])
    logger.configure(dir=args.logdir)

    logger.log("creating data loader...")
    # TODO, load shapenet data
    # data = load_data(
    if args.overfitting:
        logger.log("create overfitting memory dataset")
        data = load_memory_data(
            file_path=args.eval_data_dir,
            batch_size=args.batch_size,
            reso=args.image_size,
            reso_encoder=args.image_size_encoder,  # 224 -> 128
            num_workers=args.num_workers,
            load_depth=True  # for evaluation
        )
    else:
        logger.log("create all instances dataset")
        data = load_data(
            file_path=args.data_dir,
            batch_size=args.batch_size,
            reso=args.image_size,
            reso_encoder=args.image_size_encoder,  # 224 -> 128
            num_workers=args.num_workers,
            load_depth=True
            # load_depth=True # for evaluation
        )
    eval_data = load_eval_data(
        file_path=args.eval_data_dir,
        batch_size=args.eval_batch_size,
        reso=args.image_size,
        reso_encoder=args.image_size_encoder,  # 224 -> 128
        num_workers=args.num_workers,
        load_depth=True  # for evaluation
    )

    logger.log("creating ViT encoder and NSR decoder...")
    # device = dist_util.dev()
    device = th.device("cuda", args.local_rank)

    args.img_size = [args.image_size_encoder]
    logger.log("creating model and diffusion...")
    # * set denoise model args
    # args.denoise_in_channels = args.out_chans
    # args.denoise_out_channels = args.out_chans
    args.image_size = args.image_size_encoder  # 224, follow the triplane size

    # * create DiT diffusion model
    diffusion = create_gaussian_diffusion(
    **args_to_dict(args, diffusion_defaults().keys())
    )

    # latent_size = args.image_size_encoder // 8 # TODO, 
    latent_size = args.image_size_encoder # raw triplane diffusion
    denoise_model = DiT_models[args.dit_model](
        input_size=latent_size,
        num_classes=0, # unconditional synthesis
        in_channels=args.out_chans,
        # patch_size=args.dit_patch_size,
    )
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(denoise_model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    # model = DDP(model.to(device), device_ids=[rank])

    logger.info(f"DiT Parameters: {sum(p.numel() for p in denoise_model.parameters()):,}")

    # opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0) # TODO

    denoise_model.to(dist_util.dev())
    denoise_model.train()

    auto_encoder = create_3DAE_model(
        **args_to_dict(args,
                       encoder_and_nsr_defaults().keys()))

    auto_encoder.to(device)
    auto_encoder.eval()

    # let all processes sync up before starting with a new epoch of training

    if dist_util.get_rank() == 0:
        with open(os.path.join(args.logdir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

    args.schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion)

    opt = dnnlib.EasyDict(args_to_dict(args, loss_defaults().keys()))
    loss_class = E3DGELossClass(device, opt).to(device)

    logger.log("training...")
    dist_util.synchronize()
    TrainLoop(rec_model=auto_encoder,
              denoise_model=denoise_model,
              diffusion=diffusion,
              loss_class=loss_class,
              data=data,
              eval_data=eval_data,
              **vars(args)).run_loop()


def create_argparser(**kwargs):
    # defaults.update(model_and_diffusion_defaults())

    defaults = dict(
        use_amp=False,
        triplane_scaling_divider=10, # divide by this value
        global_seed=0,
        overfitting=False,
        num_workers=4,
        image_size=128,
        image_size_encoder=224,
        iterations=150000,
        schedule_sampler="uniform",
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
        use_fp16=False,
        fp16_scale_growth=1e-3,
        data_dir="",
        eval_data_dir="",
        # load_depth=False, # TODO
        logdir="/mnt/lustre/yslan/logs/nips23/",
        load_submodule_name='',  # for loading pretrained auto_encoder model
        ignore_resume_opt=False,
        freeze_ae=False,
        denoised_ae=True,
    )

    defaults.update(model_and_diffusion_defaults())
    defaults.update(encoder_and_nsr_defaults())  # type: ignore
    defaults.update(loss_defaults())

    defaults.update(DiT_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    # os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    # os.environ["NCCL_DEBUG"] = "INFO"
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.

    args = create_argparser().parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.gpus = th.cuda.device_count()

    args.rendering_kwargs = rendering_options_defaults(args)

    # Launch processes.
    print('Launching processes...')

    try:
        training_loop(args)
    except Exception as e:
        traceback.print_exc()
        dist_util.cleanup()  # clean port and socket when ctrl+c
