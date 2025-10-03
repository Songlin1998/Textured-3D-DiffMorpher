"""
Train a diffusion model on images.
"""
import json
import sys
import os

from tqdm import tqdm, trange
sys.path.append('.')
import torch.distributed as dist

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
from nsr.train_util_DiT import TrainLoop3DDiffusionDiT as TrainLoop
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

    denoise_model.load_state_dict(
        dist_util.load_state_dict(args.ddpm_model_path, map_location="cpu"))

    denoise_model.to(dist_util.dev())
    denoise_model.eval()

    auto_encoder = create_3DAE_model(
        **args_to_dict(args,
                       encoder_and_nsr_defaults().keys()))

    auto_encoder.load_state_dict(
        dist_util.load_state_dict(args.rec_model_path, map_location="cpu"))

    auto_encoder.to(device)
    auto_encoder.eval()

    # let all processes sync up before starting with a new epoch of training

    if dist_util.get_rank() == 0:
        with open(os.path.join(args.logdir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

    args.schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion)

    # opt = dnnlib.EasyDict(args_to_dict(args, loss_defaults().keys()))
    # loss_class = E3DGELossClass(device, opt).to(device)

    training_loop_class = TrainLoop(rec_model=auto_encoder,
                                    denoise_model=denoise_model,
                                    diffusion=diffusion,
                                    loss_class=None,
                                    data=None,
                                    eval_data=eval_data,
                                    **vars(args))


    logger.log("sampling...")
    dist_util.synchronize()

    # all_images = []
    # all_labels = []
    # while len(all_images) * args.batch_size < args.num_samples:
    for sample_idx in trange(args.num_samples):
        model_kwargs = {}

        # if args.class_cond:
        #     classes = th.randint(low=0,
        #                          high=NUM_CLASSES,
        #                          size=(args.batch_size, ),
        #                          device=dist_util.dev())
        #     model_kwargs["y"] = classes

        """ # classifier-free guidance, set up later for conditional generation
        # Labels to condition the model with (feel free to change):
        class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

        # Create sampling noise:
        n = len(class_labels)
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        y = torch.tensor(class_labels, device=device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        """

        sample_fn = (diffusion.p_sample_loop
                     if not args.use_ddim else diffusion.ddim_sample_loop)
        sample = sample_fn(
            denoise_model.forward_with_cfg_unconditional,
            (args.batch_size, args.out_chans, args.image_size,
             args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            noise=None,
            device=device
        )
        # st()

        # triplane_sampled = sample * triplane_std  # to triplane feature scale for rendering
        triplane_sampled = sample 

        training_loop_class.render_video_given_triplane(
            triplane_sampled, name_prefix=f'{sample_idx}')

def create_argparser(**kwargs):
    # defaults.update(model_and_diffusion_defaults())

    defaults = dict(
        # eval logging flags
        clip_denoised=False,
        num_samples=10,
        use_ddim=False,
        ddpm_model_path="",
        rec_model_path="",

        # model creation; training flags
        global_seed=0,
        overfitting=False,
        num_workers=1,
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
