import json
import sys
import os

sys.path.append('.')

# from dnnlib import EasyDict
import traceback

import torch as th
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np

import argparse
import dnnlib
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    args_to_dict,
    add_dict_to_argparser,
    continuous_diffusion_defaults,
    control_net_defaults,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from guided_diffusion.continuous_diffusion import make_diffusion as make_sde_diffusion
import nsr
import nsr.lsgm


from datasets.eg3d_dataset import LMDBDataset_MV_Compressed_eg3d
from nsr.script_util import create_3DAE_model, encoder_and_nsr_defaults, loss_defaults, rendering_options_defaults, eg3d_options_default, dataset_defaults
from datasets.shapenet import load_data, load_eval_data, load_memory_data
from nsr.losses.builder import E3DGELossClass
import torch
from torch_utils import legacy, misc
from torch.utils.data import Subset
from pdb import set_trace as st

from dnnlib.util import EasyDict, InfiniteSampler
from datasets.eg3d_dataset import init_dataset_kwargs

th.backends.cudnn.enabled = True 
th.backends.cudnn.benchmark = True

from transport import create_transport, Sampler
from transport.train_utils import parse_transport_args
from nsr.camera_utils import generate_input_camera, uni_mesh_path, sample_uniform_cameras_on_sphere
import trimesh

import copy

try:
    from xformers.components.feedforward.fused_mlp import FusedMLP
    print("Module imported successfully.")
except ImportError:
    print("Module failed to import.")

from peft import LoraConfig, get_peft_model


SEED = 0

import random

from scipy.stats import beta as beta_distribution

def generate_beta_tensor(
    size: int, alpha: float = 1, beta: float = 1
) -> torch.FloatTensor:
    """
    Assume size as n
    Generates a PyTorch tensor of values [x0, x1, ..., xn-1] for the Beta distribution
    where each xi satisfies F(xi) = i/(n-1) for the CDF F of the Beta distribution.

    Args:
        size (int): The number of values to generate.
        alpha (float): The alpha parameter of the Beta distribution.
        beta (float): The beta parameter of the Beta distribution.

    Returns:
        torch.Tensor: A tensor of the inverse CDF values of the Beta distribution.
    """
    # Generating the inverse CDF values
    prob_values = [i / (size - 1) for i in range(size)]
    inverse_cdf_values = beta_distribution.ppf(prob_values, alpha, beta)

    # Converting to a PyTorch tensor
    return torch.tensor(inverse_cdf_values, dtype=torch.float32)

def load_my_latent(ins,rand_pick_one=False, pick_both=False):

    
    idx = random.choice([1])
    latent = np.load(os.path.join(ins, f'latent-{idx}.npz'))  # pre-calculated VAE latent
    
    latent, fps_xyz = latent['latent_normalized'], latent['query_pcd_xyz'] # 2,768,16; 2,768,3

    
    rand_idx = 0
    latent, fps_xyz = latent[rand_idx:rand_idx+1], fps_xyz[rand_idx:rand_idx+1]
    
    return latent, fps_xyz

def normalize_pcd_act(x):
    
    xyz_std = 0.164
        
    return x / xyz_std

def interpolate_model_params(root, model_src_path, model_tgt_path, alpha, stage, keywords=['to_k', 'to_q', 'to_v', 'qkv']):
        
        src_state_dict = torch.load(model_src_path)
        tgt_state_dict = torch.load(model_tgt_path)
        
        
        new_state_dict = {}

        for param_name in src_state_dict.keys():
            
            if any(keyword in param_name for keyword in keywords):
                
                src_param = src_state_dict[param_name]
                tgt_param = tgt_state_dict[param_name]

                
                interpolated_param = (1 - alpha) * src_param + alpha * tgt_param
                new_state_dict[param_name] = interpolated_param
            else:
                
                new_state_dict[param_name] = src_state_dict[param_name]

        
        stage_path = os.path.join(root, f'morphing_{stage}')
        os.makedirs(stage_path,exist_ok=True)
        interpolated_model_path = os.path.join(stage_path, f'model_joint_denoise_rec_model0000051.pt')
        torch.save(new_state_dict, interpolated_model_path)

        return interpolated_model_path    

def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )
    
def linear(z1, z2, alpha):
    return (1-alpha)*z1+alpha*z2

def get_inversed_zs(args, prompt, latent_folder, args_1_resume_ckpt, args_2_resume_ckpt):

    args_1 = copy.deepcopy(args)
    args_2 = copy.deepcopy(args)
    logger.log("dist setup...")
    th.autograd.set_detect_anomaly(False) 
    th.cuda.set_device(
        args.local_rank)  # set this line to avoid extra memory on rank 0
    th.cuda.empty_cache()
    th.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    dist_util.setup_dist(args)
    th.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    th.backends.cudnn.allow_tf32 = args.allow_tf32
    logger.configure(dir=args.logdir)
    TrainLoop = nsr.lsgm.flow_matching_trainer.FlowMatchingEngine_gs_t23d
    
    # camera
    azimuths = []
    elevations = []
    frame_number = 10
    for i in range(frame_number): # 1030 * 5 * 10, for FID 50K
        azi, elevation = sample_uniform_cameras_on_sphere()
        # azi, elevation = azi[0] / np.pi * 180, elevation[0] / np.pi * 180
        azi, elevation = azi[0] / np.pi * 180, (elevation[0]-np.pi*0.5) / np.pi * 180 # [-0.5 pi, 0.5 pi]
        azimuths.append(azi)
        elevations.append(elevation)
    azimuths = np.array(azimuths)
    elevations = np.array(elevations)
    zero123pp_pose, _ = generate_input_camera(1.8, [[elevations[i], azimuths[i]] for i in range(frame_number)], fov=30)
    K = th.Tensor([1.3889, 0.0000, 0.5000, 0.0000, 1.3889, 0.5000, 0.0000, 0.0000, 0.0039]).to(zero123pp_pose) # keeps the same
    camera = th.cat([zero123pp_pose.reshape(frame_number,-1), K.unsqueeze(0).repeat(frame_number,1)], dim=-1)
    
    # data
    if args.objv_dataset:
        from datasets.g_buffer_objaverse import load_data, load_eval_data, load_memory_data, load_wds_data, load_data_cls
    else:  # shapenet
        from datasets.shapenet import load_data, load_eval_data, load_memory_data
    if args.i23d:
        data = load_eval_data(
            file_path=args.eval_data_dir,
            batch_size=args.eval_batch_size,
            reso=args.image_size,
            reso_encoder=args.image_size_encoder,  # 224 -> 128
            num_workers=args.num_workers,
            load_depth=True,  # for evaluation
            preprocess=auto_encoder.preprocess,
            **args_to_dict(args,
                            dataset_defaults().keys()))
    else:
        data = None # t23d sampling, only caption required
    eval_dataset = None 
    
    # stage1
    args_1.resume_checkpoint = args_1_resume_ckpt
    dist_util.setup_dist(args_1)
    logger.log("creating ViT encoder and NSR decoder...")
    device = dist_util.dev()
    args_1.img_size = [args_1.image_size_encoder]
    logger.log("creating model and diffusion...")
    if args_1.denoise_in_channels == -1:
        args_1.diffusion_input_size = args_1.image_size_encoder
        args_1.denoise_in_channels = args_1.out_chans
        args_1.denoise_out_channels = args_1.out_chans
    else:
        assert args_1.denoise_out_channels != -1
    if args_1.pred_type == 'v':  # for lsgm training
        assert args_1.predict_v == True  # for DDIM sampling
    denoise_model, diffusion = create_model_and_diffusion(
        **args_to_dict(args_1,
                       model_and_diffusion_defaults().keys()))
     # lora微调
    loraconfig = LoraConfig(
        r=4,
        lora_alpha=0.8,
        init_lora_weights="gaussian",
        target_modules=['to_k', 'to_q', 'to_v','qkv'],
    )
    denoise_model = get_peft_model(denoise_model, loraconfig).base_model.model
    opts = eg3d_options_default()
    if args_1.sr_training:
        args_1.sr_kwargs = dnnlib.EasyDict(
            channel_base=opts.cbase,
            channel_max=opts.cmax,
            fused_modconv_default='inference_only',
            use_noise=True
        )  # ! close noise injection? since noise_mode='none' in eg3d
    logger.log("creating encoder and NSR decoder...")
    auto_encoder = create_3DAE_model(
        **args_to_dict(args_1,
                       encoder_and_nsr_defaults().keys()))
    auto_encoder.to(device)
    auto_encoder.eval()
    logger.log("creating data loader...")
    if dist_util.get_rank() == 0:
        with open(os.path.join(args_1.logdir, 'args_1.json'), 'w') as f_1:
            json.dump(vars(args_1), f_1, indent=2)
    args_1.schedule_sampler = create_named_schedule_sampler(
        args_1.schedule_sampler, diffusion)
    opt = dnnlib.EasyDict(args_to_dict(args_1, loss_defaults().keys()))
    loss_class = E3DGELossClass(device, opt).to(device)
    sde_diffusion = None
    controlNet = None
    denoise_model.to(dist_util.dev())
    denoise_model.eval()
    auto_encoder.decoder.rendering_kwargs = args_1.rendering_kwargs
    training_loop_stage1 =TrainLoop(rec_model=auto_encoder,
              denoise_model=denoise_model,
              control_model=controlNet,
              diffusion=diffusion,
              sde_diffusion=sde_diffusion,
              loss_class=loss_class,
              data=data,
            #   eval_data=None,
              eval_data=eval_dataset, # return dataset
              **vars(args_1))
    
    # stage2
    args_2.triplane_scaling_divider = 0.25
    args_2.denoise_in_channels = 10
    args_2.denoise_out_channels = 10
    args_2.dit_model_arch="DiT-PCD-L-stage2-xyz2feat"
    args_2.ae_classname='vit.vit_triplane.pcd_structured_latent_space_lion_learnoffset_surfel_novaePT_sr_cascade_x8x4x4_512'
    args_2.resume_checkpoint = args_2_resume_ckpt
    args_2.snr_type = 'stage2-t23d'
    dist_util.setup_dist(args_2)
    logger.log("creating ViT encoder and NSR decoder...")
    device = dist_util.dev()
    args_2.img_size = [args_2.image_size_encoder]
    logger.log("creating model and diffusion...")
    if args_2.denoise_in_channels == -1:
        args_2.diffusion_input_size = args_2.image_size_encoder
        args_2.denoise_in_channels = args_2.out_chans
        args_2.denoise_out_channels = args_2.out_chans
    else:
        assert args_2.denoise_out_channels != -1
    if args_2.pred_type == 'v':  # for lsgm training
        assert args_2.predict_v == True  # for DDIM sampling
    denoise_model, diffusion = create_model_and_diffusion(
        **args_to_dict(args_2,
                       model_and_diffusion_defaults().keys()))
    # lora微调
    loraconfig = LoraConfig(
        r=4,
        lora_alpha=0.8,
        init_lora_weights="gaussian",
        target_modules=['to_k', 'to_q', 'to_v','qkv'],
    )
    denoise_model = get_peft_model(denoise_model, loraconfig).base_model.model
    opts = eg3d_options_default()
    if args_2.sr_training:
        args_2.sr_kwargs = dnnlib.EasyDict(
            channel_base=opts.cbase,
            channel_max=opts.cmax,
            fused_modconv_default='inference_only',
            use_noise=True
        )  # ! close noise injection? since noise_mode='none' in eg3d
    logger.log("creating encoder and NSR decoder...")
    auto_encoder = create_3DAE_model(
        **args_to_dict(args_2,
                       encoder_and_nsr_defaults().keys()))
    auto_encoder.to(device)
    auto_encoder.eval()
    logger.log("creating data loader...")
    
    if dist_util.get_rank() == 0:
        with open(os.path.join(args_2.logdir, 'args_2.json'), 'w') as f_2:
            json.dump(vars(args_2), f_2, indent=2)
    args_2.schedule_sampler = create_named_schedule_sampler(
        args_2.schedule_sampler, diffusion)
    opt = dnnlib.EasyDict(args_to_dict(args_2, loss_defaults().keys()))
    loss_class = E3DGELossClass(device, opt).to(device)
    sde_diffusion = None
    controlNet = None
    denoise_model.to(dist_util.dev())
    denoise_model.eval()
    auto_encoder.decoder.rendering_kwargs = args_2.rendering_kwargs
    training_loop_stage2 =TrainLoop(rec_model=auto_encoder,
              denoise_model=denoise_model,
              control_model=controlNet,
              diffusion=diffusion,
              sde_diffusion=sde_diffusion,
              loss_class=loss_class,
              data=data,
            #   eval_data=None,
              eval_data=eval_dataset, # return dataset
              **vars(args_2))
    
    # 获取inverse后的
    def get_zs_reversed(latent_folder, caption):
        
        latent, fps_xyz = load_my_latent(latent_folder, True) # analyzing xyz/latent disentangled diffusion
        latent, fps_xyz = latent[0], fps_xyz[0] # remove batch dim here
        normalized_fps_xyz = normalize_pcd_act(fps_xyz)
        
        fps_xyz = th.tensor(fps_xyz).unsqueeze(0)
        samples_1 = th.tensor(normalized_fps_xyz).unsqueeze(0).to('cuda')
        samples_2 = th.tensor(latent).unsqueeze(0).to('cuda')
        
        # 正向合成latent code
        # stage1
        N_1 = 1
        batch_c_1 = {'caption': caption}
        c_1, uc_1 = training_loop_stage1.get_condition(batch_c_1)
        batch_c_1.update({
            'fps-xyz': fps_xyz.to(training_loop_stage2.dtype).to(dist_util.dev())
        })
        # stage2
        N_2= 1
        c_2, uc_2 = training_loop_stage2.get_condition(batch_c_1)
    
        zs_2_reversed = training_loop_stage2.get_reverse(zs=samples_2,c=c_2,uc=uc_2,N=N_2)
        zs_1_reversed = training_loop_stage1.get_reverse(zs=samples_1,c=c_1,uc=uc_1,N=N_1)
        
        return zs_1_reversed, zs_2_reversed, c_1, uc_1, c_2, uc_2, samples_1, samples_2
    
    prompt_src = prompt
    
    zs_1_reversed, zs_2_reversed, c_1, uc_1, c_2, uc_2, samples_1, samples_2 = get_zs_reversed(
        latent_folder=latent_folder,
        caption=prompt_src,
    )
    
    dist_util.synchronize()
    logger.log('sampling complete')
    
    return zs_1_reversed, zs_2_reversed, c_1, uc_1, c_2, uc_2

def morphing_loop(args, prompt_src, prompt_tgt, args_1_resume_ckpt, args_2_resume_ckpt, idx, zs_1_src, zs_2_src, zs_1_tgt, zs_2_tgt, alpha,
                  c_1_src_, uc_1_src_, c_2_src_, uc_2_src_,c_1_tgt_, uc_1_tgt_, c_2_tgt_, uc_2_tgt_,use_attn):

    # if alpha == 0 or alpha == 1:
    #     use_attn = False
    # else:
    #     use_attn = True
    
    print(alpha, ': ', use_attn)
    
    args_1 = copy.deepcopy(args)
    args_2 = copy.deepcopy(args)
    logger.log("dist setup...")
    th.autograd.set_detect_anomaly(False) 
    th.cuda.set_device(
        args.local_rank)  # set this line to avoid extra memory on rank 0
    th.cuda.empty_cache()
    th.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    dist_util.setup_dist(args)
    th.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    th.backends.cudnn.allow_tf32 = args.allow_tf32
    logger.configure(dir=args.logdir)
    TrainLoop = nsr.lsgm.flow_matching_trainer.FlowMatchingEngine_gs_t23d
    
    # camera
    azimuths = []
    elevations = []
    frame_number = 10
    for i in range(frame_number): # 1030 * 5 * 10, for FID 50K
        azi, elevation = sample_uniform_cameras_on_sphere()
        # azi, elevation = azi[0] / np.pi * 180, elevation[0] / np.pi * 180
        azi, elevation = azi[0] / np.pi * 180, (elevation[0]-np.pi*0.5) / np.pi * 180 # [-0.5 pi, 0.5 pi]
        azimuths.append(azi)
        elevations.append(elevation)
    azimuths = np.array(azimuths)
    elevations = np.array(elevations)
    zero123pp_pose, _ = generate_input_camera(1.8, [[elevations[i], azimuths[i]] for i in range(frame_number)], fov=30)
    K = th.Tensor([1.3889, 0.0000, 0.5000, 0.0000, 1.3889, 0.5000, 0.0000, 0.0000, 0.0039]).to(zero123pp_pose) # keeps the same
    camera = th.cat([zero123pp_pose.reshape(frame_number,-1), K.unsqueeze(0).repeat(frame_number,1)], dim=-1)
    
    # data
    if args.objv_dataset:
        from datasets.g_buffer_objaverse import load_data, load_eval_data, load_memory_data, load_wds_data, load_data_cls
    else:  # shapenet
        from datasets.shapenet import load_data, load_eval_data, load_memory_data
    if args.i23d:
        data = load_eval_data(
            file_path=args.eval_data_dir,
            batch_size=args.eval_batch_size,
            reso=args.image_size,
            reso_encoder=args.image_size_encoder,  # 224 -> 128
            num_workers=args.num_workers,
            load_depth=True,  # for evaluation
            preprocess=auto_encoder.preprocess,
            **args_to_dict(args,
                            dataset_defaults().keys()))
    else:
        data = None # t23d sampling, only caption required
    eval_dataset = None 
    
    # stage1: morphing
    # args_1.resume_checkpoint = '/mnt/slurm_home/slyang/projects/gaussian-anything/train-finetune-stage1/model_joint_denoise_rec_model0000051.pt'
    args_1.resume_checkpoint = args_1_resume_ckpt
    dist_util.setup_dist(args_1)
    logger.log("creating ViT encoder and NSR decoder...")
    device = dist_util.dev()
    args_1.img_size = [args_1.image_size_encoder]
    logger.log("creating model and diffusion...")
    if args_1.denoise_in_channels == -1:
        args_1.diffusion_input_size = args_1.image_size_encoder
        args_1.denoise_in_channels = args_1.out_chans
        args_1.denoise_out_channels = args_1.out_chans
    else:
        assert args_1.denoise_out_channels != -1
    if args_1.pred_type == 'v':  # for lsgm training
        assert args_1.predict_v == True  # for DDIM sampling
    denoise_model, diffusion = create_model_and_diffusion(
        **args_to_dict(args_1,
                    model_and_diffusion_defaults().keys()))
    opts = eg3d_options_default()
    if args_1.sr_training:
        args_1.sr_kwargs = dnnlib.EasyDict(
            channel_base=opts.cbase,
            channel_max=opts.cmax,
            fused_modconv_default='inference_only',
            use_noise=True
        )  # ! close noise injection? since noise_mode='none' in eg3d
    logger.log("creating encoder and NSR decoder...")
    auto_encoder = create_3DAE_model(
        **args_to_dict(args_1,
                    encoder_and_nsr_defaults().keys()))
    auto_encoder.to(device)
    auto_encoder.eval()
    logger.log("creating data loader...")
    if dist_util.get_rank() == 0:
        with open(os.path.join(args_1.logdir, 'args_1.json'), 'w') as f_1:
            json.dump(vars(args_1), f_1, indent=2)
    args_1.schedule_sampler = create_named_schedule_sampler(
        args_1.schedule_sampler, diffusion)
    opt = dnnlib.EasyDict(args_to_dict(args_1, loss_defaults().keys()))
    loss_class = E3DGELossClass(device, opt).to(device)
    sde_diffusion = None
    controlNet = None
    # lora微调
    loraconfig = LoraConfig(
        r=4,
        lora_alpha=0.8,
        init_lora_weights="gaussian",
        target_modules=['to_k', 'to_q', 'to_v','qkv'],
    )
    denoise_model = get_peft_model(denoise_model, loraconfig).base_model.model
    denoise_model.to(dist_util.dev())
    denoise_model.eval()
    auto_encoder.decoder.rendering_kwargs = args_1.rendering_kwargs
    training_loop_stage1 =TrainLoop(rec_model=auto_encoder,
            denoise_model=denoise_model,
            control_model=controlNet,
            diffusion=diffusion,
            sde_diffusion=sde_diffusion,
            loss_class=loss_class,
            data=data,
            #   eval_data=None,
            eval_data=eval_dataset, # return dataset
            **vars(args_1))
    
    # stage2: morphing
    args_2.triplane_scaling_divider = 0.25
    args_2.denoise_in_channels = 10
    args_2.denoise_out_channels = 10
    args_2.dit_model_arch="DiT-PCD-L-stage2-xyz2feat"
    args_2.ae_classname='vit.vit_triplane.pcd_structured_latent_space_lion_learnoffset_surfel_novaePT_sr_cascade_x8x4x4_512'
    # args_2.resume_checkpoint = '/mnt/slurm_home/slyang/projects/gaussian-anything/train-finetune-stage2/model_joint_denoise_rec_model0000051.pt'
    args_2.resume_checkpoint = args_2_resume_ckpt
    args_2.snr_type = 'stage2-t23d'
    dist_util.setup_dist(args_2)
    logger.log("creating ViT encoder and NSR decoder...")
    device = dist_util.dev()
    args_2.img_size = [args_2.image_size_encoder]
    logger.log("creating model and diffusion...")
    if args_2.denoise_in_channels == -1:
        args_2.diffusion_input_size = args_2.image_size_encoder
        args_2.denoise_in_channels = args_2.out_chans
        args_2.denoise_out_channels = args_2.out_chans
    else:
        assert args_2.denoise_out_channels != -1
    if args_2.pred_type == 'v':  # for lsgm training
        assert args_2.predict_v == True  # for DDIM sampling
    denoise_model, diffusion = create_model_and_diffusion(
        **args_to_dict(args_2,
                    model_and_diffusion_defaults().keys()))
    opts = eg3d_options_default()
    if args_2.sr_training:
        args_2.sr_kwargs = dnnlib.EasyDict(
            channel_base=opts.cbase,
            channel_max=opts.cmax,
            fused_modconv_default='inference_only',
            use_noise=True
        )  # ! close noise injection? since noise_mode='none' in eg3d
    logger.log("creating encoder and NSR decoder...")
    auto_encoder = create_3DAE_model(
        **args_to_dict(args_2,
                    encoder_and_nsr_defaults().keys()))
    auto_encoder.to(device)
    auto_encoder.eval()
    logger.log("creating data loader...")
    
    if dist_util.get_rank() == 0:
        with open(os.path.join(args_2.logdir, 'args_2.json'), 'w') as f_2:
            json.dump(vars(args_2), f_2, indent=2)
    args_2.schedule_sampler = create_named_schedule_sampler(
        args_2.schedule_sampler, diffusion)
    opt = dnnlib.EasyDict(args_to_dict(args_2, loss_defaults().keys()))
    loss_class = E3DGELossClass(device, opt).to(device)
    sde_diffusion = None
    controlNet = None
    # lora微调
    loraconfig = LoraConfig(
        r=4,
        lora_alpha=0.8,
        init_lora_weights="gaussian",
        target_modules=['to_k', 'to_q', 'to_v','qkv'],
    )
    denoise_model = get_peft_model(denoise_model, loraconfig).base_model.model
    denoise_model.to(dist_util.dev())
    denoise_model.eval()
    auto_encoder.decoder.rendering_kwargs = args_2.rendering_kwargs
    training_loop_stage2 =TrainLoop(rec_model=auto_encoder,
            denoise_model=denoise_model,
            control_model=controlNet,
            diffusion=diffusion,
            sde_diffusion=sde_diffusion,
            loss_class=loss_class,
            data=data,
            #   eval_data=None,
            eval_data=eval_dataset, # return dataset
            **vars(args_2))
    
    # stage1
    N_1= 1
    batch_c_1_src = {'caption': prompt_src}
    batch_c_1_tgt = {'caption': prompt_tgt}
    batch_c_1 = {'caption': prompt_src+'_'+prompt_tgt+'_'+str(alpha.item())}
    c_1_src, uc_1_src = training_loop_stage1.get_condition(batch_c_1_src)
    c_1_tgt, uc_1_tgt = training_loop_stage1.get_condition(batch_c_1_tgt)
    
    # stage1的condition的morphing
    c_1 = {}
    uc_1 = {}
    
    for key in c_1_src.keys():
        c_1_morphed_val = linear(c_1_src[key],c_1_tgt[key],alpha)
        # c_1_morphed_val = frequency_domain_interpolation(c_1_src[key].to(torch.float32),c_1_tgt[key].to(torch.float32),alpha)
        c_1.update({key: c_1_morphed_val})
        uc_1_morphed_val = linear(uc_1_src[key],uc_1_tgt[key],alpha)
        # uc_1_morphed_val = frequency_domain_interpolation(uc_1_src[key].to(torch.float32),uc_1_tgt[key].to(torch.float32),alpha)
        uc_1.update({key: uc_1_morphed_val})
    
    zs_1 = slerp(zs_1_src,zs_1_tgt,alpha)
    
    # z_shape = (N_1, 768, training_loop_stage1.ddpm_model.in_channels)
    # zs_1 = training_loop_stage1.get_noise(batch_size=N_1,shape=z_shape[1:],seed=33)
    # print('------------0------------:',zs_1-zs_1_src)
    samples_1 = training_loop_stage1.get_samples(zs=zs_1,c=c_1, uc=uc_1, N=N_1,use_attn=use_attn,alpha=alpha,stage=1,
                                                 zs_src=zs_1_src,cond_src=c_1_src_,uc_src=uc_1_src_,
                                                 zs_tgt=zs_1_tgt,cond_tgt=c_1_tgt_,uc_tgt=uc_1_tgt_) # 最后一个为num_samples
    point_path = training_loop_stage1.save_results(idx=idx,
                                        batch_c = batch_c_1,
                                        samples=samples_1,
                                        save_img = args.save_img,
                                        camera=camera,
                                        latent_key='no')
    
    # 取stage1的cloud point (取到的就是morphed的point cloud)
    fps_xyz = torch.from_numpy(trimesh.load(point_path).vertices).clip(-0.45,0.45).unsqueeze(0) # torch.Size([1, 768, 3])
    edited_fps_xyz = fps_xyz.clone() # B N 3
    z_dim_coord = edited_fps_xyz[..., 2]
    edited_fps_xyz[..., 2] = th.where(z_dim_coord>0.24, z_dim_coord+0.075, z_dim_coord)
    batch_c_1.update({
        'fps-xyz': fps_xyz.to(training_loop_stage2.dtype).to(dist_util.dev())
    })
    
    # stage2
    N_2= 1
    c_2, uc_2 = training_loop_stage2.get_condition(batch_c_1)
    
    # stage2的condition的morphing
    for key in c_1.keys():
        c_2[key] = c_1[key]
        uc_2[key] = uc_1[key]
    
    zs_2 = slerp(zs_2_src,zs_2_tgt,alpha)
    
    # z_shape = (N_2, 768, training_loop_stage2.ddpm_model.in_channels)
    # zs_2 = training_loop_stage2.get_noise(batch_size=N_2,shape=z_shape[1:],seed=42)
    samples_2 = training_loop_stage2.get_samples(zs=zs_2,c=c_2, uc=uc_2, N=N_2, use_attn=use_attn, alpha=alpha,stage=2,
                                                 zs_src=zs_2_src,cond_src=c_2_src_,uc_src=uc_2_src_,
                                                 zs_tgt=zs_2_tgt,cond_tgt=c_2_tgt_,uc_tgt=uc_2_tgt_) # 最后一个为num_samples
    training_loop_stage2.save_results(idx=idx,
                                        batch_c = batch_c_1,
                                        samples = samples_2,
                                        save_img = args.save_img,
                                        camera = camera,
                                        latent_key='latent')
    
    dist_util.synchronize()
    logger.log('sampling complete')


def create_argparser(**kwargs):
    # defaults.update(model_and_diffusion_defaults())

    defaults = dict(
        dataset_size=-1,
        diffusion_input_size=-1,
        trainer_name='adm',
        use_amp=False,
        train_vae=True,  # jldm?
        triplane_scaling_divider=1.0,  # divide by this value
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
        eval_batch_size=12,
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
        load_depth=True,  # TODO
        logdir="/mnt/lustre/yslan/logs/nips23/",
        load_submodule_name='',  # for loading pretrained auto_encoder model
        ignore_resume_opt=False,
        # freeze_ae=False,
        denoised_ae=True,
        diffusion_ce_anneal=False,
        use_lmdb=False,
        interval=1,
        freeze_triplane_decoder=False,
        objv_dataset=False,
        use_eos_feature=False,
        clip_grad_throld=1.0,
        allow_tf32=True,
        save_img=False,
        use_train_trajectory=
        False,  # use train trajectory to sample images for fid calculation
        unconditional_guidance_scale=1.0,
        num_samples=10,
        num_instances=10, # for i23d, loop different condition
    )

    defaults.update(model_and_diffusion_defaults())
    defaults.update(continuous_diffusion_defaults())
    defaults.update(encoder_and_nsr_defaults())  # type: ignore
    defaults.update(dataset_defaults())  # type: ignore
    defaults.update(loss_defaults())
    defaults.update(control_net_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    # ! add transport args
    parse_transport_args(parser)

    return parser

import subprocess


def load_args_from_sh(file_path):
    command = f"source {file_path} && env"
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
    env_vars = {}

    for line in proc.stdout:
        key, _, value = line.decode('utf-8').partition('=')
        env_vars[key.strip()] = value.strip()

    proc.communicate()
    return env_vars


if __name__ == "__main__":
    # os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    # os.environ["NCCL_DEBUG"] = "INFO"
    th.multiprocessing.set_start_method('spawn')

    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.
 
    args = create_argparser().parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.gpus = th.cuda.device_count()
    args.rendering_kwargs = rendering_options_defaults(args)
    
    try:
        # morphing
        N = 25
        prompt_src = "a purple teapot."
        prompt_tgt = "A small green teapot with a spherical design."
        
       
        alpha_list = generate_beta_tensor(N)
        rearranged_alpha_list = torch.cat((alpha_list[:1], alpha_list[-1:], alpha_list[1:-1]))
        alpha_list = list(rearranged_alpha_list)
        
        
        idx_list = torch.arange(N, dtype=torch.int32)
        rearranged_idx_list = torch.cat((idx_list[:1], idx_list[-1:], idx_list[1:-1]))
        idx_list = list(rearranged_idx_list)
        
        
        stage1_ckpt_path_src = '/mnt/slurm_home/slyang/projects/gaussian-anything/train-finetune-stage1-src/model_joint_denoise_rec_model0000501.pt'
        stage1_ckpt_path_tgt = '/mnt/slurm_home/slyang/projects/gaussian-anything/train-finetune-stage1-tgt/model_joint_denoise_rec_model0000501.pt'
        stage2_ckpt_path_src = '/mnt/slurm_home/slyang/projects/gaussian-anything/train-finetune-stage2-src/model_joint_denoise_rec_model0000501.pt'
        stage2_ckpt_path_tgt = '/mnt/slurm_home/slyang/projects/gaussian-anything/train-finetune-stage2-tgt/model_joint_denoise_rec_model0000501.pt'
        
        latent_folder_src = '/mnt/slurm_home/slyang/projects/gaussian-anything/datasets/latent_codes_source'
        latent_folder_tgt = '/mnt/slurm_home/slyang/projects/gaussian-anything/datasets/latent_codes_target'
        
        zs_1_src, zs_2_src, c_1_src, uc_1_src, c_2_src, uc_2_src = get_inversed_zs(args, prompt_src, latent_folder_src, stage1_ckpt_path_src, stage2_ckpt_path_src)
        zs_1_tgt, zs_2_tgt, c_1_tgt, uc_1_tgt, c_2_tgt, uc_2_tgt = get_inversed_zs(args, prompt_tgt, latent_folder_tgt, stage1_ckpt_path_tgt, stage2_ckpt_path_tgt)

        from tqdm import tqdm
        
        for idx, alpha in enumerate(tqdm(alpha_list)):
            args_1_ckpt_path = interpolate_model_params(args.logdir, stage1_ckpt_path_src, stage1_ckpt_path_tgt, alpha, stage=1)
            args_2_ckpt_path = interpolate_model_params(args.logdir, stage2_ckpt_path_src, stage2_ckpt_path_tgt, alpha, stage=2)
            morphing_loop(args, prompt_src, prompt_tgt, args_1_ckpt_path, args_2_ckpt_path, idx_list[idx], zs_1_src, zs_2_src,zs_1_tgt, zs_2_tgt, alpha,
                        c_1_src, uc_1_src, c_2_src, uc_2_src,c_1_tgt, uc_1_tgt, c_2_tgt, uc_2_tgt,use_attn=True)
            
    
    except Exception as e:
        print(e)
        traceback.print_exc()
        dist_util.cleanup()  
