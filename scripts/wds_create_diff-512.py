"""
Train a diffusion model on images.
"""
# import imageio
import torch as th
from xformers.triton import FusedLayerNorm as LayerNorm
import lz4.frame
import gzip
import random
import json
import sys
import os
import lmdb
from tqdm import tqdm
sys.path.append('.')
import torch.distributed as dist
import pickle
import traceback
from PIL import Image
import torch.multiprocessing as mp
import lzma
import webdataset as wds
import numpy as np

from torch.utils.data import DataLoader, Dataset
import imageio.v3 as iio

import argparse
import dnnlib
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    args_to_dict,
    add_dict_to_argparser,
)
# from nsr.train_util import TrainLoop3DRec as TrainLoop
# from nsr.train_nv_util import TrainLoop3DRecNV, TrainLoop3DRec, TrainLoop3DRecNVPatch
from nsr.script_util import create_3DAE_model, encoder_and_nsr_defaults, loss_defaults, rendering_options_defaults, eg3d_options_default
from datasets.shapenet import load_data, load_data_for_lmdb, load_eval_data, load_memory_data
# from nsr.losses.builder import E3DGELossClass
from datasets.eg3d_dataset import init_dataset_kwargs

# from .lmdb_create import encode_and_compress_image

def encode_and_compress_image(inp_array, is_image=False, compress=True):
    # Read the image using imageio
    # image = imageio.v3.imread(image_path)

    # Convert the image to bytes
    # with io.BytesIO() as byte_buffer:
    #     imageio.imsave(byte_buffer, image, format="png")
    #     image_bytes = byte_buffer.getvalue()
    if is_image:
        inp_bytes = iio.imwrite("<bytes>", inp_array, extension=".png")
    else:
        inp_bytes = inp_array.tobytes()

    # Compress the image data using gzip
    if compress:
        # compressed_data = gzip.compress(inp_bytes)
        compressed_data = lz4.frame.compress(inp_bytes)
        return compressed_data
    else:
        return inp_bytes




from pdb import set_trace as st
import bz2

# th.backends.cuda.matmul.allow_tf32 = True # https://huggingface.co/docs/diffusers/optimization/fp16



def training_loop(args):
    # def training_loop(args):
    # dist_util.setup_dist(args)
    # th.autograd.set_detect_anomaly(True) # type: ignore
    th.autograd.set_detect_anomaly(False)  # type: ignore
    # https://blog.csdn.net/qq_41682740/article/details/126304613

    SEED = args.seed

    # dist.init_process_group(backend='nccl', init_method='env://', rank=args.local_rank, world_size=th.cuda.device_count())
    # logger.log(f"{args.local_rank=} init complete, seed={SEED}")
    # th.cuda.set_device(args.local_rank)
    th.cuda.empty_cache()

    # * deterministic algorithms flags
    th.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # logger.configure(dir=args.logdir, format_strs=["tensorboard", "csv"])
    logger.configure(dir=args.logdir)

    logger.log("creating encoder and NSR decoder...")
    # device = dist_util.dev()
    # device = th.device("cuda", args.local_rank)

    # shared eg3d opts
    opts = eg3d_options_default()

    if args.sr_training:
        args.sr_kwargs = dnnlib.EasyDict(
            channel_base=opts.cbase,
            channel_max=opts.cmax,
            fused_modconv_default='inference_only',
            use_noise=True
        )  # ! close noise injection? since noise_mode='none' in eg3d


    if args.objv_dataset:
        from datasets.g_buffer_objaverse import load_data, load_eval_data, load_memory_data, load_data_for_lmdb
    else: # shapenet
        from datasets.shapenet import load_data, load_eval_data, load_memory_data, load_data_for_lmdb

    # auto_encoder = create_3DAE_model(
    #     **args_to_dict(args,
    #                    encoder_and_nsr_defaults().keys()))
    # auto_encoder.to(device)
    # auto_encoder.train()

    logger.log("creating data loader...")
    # data = load_data(
    # st()
    # if args.overfitting:
    #     data = load_memory_data(
    #         file_path=args.data_dir,
    #         batch_size=args.batch_size,
    #         reso=args.image_size,
    #         reso_encoder=args.image_size_encoder,  # 224 -> 128
    #         num_workers=args.num_workers,
    #         # load_depth=args.depth_lambda > 0
    #         load_depth=True  # for evaluation
    #     )
    data, dataset_name, dataset_size = load_data_for_lmdb(
        file_path=args.data_dir,
        batch_size=args.batch_size,
        reso=args.image_size,
        reso_encoder=args.image_size_encoder,  # 224 -> 128
        num_workers=args.num_workers,
        load_depth=True,
        preprocess=None,
        dataset_size=args.dataset_size,
        trainer_name=args.trainer_name,
        shuffle_across_cls=args.shuffle_across_cls,
        wds_split=args.wds_split
        # wds_output_path=os.path.join(logger.get_dir(), f'wds-%06d.tar')
        # load_depth=True # for evaluation
    )

    args.img_size = [args.image_size_encoder]
    # let all processes sync up before starting with a new epoch of training
    dist_util.synchronize()

    opt = dnnlib.EasyDict(args_to_dict(args, loss_defaults().keys()))

    logger.log("training...")

    def convert_to_wds_compressed(dataset_loader, dataset_size, lmdb_path, start_shard, wds_split):
        """
        Convert a PyTorch dataset to LMDB format.

        Parameters:
        - dataset: PyTorch dataset
        - lmdb_path: Path to store the LMDB database
        """
        # env = lmdb.open(lmdb_path, map_size=1024 ** 4, readahead=False)  # Adjust map_size based on your dataset size
        sink = wds.ShardWriter(lmdb_path, start_shard=start_shard)

        # with env.begin(write=True) as txn:

        # with env.begin(write=True) as txn:
            # txn.put("length".encode("utf-8"), str(dataset_size).encode("utf-8"))

        for idx, sample in enumerate(tqdm(dataset_loader)):
            # pass
            # remove the batch index of returned dict sample

            # sample_ins = sample.pop('ins')
            # sample.pop('depth')
            # sample.pop('bbox') # only encode raw_img, caption, ins and c

            # assert [sample_ins[i]==sample_ins[0] for i in range(0,len(sample_ins))], sample_ins # check the batch is the same instnace

            sample = {
                # k:v.squeeze(0).cpu().numpy() if isinstance(v, th.Tensor) else v[0] for k, v in sample.items()
                k:v.cpu().numpy() if isinstance(v, th.Tensor) else v for k, v in sample.items()
                # k:v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in sample.items()
            }

            # encode batch images/depths/strings?  no need to encode ins/fname here; just save the caption

            # sample = dataset_loader[idx]
            # compressed_sample = {}
            # sample['ins'] = sample[0]

            # sample['caption'] = sample.pop('caption')[0]
            sample['raw_img'] = sample['raw_img'].squeeze(0) # 512 512 3


            # for k, v in sample.items():

                # key = f'{idx}-{k}'.encode('utf-8')

                # if 'img' in k: # only bytes required? laod the 512 depth bytes only.
                #     v = encode_and_compress_image(v, is_image=True, compress=True)
                #     # v = encode_and_compress_image(v, is_image=True, compress=False)
                # # elif 'depth' in k:
                # elif isinstance(v, str):
                #     v = v.encode('utf-8') # caption / instance name
                # else: # ! C
                #     v = encode_and_compress_image(v.astype(np.float32), is_image=False, compress=True)
                #     # v = encode_and_compress_image(v.astype(np.float32), is_image=False, compress=False)
                
                # compressed_sample[k] = v

            sink.write({
                "__key__": f"sample_{wds_split:03d}_{idx:07d}",
                # 'raw_img.png': sample['raw_img'],
                # 'raw_img.png': encode_and_compress_image(sample['raw_img'], is_image=True, compress=True)
                'raw_img': encode_and_compress_image(sample['raw_img'], is_image=True, compress=True),
                'ins': sample['ins'][0].encode('utf-8')
            })

            # break
            # if idx > 25:
            #     break

        sink.close()


    # convert_to_lmdb(data, os.path.join(logger.get_dir(), dataset_name)) convert_to_lmdb_compressed(data, os.path.join(logger.get_dir(), dataset_name))
    # convert_to_lmdb_compressed(data, os.path.join(logger.get_dir()), dataset_size) 
    convert_to_wds_compressed(data, dataset_size, os.path.join(logger.get_dir(), f'wds-%06d.tar'), args.start_shard, args.wds_split) 



def create_argparser(**kwargs):
    # defaults.update(model_and_diffusion_defaults())

    defaults = dict(
        seed=0,
        dataset_size=-1,
        trainer_name='input_rec',
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
        # test warm up pose sampling training
        objv_dataset=False,
        pose_warm_up_iter=-1,
        start_shard=0,
        shuffle_across_cls=False,
        wds_split=1, # out of 4
    )

    defaults.update(encoder_and_nsr_defaults())  # type: ignore
    defaults.update(loss_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    # os.environ[
    # "TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.
    # os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    # os.environ["NCCL_DEBUG"]="INFO"

    args = create_argparser().parse_args()
    # args.local_rank = int(os.environ["LOCAL_RANK"])
    args.gpus = th.cuda.device_count()

    opts = args

    args.rendering_kwargs = rendering_options_defaults(opts)

    # print(args)
    with open(os.path.join(args.logdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Launch processes.
    print('Launching processes...')

    # try:
    training_loop(args)
    # except KeyboardInterrupt as e:
    # except Exception as e:
    #     # print(e)
    #     traceback.print_exc()
    #     dist_util.cleanup() # clean port and socket when ctrl+c
