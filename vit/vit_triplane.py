import math
from pathlib import Path
# from pytorch3d.ops import create_sphere
import torchvision
import point_cloud_utils as pcu
from tqdm import trange
import random
import einops
from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from functools import partial
from torch.profiler import profile, record_function, ProfilerActivity

from nsr.networks_stylegan2 import Generator as StyleGAN2Backbone
from nsr.volumetric_rendering.renderer import ImportanceRenderer, ImportanceRendererfg_bg
from nsr.volumetric_rendering.ray_sampler import RaySampler
from nsr.triplane import OSGDecoder, Triplane, Triplane_fg_bg_plane
# from nsr.losses.helpers import ResidualBlock
from utils.dust3r.heads.dpt_head import create_dpt_head_ln3diff
from utils.nerf_utils import get_embedder
from vit.vision_transformer import TriplaneFusionBlockv4_nested, TriplaneFusionBlockv4_nested_init_from_dino_lite, TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout, VisionTransformer, TriplaneFusionBlockv4_nested_init_from_dino

from .vision_transformer import Block, VisionTransformer
from .utils import trunc_normal_

from guided_diffusion import dist_util, logger

from pdb import set_trace as st

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from torch_utils.components import PixelShuffleUpsample, ResidualBlock, Upsample, PixelUnshuffleUpsample, Conv3x3TriplaneTransformation
from torch_utils.distributions.distributions import DiagonalGaussianDistribution
from nsr.superresolution import SuperresolutionHybrid2X, SuperresolutionHybrid4X

from torch.nn.parameter import Parameter, UninitializedParameter, UninitializedBuffer

from nsr.common_blks import ResMlp
from timm.models.vision_transformer import PatchEmbed, Mlp
from .vision_transformer import *

from dit.dit_models import get_2d_sincos_pos_embed
from dit.dit_decoder import DiTBlock2
from torch import _assert
from itertools import repeat
import collections.abc

from nsr.srt.layers import Transformer as SRT_TX
from nsr.srt.layers import PreNorm

# from diffusers.models.upsampling import Upsample2D

from torch_utils.components import NearestConvSR
from timm.models.vision_transformer import PatchEmbed

from utils.general_utils import matrix_to_quaternion, quaternion_raw_multiply, build_rotation

from nsr.gs import GaussianRenderer

from utils.dust3r.heads import create_dpt_head

from ldm.modules.attention import MemoryEfficientCrossAttention, CrossAttention

# from nsr.geometry.camera.perspective_camera import PerspectiveCamera
# from nsr.geometry.render.neural_render import NeuralRender
# from nsr.geometry.rep_3d.flexicubes_geometry import FlexiCubesGeometry
# from utils.mesh_util import xatlas_uvmap


# From PyTorch internals
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)


def approx_gelu():
    return nn.GELU(approximate="tanh")


def init_gaussian_prediction(gaussian_pred_mlp):

    # https://github.com/szymanowiczs/splatter-image/blob/98b465731c3273bf8f42a747d1b6ce1a93faf3d6/configs/dataset/chairs.yaml#L15

    out_channels = [3, 1, 3, 4, 3]  # xyz, opacity, scale, rotation, rgb
    scale_inits = [  # ! avoid affecting final value (offset) 
        0,  #xyz_scale
        0.0,  #cfg.model.opacity_scale, 
        # 0.001,  #cfg.model.scale_scale,
        0,  #cfg.model.scale_scale,
        1,  # rotation
        0
    ]  # rgb

    bias_inits = [
        0.0,  # cfg.model.xyz_bias, no deformation here
        0,  # cfg.model.opacity_bias, sigmoid(0)=0.5 at init
        -2.5,  # scale_bias
        0.0,  # rotation
        0.5
    ]  # rgb

    start_channels = 0

    # for out_channel, b, s in zip(out_channels, bias, scale):
    for out_channel, b, s in zip(out_channels, bias_inits, scale_inits):
        # nn.init.xavier_uniform_(
        #     self.superresolution['conv_sr'].dpt.head[-1].weight[
        #         start_channels:start_channels + out_channel, ...], s)
        nn.init.constant_(
            gaussian_pred_mlp.weight[start_channels:start_channels +
                                     out_channel, ...], s)
        nn.init.constant_(
            gaussian_pred_mlp.bias[start_channels:start_channels +
                                   out_channel], b)
        start_channels += out_channel


class PatchEmbedTriplane(nn.Module):
    """ GroupConv patchembeder on triplane
    """

    def __init__(
        self,
        img_size=32,
        patch_size=2,
        in_chans=4,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
        plane_n=3,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.plane_n = plane_n
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans,
                              embed_dim * self.plane_n,
                              kernel_size=patch_size,
                              stride=patch_size,
                              bias=bias,
                              groups=self.plane_n)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # st()
        B, C, H, W = x.shape
        _assert(
            H == self.img_size[0],
            f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        )
        _assert(
            W == self.img_size[1],
            f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        )
        x = self.proj(x)  # B 3*C token_H token_W

        x = x.reshape(B, x.shape[1] // self.plane_n, self.plane_n, x.shape[-2],
                      x.shape[-1])  # B C 3 H W

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BC3HW -> B 3HW C
        x = self.norm(x)
        return x


class PatchEmbedTriplaneRodin(PatchEmbedTriplane):

    def __init__(self,
                 img_size=32,
                 patch_size=2,
                 in_chans=4,
                 embed_dim=768,
                 norm_layer=None,
                 flatten=True,
                 bias=True):
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer,
                         flatten, bias)
        self.proj = RodinRollOutConv3D_GroupConv(in_chans,
                                                 embed_dim * 3,
                                                 kernel_size=patch_size,
                                                 stride=patch_size,
                                                 padding=0)


# https://github.com/pytorch/pytorch/issues/8985
class ConditionalBatchNorm2d(nn.Module):

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(
            1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:,
                               num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(
            -1, self.num_features, 1, 1)
        return out


class ConditionalInstanceNorm2d(nn.Module):

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.IN = nn.InstanceNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(
            1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:,
                               num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.IN(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(
            -1, self.num_features, 1, 1)
        return out


# class InstanceNorm2dDecopmosed(nn.Module):
class InstanceNorm2dDecopmosed(nn.Module):

    def __init__(self,
                 num_features,
                 affine=True,
                 track_running_stats=True,
                 device=None,
                 dtype=None):
        # factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        assert affine, 'just handle affine separately'

        self.affine = affine
        self.num_features = num_features
        self.norm_layer = nn.InstanceNorm2d(
            num_features,
            affine=self.affine,
            track_running_stats=track_running_stats)

    def norm_wo_affine(self, input):
        # return self.IN(x)
        return torch.nn.functional.instance_norm(
            input,
            self.norm_layer.running_mean,
            self.norm_layer.running_var,
            # self.IN.weight,
            None,
            # self.IN.bias,
            None,
            self.norm_layer.training
            or not self.norm_layer.track_running_stats,
            self.norm_layer.momentum,
            self.norm_layer.eps)

    def norm_affine(self, out):  # do affine transformation manually
        assert self.affine
        logger.log(out.shape, self.norm_layer.weight.shape,
                   self.norm_layer.bias.shape)

        out = out * self.norm_layer.weight.unsqueeze(0)  # 1, num_features
        out = out + self.norm_layer.bias.unsqueeze(0)  # 1, num_features

        return out

    def forward(self, x):
        out = self.norm_wo_affine(x)
        out = self.norm_affine(out)
        return out


class GroupNorm2dDecopmosed(nn.Module):

    def __init__(self,
                 num_features,
                 affine=True,
                 device=None,
                 dtype=None,
                 group_divider=2):
        # factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        # assert affine, 'just handle affine separately'

        self.affine = affine
        self.num_features = num_features
        self.norm_layer = nn.GroupNorm(num_features // group_divider,
                                       num_features,
                                       affine=self.affine,
                                       device=device)

    def norm_wo_affine(self, input):
        # return self.IN(x)
        return torch.nn.functional.group_norm(input,
                                              self.norm_layer.num_groups, None,
                                              None, self.norm_layer.eps)

    def norm_affine(self, out):  # do affine transformation manually
        if not self.affine:
            return out
        # logger.log(out.shape, self.norm_layer.weight.unsqueeze(0).shape)

        out = out * self.norm_layer.weight.unsqueeze(0).view(
            1, -1, 1, 1)  # 1, num_features
        out = out + self.norm_layer.bias.unsqueeze(0).view(
            1, -1, 1, 1)  # 1, num_features

        return out

    def forward(self, x):
        out = self.norm_wo_affine(x)
        if self.affine:
            out = self.norm_affine(out)
        return out


class ViTTriplaneDecomposed(nn.Module):

    def __init__(
        self,
        vit_decoder,
        triplane_decoder: Triplane,
        cls_token=False,
        decoder_pred_size=-1,
        unpatchify_out_chans=-1,
        sr_ratio=2,
    ) -> None:
        super().__init__()
        self.superresolution = None

        self.decomposed_IN = False

        self.decoder_pred_3d = None
        self.transformer_3D_blk = None
        self.logvar = None

        self.cls_token = cls_token
        self.vit_decoder = vit_decoder
        self.triplane_decoder = triplane_decoder
        # triplane_sr_ratio = self.triplane_decoder.triplane_size / self.vit_decoder.img_size
        # self.decoder_pred = nn.Linear(self.vit_decoder.embed_dim,
        #                               self.vit_decoder.patch_size**2 *
        #                               self.triplane_decoder.out_chans,
        #                               bias=True)  # decoder to pat

        # self.patch_size = self.vit_decoder.patch_embed.patch_size
        self.patch_size = 14  # TODO, hard coded here
        if isinstance(self.patch_size, tuple):  # dino-v2
            self.patch_size = self.patch_size[0]

        # self.img_size = self.vit_decoder.patch_embed.img_size
        self.img_size = None  # TODO, hard coded
        if decoder_pred_size == -1:
            decoder_pred_size = self.patch_size**2 * self.triplane_decoder.out_chans

        if unpatchify_out_chans == -1:
            self.unpatchify_out_chans = self.triplane_decoder.out_chans
        else:
            self.unpatchify_out_chans = unpatchify_out_chans

        self.decoder_pred = nn.Linear(
            self.vit_decoder.embed_dim,
            decoder_pred_size,
            #   self.patch_size**2 *
            #   self.triplane_decoder.out_chans,
            bias=True)  # decoder to pat
        # st()

    def triplane_decode(self, latent, c):
        ret_dict = self.triplane_decoder(latent, c)  # triplane latent -> imgs
        ret_dict.update({'latent': latent})
        return ret_dict

    def triplane_renderer(self, latent, coordinates, directions):

        planes = latent.view(len(latent), 3,
                             self.triplane_decoder.decoder_in_chans,
                             latent.shape[-2],
                             latent.shape[-1])  # BS 96 256 256

        ret_dict = self.triplane_decoder.renderer.run_model(
            planes, self.triplane_decoder.decoder, coordinates, directions,
            self.triplane_decoder.rendering_kwargs)  # triplane latent -> imgs
        # ret_dict.update({'latent': latent})
        return ret_dict

        # * increase encoded encoded latent dim to match decoder

    def forward_vit_decoder(self, x, img_size=None):
        # latent: (N, L, C) from DINO/CLIP ViT encoder

        # * also dino ViT
        # add positional encoding to each token
        if img_size is None:
            img_size = self.img_size

        if self.cls_token:
            x = x + self.vit_decoder.interpolate_pos_encoding(
                x, img_size, img_size)[:, :]  # B, L, C
        else:
            x = x + self.vit_decoder.interpolate_pos_encoding(
                x, img_size, img_size)[:, 1:]  # B, L, C

        for blk in self.vit_decoder.blocks:
            x = blk(x)
        x = self.vit_decoder.norm(x)

        return x

    def unpatchify(self, x, p=None, unpatchify_out_chans=None):
        """
        x: (N, L, patch_size**2 * self.out_chans)
        imgs: (N, self.out_chans, H, W)
        """
        # st()
        if unpatchify_out_chans is None:
            unpatchify_out_chans = self.unpatchify_out_chans
        # p = self.vit_decoder.patch_size
        if self.cls_token:  # TODO, how to better use cls token
            x = x[:, 1:]

        if p is None:  # assign upsample patch size
            p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, unpatchify_out_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], unpatchify_out_chans, h * p,
                                h * p))
        return imgs

    def forward(self, latent, c, img_size):
        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        if self.cls_token:
            # latent, cls_token = latent[:, 1:], latent[:, :1]
            cls_token = latent[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        # st()
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # TODO 2D convolutions -> Triplane
        # * triplane rendering
        # ret_dict = self.forward_triplane_decoder(latent,
        #                                          c)  # triplane latent -> imgs
        ret_dict = self.triplane_decoder(planes=latent, c=c)
        ret_dict.update({'latent': latent, 'cls_token': cls_token})

        return ret_dict


class ViTTriplaneDecomposed_SR_v1(ViTTriplaneDecomposed):
    """v1, add basic convolution layer before triplane decoder
    """

    def __init__(
        self,
        vit_decoder: VisionTransformer,
        triplane_decoder: Triplane,
        cls_token,
        normalize_feat=True,
    ) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token)
        self.superresolution = nn.Conv2d(triplane_decoder.out_chans,
                                         triplane_decoder.out_chans,
                                         3,
                                         padding=1)

        self.normalize_feat = normalize_feat
        # if normalize_feat:
        # self.BN_norm = nn.BatchNorm2d(vit_decoder.embed_dim, affine=False) # simply do the normalization

    def forward(self, latent, c, img_size):
        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * smooth convolution before triplane
        latent = self.superresolution(latent)

        if self.normalize_feat:
            latent = latent.clamp(-50, 50)

        # * triplane rendering
        ret_dict = self.triplane_decoder(latent, c)  # triplane latent -> imgs
        ret_dict.update({'latent': latent})

        return ret_dict


class ViTTriplaneDecomposed_SR_v1_dualSRconv(ViTTriplaneDecomposed):
    """v1, add basic convolution layer before triplane decoder
    """

    def __init__(
        self,
        vit_decoder: VisionTransformer,
        triplane_decoder: Triplane,
        cls_token,
        normalize_feat=True,
    ) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token)
        self.superresolution = nn.ModuleDict({
            'conv1':
            nn.Conv2d(triplane_decoder.out_chans,
                      triplane_decoder.out_chans,
                      3,
                      padding=1),
            'conv2':
            nn.Conv2d(triplane_decoder.out_chans,
                      triplane_decoder.out_chans,
                      3,
                      padding=1)
        })

        self.normalize_feat = normalize_feat

    def triplane_decode(self, latent, c):

        latent = self.superresolution['conv1'](latent)

        latent = self.superresolution['conv2'](latent)

        if self.normalize_feat:
            latent = latent.clamp(-50, 50)

        # * triplane rendering
        ret_dict = self.triplane_decoder(latent, c)  # triplane latent -> imgs
        ret_dict.update({'latent': latent})

        return ret_dict

    def forward(self, latent, c, img_size):
        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * smooth convolution before triplane

        return self.triplane_decode(latent, c)


# * =========== STUDY NORMALIZTION ========


class ViTTriplaneDecomposed_SR_v1_IN(ViTTriplaneDecomposed):
    """v1, add basic convolution layer before triplane decoder
    """

    def __init__(
        self,
        vit_decoder: VisionTransformer,
        triplane_decoder: Triplane,
        cls_token,
        normalize_feat=True,
    ) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token)
        self.superresolution = nn.Conv2d(triplane_decoder.out_chans,
                                         triplane_decoder.out_chans,
                                         3,
                                         padding=1)

        if normalize_feat:
            self.IN = nn.InstanceNorm2d(
                triplane_decoder.out_chans,
                affine=False)  # simply do the normalization
        self.normalize_feat = normalize_feat

    def forward(self, latent, c, img_size):
        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * smooth convolution before triplane
        latent = self.superresolution(latent)

        if self.normalize_feat:
            # latent = (latent - 10) / (3 * 50)
            latent = self.IN(latent)  # normalize to mean 0, std 1
            latent = latent.clamp(-50, 50)
            # latent = latent.clamp(-1, 1)
            # latent = (latent / 3. ).clamp(-1, 1)  # contain the 3-sigma range, clip to [-1,1]

        # * triplane rendering
        ret_dict = self.triplane_decoder(latent, c)  # triplane latent -> imgs
        ret_dict.update({'latent': latent})

        return ret_dict


class ViTTriplaneDecomposed_SR_v1_IN_twoconv(ViTTriplaneDecomposed):
    """v1, add basic convolution layer before triplane decoder
    """

    def __init__(
        self,
        vit_decoder: VisionTransformer,
        triplane_decoder: Triplane,
        cls_token,
        normalize_feat=True,
    ) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token)
        self.superresolution = nn.ModuleDict({
            'conv1':
            nn.Conv2d(triplane_decoder.out_chans,
                      triplane_decoder.out_chans,
                      3,
                      padding=1),
            'conv2':
            nn.Conv2d(triplane_decoder.out_chans,
                      triplane_decoder.out_chans,
                      3,
                      padding=1)
        })

        if normalize_feat:
            self.IN = nn.InstanceNorm2d(
                triplane_decoder.out_chans,
                affine=False)  # simply do the normalization
        self.normalize_feat = normalize_feat

    def triplane_decode(self, latent_normalized, c):

        latent_denormalized = self.superresolution['conv2'](latent_normalized)

        # * triplane rendering
        ret_dict = self.triplane_decoder(latent_denormalized,
                                         c)  # triplane latent -> imgs
        ret_dict.update({
            'latent_normalized': latent_normalized,
            'latent_denormalized': latent_denormalized
        })

        return ret_dict

    def forward(self, latent, c, img_size):
        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * smooth convolution before triplane
        latent = self.superresolution['conv1'](latent)

        if self.normalize_feat:
            # latent = (latent - 10) / (3 * 50)
            latent_normalized = self.IN(latent).clamp(
                -50, 50)  # normalize to mean 0, std 1
        else:
            latent_normalized = latent.clamp(-50, 50)
            # latent = latent.clamp(-1, 1)
            # latent = (latent / 3. ).clamp(-1, 1)  # contain the 3-sigma range, clip to [-1,1]

        return self.triplane_decode(latent_normalized, c)


class ViTTriplaneDecomposed_SR_v1_IN_twoconv_affineTrue(ViTTriplaneDecomposed):
    """v1, add basic convolution layer before triplane decoder
    """

    def __init__(
        self,
        vit_decoder: VisionTransformer,
        triplane_decoder: Triplane,
        cls_token,
        normalize_feat=True,
    ) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token)
        self.superresolution = nn.ModuleDict({
            'conv1':
            nn.Conv2d(triplane_decoder.out_chans,
                      triplane_decoder.out_chans,
                      3,
                      padding=1),
            'conv2':
            nn.Conv2d(triplane_decoder.out_chans,
                      triplane_decoder.out_chans,
                      3,
                      padding=1)
        })

        if normalize_feat:
            self.IN = nn.InstanceNorm2d(
                triplane_decoder.out_chans,
                affine=True,
                track_running_stats=False)  # simply do the normalization
        self.normalize_feat = normalize_feat

    def triplane_decode(self, latent_normalized, c):

        latent_denormalized = self.superresolution['conv2'](latent_normalized)

        # * triplane rendering
        ret_dict = self.triplane_decoder(latent_denormalized,
                                         c)  # triplane latent -> imgs
        ret_dict.update({
            'latent_normalized': latent_normalized,
            'latent_denormalized': latent_denormalized
        })

        return ret_dict

    def forward(self, latent, c, img_size):
        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * smooth convolution before triplane
        latent = self.superresolution['conv1'](latent)

        if self.normalize_feat:
            # latent = (latent - 10) / (3 * 50)
            latent_normalized = self.IN(latent).clamp(
                -50, 50)  # normalize to mean 0, std 1
        else:
            latent_normalized = latent.clamp(-50, 50)
            # latent = latent.clamp(-1, 1)
            # latent = (latent / 3. ).clamp(-1, 1)  # contain the 3-sigma range, clip to [-1,1]

        return self.triplane_decode(latent_normalized, c)


class ViTTriplaneDecomposed_SR_v1_IN_twoconv_affine_true_track_mean(
        ViTTriplaneDecomposed):
    """v1, add basic convolution layer before triplane decoder
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 *args,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token, **kwargs
                         #  decoder_pred_size=
                         )

        # ! add all parameters inside the superresolution to add in the optimizer
        self.superresolution = nn.ModuleDict({
            'conv1':
            nn.Conv2d(triplane_decoder.out_chans,
                      triplane_decoder.out_chans,
                      3,
                      padding=1),
            'conv2':
            nn.Conv2d(triplane_decoder.out_chans,
                      triplane_decoder.out_chans,
                      3,
                      padding=1),
            # 'norm':
            # nn.InstanceNorm2d(
            #     triplane_decoder.out_chans,
            #     affine=True,
            #     track_running_stats=True)  # simply do the normalization
        })

        if normalize_feat:
            self.IN = nn.InstanceNorm2d(
                triplane_decoder.out_chans,
                affine=True,
                track_running_stats=True
            )  # ! simply do the normalization, not learnable, to remove
            # self.IN = GroupNorm2dDecopmosed(triplane_decoder.out_chans)

        self.normalize_feat = normalize_feat

    def vit_decode(self, latent, img_size):

        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * smooth convolution before triplane
        latent = self.superresolution['conv1'](latent)

        if self.normalize_feat:
            # latent = (latent - 10) / (3 * 50)
            latent_normalized = self.IN(latent)  # .clamp(
            # latent_normalized = self.IN(latent).clamp(
            #     # latent_normalized = self.superresolution['norm'](latent).clamp(
            #     -50,
            #     50)  # normalize to mean 0, std 1
        else:
            # ? clamp latent determines convergence or not?
            # latent_normalized = latent.clamp(-50, 50)
            latent_normalized = latent

        return latent_normalized

    def triplane_decode(self, latent_normalized, c):

        latent_denormalized = self.superresolution['conv2'](latent_normalized)

        # * triplane rendering
        ret_dict = self.triplane_decoder(latent_denormalized,
                                         c)  # triplane latent -> imgs
        ret_dict.update({
            'latent_normalized': latent_normalized,
            'latent_denormalized': latent_denormalized
        })

        return ret_dict

    def forward(self, latent, c, img_size):

        latent_normalized = self.vit_decode(latent, img_size)
        return self.triplane_decode(latent_normalized, c)


class ViTTriplaneDecomposed_SR_v1_LN_twoconv_affine_true_track_mean(
        ViTTriplaneDecomposed):
    """v1, add basic convolution layer before triplane decoder
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 *args,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            #  decoder_pred_size=
        )

        # ! add all parameters inside the superresolution to add in the optimizer
        self.superresolution = nn.ModuleDict({
            'conv1':
            nn.Conv2d(triplane_decoder.out_chans,
                      triplane_decoder.out_chans,
                      3,
                      padding=1),
            'conv2':
            nn.Conv2d(triplane_decoder.out_chans,
                      triplane_decoder.out_chans,
                      3,
                      padding=1),
            # 'norm':
            # nn.InstanceNorm2d(
            #     triplane_decoder.out_chans,
            #     affine=True,
            #     track_running_stats=True)  # simply do the normalization
        })

        if normalize_feat:
            self.LN = nn.LayerNorm(
                [triplane_decoder.out_chans, 224, 224],
                elementwise_affine=True,
            )  # ! simply do the normalization, not learnable, to remove
            # self.IN = GroupNorm2dDecopmosed(triplane_decoder.out_chans)

        self.normalize_feat = normalize_feat

    def vit_decode(self, latent, img_size):

        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * smooth convolution before triplane
        latent = self.superresolution['conv1'](latent)

        if self.normalize_feat:
            # latent = (latent - 10) / (3 * 50)
            latent_normalized = self.LN(latent)  # .clamp(
            # latent_normalized = self.IN(latent).clamp(
            #     # latent_normalized = self.superresolution['norm'](latent).clamp(
            #     -50,
            #     50)  # normalize to mean 0, std 1
        else:
            # ? clamp latent determines convergence or not?
            # latent_normalized = latent.clamp(-50, 50)
            latent_normalized = latent

        return latent_normalized

    def triplane_decode(self, latent_normalized, c):

        latent_denormalized = self.superresolution['conv2'](latent_normalized)

        # * triplane rendering
        ret_dict = self.triplane_decoder(latent_denormalized,
                                         c)  # triplane latent -> imgs
        ret_dict.update({
            'latent_normalized': latent_normalized,
            'latent_denormalized': latent_denormalized
        })

        return ret_dict

    def forward(self, latent, c, img_size):

        latent_normalized = self.vit_decode(latent, img_size)
        return self.triplane_decode(latent_normalized, c)


class ViTTriplaneDecomposed_SR_v1_BN_twoconv_affine_true_track_mean(
        ViTTriplaneDecomposed):
    """v1, add basic convolution layer before triplane decoder
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 *args,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            #  decoder_pred_size=
        )

        # ! add all parameters inside the superresolution to add in the optimizer
        self.superresolution = nn.ModuleDict({
            'conv1':
            nn.Conv2d(triplane_decoder.out_chans,
                      triplane_decoder.out_chans,
                      3,
                      padding=1),
            'conv2':
            nn.Conv2d(triplane_decoder.out_chans,
                      triplane_decoder.out_chans,
                      3,
                      padding=1),
            # 'norm':
            # nn.InstanceNorm2d(
            #     triplane_decoder.out_chans,
            #     affine=True,
            #     track_running_stats=True)  # simply do the normalization
        })

        if normalize_feat:
            self.BN = nn.BatchNorm2d(
                triplane_decoder.out_chans,
                affine=True,
            )  # ! simply do the normalization, not learnable, to remove
            # self.IN = GroupNorm2dDecopmosed(triplane_decoder.out_chans)

        self.normalize_feat = normalize_feat

    def vit_decode(self, latent, img_size):

        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * smooth convolution before triplane
        latent = self.superresolution['conv1'](latent)

        if self.normalize_feat:
            # latent = (latent - 10) / (3 * 50)
            latent_normalized = self.BN(latent)  # .clamp(
            # latent_normalized = self.IN(latent).clamp(
            #     # latent_normalized = self.superresolution['norm'](latent).clamp(
            #     -50,
            #     50)  # normalize to mean 0, std 1
        else:
            # ? clamp latent determines convergence or not?
            # latent_normalized = latent.clamp(-50, 50)
            latent_normalized = latent

        return latent_normalized

    def triplane_decode(self, latent_normalized, c):

        latent_denormalized = self.superresolution['conv2'](latent_normalized)

        # * triplane rendering
        ret_dict = self.triplane_decoder(latent_denormalized,
                                         c)  # triplane latent -> imgs
        ret_dict.update({
            'latent_normalized': latent_normalized,
            'latent_denormalized': latent_denormalized
        })

        return ret_dict

    def forward(self, latent, c, img_size):

        latent_normalized = self.vit_decode(latent, img_size)
        return self.triplane_decode(latent_normalized, c)


class ViTTriplaneDecomposed_SR_v1_GN_twoconv_affine_true_track_mean(
        ViTTriplaneDecomposed):
    """v1, add basic convolution layer before triplane decoder
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 *args,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            #  decoder_pred_size=
        )

        # ! add all parameters inside the superresolution to add in the optimizer
        self.superresolution = nn.ModuleDict({
            'conv1':
            nn.Conv2d(triplane_decoder.out_chans,
                      triplane_decoder.out_chans,
                      3,
                      padding=1),
            'conv2':
            nn.Conv2d(triplane_decoder.out_chans,
                      triplane_decoder.out_chans,
                      3,
                      padding=1),
            # 'norm':
            # nn.InstanceNorm2d(
            #     triplane_decoder.out_chans,
            #     affine=True,
            #     track_running_stats=True)  # simply do the normalization
        })

        if normalize_feat:
            self.GN = nn.GroupNorm(
                # triplane_decoder.out_chans // 4, triplane_decoder.out_chans,
                # triplane_decoder.out_chans // 6, triplane_decoder.out_chans,
                triplane_decoder.out_chans // 3,
                triplane_decoder.out_chans,
                # affine=True
            )  # ! simply do the normalization, not learnable, to remove
            # self.IN = GroupNorm2dDecopmosed(triplane_decoder.out_chans)

        self.normalize_feat = normalize_feat

    def vit_decode(self, latent, img_size):

        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * smooth convolution before triplane
        latent = self.superresolution['conv1'](latent)

        if self.normalize_feat:
            # latent = (latent - 10) / (3 * 50)
            latent_normalized = self.GN(latent)  # .clamp(
            # latent_normalized = self.IN(latent).clamp(
            #     # latent_normalized = self.superresolution['norm'](latent).clamp(
            #     -50,
            #     50)  # normalize to mean 0, std 1
        else:
            # ? clamp latent determines convergence or not?
            # latent_normalized = latent.clamp(-50, 50)
            latent_normalized = latent

        return latent_normalized

    def triplane_decode(self, latent_normalized, c):

        latent_denormalized = self.superresolution['conv2'](latent_normalized)

        # * triplane rendering
        ret_dict = self.triplane_decoder(latent_denormalized,
                                         c)  # triplane latent -> imgs
        ret_dict.update({
            'latent_normalized': latent_normalized,
            'latent_denormalized': latent_denormalized
        })

        return ret_dict

    def forward(self, latent, c, img_size):

        latent_normalized = self.vit_decode(latent, img_size)
        return self.triplane_decode(latent_normalized, c)


# class ViTTriplaneDecomposed_SR_v1_LN_twoconv_affine_true_track_mean(
#         ViTTriplaneDecomposed_SR_v1_IN_twoconv_affine_true_track_mean):

#     def __init__(self,
#                  vit_decoder: VisionTransformer,
#                  triplane_decoder: Triplane,
#                  cls_token,
#                  normalize_feat=True) -> None:
#         super().__init__(vit_decoder, triplane_decoder, cls_token,
#                          normalize_feat)

#         self.superresolution.update({
#             'norm':
#             nn.LayerNorm(
#                 [triplane_decoder.out_chans, ],
#                 # affine=True,
#                 # track_running_stats=True
#                 )  # simply do the normalization
#         })

#         self.IN = nn.Identity()


class ViTTriplaneDecomposed_SR_v1_GN_twoconv_affine(
        ViTTriplaneDecomposed_SR_v1_IN_twoconv_affine_true_track_mean):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, **kwargs)

        self.IN = nn.Identity()

        self.superresolution.update({
            'norm':
            GroupNorm2dDecopmosed(triplane_decoder.out_chans, group_divider=6)
        })
        # {'norm': GroupNorm2dDecopmosed(triplane_decoder.out_chans, group_divider=2)})
        self.register_buffer('w_avg',
                             torch.zeros([512]))  # will replace externally

    def vit_decode_backbone(self, latent, img_size):
        return self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

    def vit_decode_postprocess(
        self,
        latent_from_vit,
    ):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * smooth convolution before triplane
        latent = self.superresolution['conv1'](latent)

        if self.normalize_feat:
            latent_normalized = self.superresolution['norm'].norm_wo_affine(
                latent)  # type: ignore
        else:
            latent_normalized = latent

        return dict(latent_normalized=latent_normalized,
                    cls_token=cls_token,
                    latent_from_vit=latent_from_vit)

    def vit_decode(self, latent, img_size):

        latent_from_vit = self.vit_decode_backbone(latent, img_size)

        return self.vit_decode_postprocess(latent_from_vit)

    def triplane_decode(self, vit_decode_out, c, return_raw_only=False):

        if isinstance(vit_decode_out, dict):
            latent_normalized, sr_w_code = (vit_decode_out.get(k, None)
                                            for k in ('latent_normalized',
                                                      'sr_w_code'))

            latent_normalized = vit_decode_out['latent_normalized']

        else:
            latent_normalized = vit_decode_out
            sr_w_code = None
            vit_decode_out = dict(latent_normalized=latent_normalized
                                  )  # for later dict update compatability

        latent_denormalized = self.superresolution['norm'].norm_affine(
            latent_normalized)  # type: ignore
        latent_denormalized = self.superresolution['conv2'](
            latent_denormalized)

        # * triplane rendering

        ret_dict = self.triplane_decoder(
            latent_denormalized,
            c,
            ws=sr_w_code,
            return_raw_only=return_raw_only)  # triplane latent -> imgs
        ret_dict.update({
            # 'latent_normalized': latent_normalized,
            'latent_denormalized': latent_denormalized,
            **vit_decode_out
        })

        return ret_dict


class ViTTriplaneDecomposed_SR_v1_GN_twoconv_noaffine(  # used for reconstruction ablation
        ViTTriplaneDecomposed_SR_v1_IN_twoconv_affine_true_track_mean):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat)

        self.IN = nn.Identity()

        self.superresolution.update({
            'norm':
            GroupNorm2dDecopmosed(triplane_decoder.out_chans, affine=False)
        })

    def vit_decode_backbone(self, latent, img_size):
        return self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

    def vit_decode_postprocess(
        self,
        latent_from_vit,
    ):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * smooth convolution before triplane
        latent = self.superresolution['conv1'](latent)

        if self.normalize_feat:
            latent_normalized = self.superresolution['norm'].norm_wo_affine(
                latent)  # type: ignore
        else:
            latent_normalized = latent

        return dict(latent_normalized=latent_normalized,
                    cls_token=cls_token,
                    latent_from_vit=latent_from_vit)

    def vit_decode(self, latent, img_size):

        latent_from_vit = self.vit_decode_backbone(latent, img_size)

        return self.vit_decode_postprocess(latent_from_vit)

    def triplane_decode(self, vit_decode_out, c, return_raw_only=False):

        if isinstance(vit_decode_out, dict):
            latent_normalized, sr_w_code = (vit_decode_out.get(k, None)
                                            for k in ('latent_normalized',
                                                      'sr_w_code'))

            latent_normalized = vit_decode_out['latent_normalized']

        else:
            latent_normalized = vit_decode_out
            sr_w_code = None
            vit_decode_out = dict(latent_normalized=latent_normalized
                                  )  # for later dict update compatability

        latent_denormalized = self.superresolution['norm'].norm_affine(
            latent_normalized)  # type: ignore
        latent_denormalized = self.superresolution['conv2'](
            latent_denormalized)

        # * triplane rendering

        ret_dict = self.triplane_decoder(
            latent_denormalized,
            c,
            ws=sr_w_code,
            return_raw_only=return_raw_only)  # triplane latent -> imgs
        ret_dict.update({
            # 'latent_normalized': latent_normalized,
            'latent_denormalized': latent_denormalized,
            **vit_decode_out
        })

        return ret_dict


# * SR version of the above


class ViTTriplaneDecomposed_SR_v1_GN_twoconv_affineSR(
        ViTTriplaneDecomposed_SR_v1_GN_twoconv_affine):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 w_avg=torch.zeros([512])) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat)

        self.w_proj = Mlp(self.vit_decoder.num_features, out_features=512)
        assert self.cls_token, 'requires [cls] to project SR w code'

        self.register_buffer('w_avg', w_avg)  # will replace externally

    def vit_decode(self, latent, img_size):

        vit_decode_dict = super().vit_decode(latent, img_size)

        sr_w_code = self.w_proj(
            vit_decode_dict['cls_token']) + self.w_avg.reshape(
                1, 1, -1)  # type: ignore

        vit_decode_dict.update(dict(sr_w_code=sr_w_code))

        return vit_decode_dict


class ViTTriplaneDecomposed_SR_v1_GN_twoconv_noaffine_true_track_mean(
        ViTTriplaneDecomposed_SR_v1_GN_twoconv_affine):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat)

        self.IN = nn.Identity()

        self.superresolution.update({
            'norm':
            nn.GroupNorm(triplane_decoder.out_chans // 4,
                         triplane_decoder.out_chans,
                         affine=False)
        })


class ViTTriplaneDecomposed_SR_v1_IN_twoconv_affine_true_track_meanDecomposedIN(
        ViTTriplaneDecomposed_SR_v1_IN_twoconv_affine_true_track_mean):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat)

        self.decomposed_IN = True
        if normalize_feat:
            self.superresolution.update(
                dict(IN=InstanceNorm2dDecopmosed(triplane_decoder.out_chans,
                                                 affine=True,
                                                 track_running_stats=True)))

    def vit_decode(self, latent, img_size):

        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * smooth convolution before triplane
        latent = self.superresolution['conv1'](latent)

        # if self.normalize_feat:
        # latent = (latent - 10) / (3 * 50)
        latent_normalized = self.superresolution['norm'].norm_wo_affine(
            latent)  # type: ignore

        return latent_normalized  # unit normal distribution

    def triplane_decode(self, latent_normalized, c):

        latent_normalized = self.superresolution['norm'].norm_affine(
            latent_normalized)  # type: ignore
        latent_denormalized = self.superresolution['conv2'](latent_normalized)

        # * triplane rendering
        ret_dict = self.triplane_decoder(latent_denormalized,
                                         c)  # triplane latent -> imgs
        ret_dict.update({
            'latent_normalized': latent_normalized,
            'latent_denormalized': latent_denormalized
        })

        return ret_dict


class ViTTriplaneDecomposed_SR_v1_IN_twoconv_affine_true_track_mean_patchUpsampleHybrid(
        ViTTriplaneDecomposed):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2) -> None:

        patch_size = vit_decoder.patch_embed.patch_size

        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]

        super().__init__(vit_decoder,
                         triplane_decoder,
                         cls_token,
                         decoder_pred_size=(patch_size // 2)**2 *
                         (triplane_decoder.out_chans * 2),
                         unpatchify_out_chans=triplane_decoder.out_chans * 2)

        # vit_output_chans = triplane_decoder.out_chans  # 8

        # vit_output_upsampled_chans = 32  # upsample
        # self.vit_output_upsampled_chans = vit_output_upsampled_chans

        self.superresolution = nn.ModuleDict({
            'proj_upsample':
            PixelUnshuffleUpsample(
                output_dim=triplane_decoder.out_chans * 2,
                num_feat=int(triplane_decoder.out_chans * 1.5),
                num_out_ch=triplane_decoder.out_chans,
                sr_ratio=sr_ratio),  # 4x SR too heavy to converge.

            # nn.Conv2d(vit_output_chans,
            #           vit_output_upsampled_chans,
            #           kernel_size=patch_size,
            #           stride=patch_size),
            # 'conv1':
            # nn.Conv2d(vit_output_upsampled_chans,
            #           vit_output_upsampled_chans,
            #           3,
            #           padding=1),
            'conv2':
            nn.Conv2d(triplane_decoder.out_chans,
                      triplane_decoder.out_chans,
                      3,
                      padding=1)
        })

        if normalize_feat:
            self.IN = nn.InstanceNorm2d(
                triplane_decoder.out_chans,
                affine=True,
                track_running_stats=True)  # simply do the normalization
        self.normalize_feat = normalize_feat

    def vit_decode(self, latent, img_size):

        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size

        # logger.log('after decode_pred: ', latent.shape)

        # ! upsample dimention to 32

        latent = self.unpatchify(
            latent,
            p=7,
        )  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * smooth convolution before triplane
        # latent = self.superresolution['conv1'](latent) # 2D->3D projectio, no residual connection.

        latent = self.superresolution['proj_upsample'](
            latent,
            input_skip_connection=False)  # still needed the above conv?

        if self.normalize_feat:
            latent_normalized = self.IN(latent).clamp(
                -50, 50)  # normalize to mean 0, std 1
        else:
            latent_normalized = latent.clamp(-50, 50)

        return latent_normalized

    def triplane_decode(self, latent_normalized, c):

        latent_denormalized = self.superresolution['conv2'](latent_normalized)

        # * triplane rendering
        ret_dict = self.triplane_decoder(latent_denormalized,
                                         c)  # triplane latent -> imgs
        ret_dict.update({
            'latent_normalized': latent_normalized,
            'latent_denormalized': latent_denormalized
        })

        return ret_dict


class ViTTriplaneDecomposed_SR_v1_IN_twoconv_affine_true_track_mean_patchUpsampleHybrid3X(
        ViTTriplaneDecomposed):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True) -> None:

        patch_size = vit_decoder.patch_embed.patch_size

        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]

        super().__init__(vit_decoder,
                         triplane_decoder,
                         cls_token,
                         decoder_pred_size=(patch_size // 2)**2 *
                         (triplane_decoder.out_chans * 2),
                         unpatchify_out_chans=triplane_decoder.out_chans * 2)

        # vit_output_chans = triplane_decoder.out_chans  # 8

        # vit_output_upsampled_chans = 32  # upsample
        # self.vit_output_upsampled_chans = vit_output_upsampled_chans

        self.superresolution = nn.ModuleDict({
            'proj_upsample':
            PixelUnshuffleUpsample(output_dim=triplane_decoder.out_chans * 2,
                                   num_feat=int(triplane_decoder.out_chans *
                                                1.5),
                                   num_out_ch=triplane_decoder.out_chans,
                                   sr_ratio=3),  # 4x SR too heavy to converge.

            # nn.Conv2d(vit_output_chans,
            #           vit_output_upsampled_chans,
            #           kernel_size=patch_size,
            #           stride=patch_size),
            # 'conv1':
            # nn.Conv2d(vit_output_upsampled_chans,
            #           vit_output_upsampled_chans,
            #           3,
            #           padding=1),
            'conv2':
            nn.Conv2d(triplane_decoder.out_chans,
                      triplane_decoder.out_chans,
                      3,
                      padding=1)
        })

        if normalize_feat:
            self.IN = nn.InstanceNorm2d(
                triplane_decoder.out_chans,
                affine=True,
                track_running_stats=True)  # simply do the normalization
        self.normalize_feat = normalize_feat

    def vit_decode(self, latent, img_size):

        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size

        # logger.log('after decode_pred: ', latent.shape)

        # ! upsample dimention to 32

        latent = self.unpatchify(
            latent,
            p=7,
        )  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * smooth convolution before triplane
        # latent = self.superresolution['conv1'](latent) # 2D->3D projectio, no residual connection.

        latent = self.superresolution['proj_upsample'](
            latent,
            input_skip_connection=False)  # still needed the above conv?

        if self.normalize_feat:
            latent_normalized = self.IN(latent).clamp(
                -50, 50)  # normalize to mean 0, std 1
        else:
            latent_normalized = latent.clamp(-50, 50)

        return latent_normalized

    def triplane_decode(self, latent_normalized, c):

        latent_denormalized = self.superresolution['conv2'](latent_normalized)

        # * triplane rendering
        ret_dict = self.triplane_decoder(latent_denormalized,
                                         c)  # triplane latent -> imgs
        ret_dict.update({
            'latent_normalized': latent_normalized,
            'latent_denormalized': latent_denormalized
        })

        return ret_dict


class ViTTriplaneDecomposed_SR_v1_GN_twoconv_affine_true_track_mean_patchUpsampleHybrid(
        ViTTriplaneDecomposed_SR_v1_IN_twoconv_affine_true_track_mean_patchUpsampleHybrid
):

    def __init__(
        self,
        vit_decoder: VisionTransformer,
        triplane_decoder: Triplane,
        cls_token,
        normalize_feat=True,
        sr_ratio=2,
    ) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio=sr_ratio,
        )

        self.normalize_feat = normalize_feat
        self.IN = nn.Identity()

        self.superresolution.update(
            {'norm': GroupNorm2dDecopmosed(triplane_decoder.out_chans)})

    def vit_decode(self, latent, img_size):

        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        if self.cls_token:
            # latent, cls_token = latent[:, 1:], latent[:, :1]
            cls_token = latent[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size

        # logger.log('after decode_pred: ', latent.shape)

        # ! upsample dimention to 32

        latent = self.unpatchify(
            latent,
            p=7,
        )  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * smooth convolution before triplane
        # latent = self.superresolution['conv1'](latent) # 2D->3D projectio, no residual connection.

        latent = self.superresolution['proj_upsample'](
            latent,
            input_skip_connection=False)  # still needed the above conv?

        # ! resize from 448 triplane to 336
        if latent.shape[-1] == 448:
            latent = torch.nn.Upsample(size=(336, 336), mode='bilinear')(
                latent)  # to alleviate diffusion learning difficulty

        latent_normalized = self.superresolution['norm'].norm_wo_affine(
            latent)  # type: ignore

        # return latent_normalized  # unit normal distribution

        return dict(latent_normalized=latent_normalized, cls_token=cls_token)
        # ret_dict.update({'latent_normalized': latent_normalized, 'latent_denormalized': latent_denormalized})

    def triplane_decode(self, latent_normalized, c, return_raw_only=False):

        if isinstance(latent_normalized, dict):
            latent_normalized, sr_w_code = (latent_normalized.get(k, None)
                                            for k in ('latent_normalized',
                                                      'sr_w_code'))
            # latent_normalized = (latent_normalized[k] for k in ('latent_normalized'))
        else:
            sr_w_code = None

        latent_normalized = self.superresolution['norm'].norm_affine(
            latent_normalized)  # type: ignore
        latent_denormalized = self.superresolution['conv2'](latent_normalized)

        # * triplane rendering
        ret_dict = self.triplane_decoder(
            latent_denormalized,
            c,
            ws=sr_w_code,
            return_raw_only=return_raw_only)  # triplane latent -> imgs
        ret_dict.update(
            dict(sr_w_code=sr_w_code,
                 latent_denormalized=latent_denormalized,
                 latent_normalized=latent_normalized))

        return ret_dict


class ViTTriplaneDecomposed_SR_v1_GN_twoconv_affine_true_track_mean_patchUpsampleHybridSR(
        ViTTriplaneDecomposed_SR_v1_GN_twoconv_affine_true_track_mean_patchUpsampleHybrid
):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane,
            cls_token,
            normalize_feat=True,
            w_avg=torch.zeros([512]),
            sr_ratio=2,
    ) -> None:
        super().__init__(vit_decoder,
                         triplane_decoder,
                         cls_token,
                         normalize_feat,
                         sr_ratio=sr_ratio)

        self.w_proj = Mlp(self.vit_decoder.num_features, out_features=512)
        assert self.cls_token, 'requires [cls] to project SR w code'

        self.register_buffer('w_avg', w_avg)  # will replace externally

    def vit_decode(self, latent, img_size):

        vit_decode_dict = super().vit_decode(latent, img_size)

        sr_w_code = self.w_proj(
            vit_decode_dict['cls_token']) + self.w_avg.reshape(
                1, 1, -1)  # type: ignore

        vit_decode_dict.update(dict(sr_w_code=sr_w_code))

        return vit_decode_dict


class ViTTriplaneDecomposed_SR_v1_IN_twoconv_track_mean(ViTTriplaneDecomposed):
    """v1, add basic convolution layer before triplane decoder
    """

    def __init__(
        self,
        vit_decoder: VisionTransformer,
        triplane_decoder: Triplane,
        cls_token,
        normalize_feat=True,
    ) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token)
        self.superresolution = nn.ModuleDict({
            'conv1':
            nn.Conv2d(triplane_decoder.out_chans,
                      triplane_decoder.out_chans,
                      3,
                      padding=1),
            'conv2':
            nn.Conv2d(triplane_decoder.out_chans,
                      triplane_decoder.out_chans,
                      3,
                      padding=1)
        })

        if normalize_feat:
            self.IN = nn.InstanceNorm2d(
                triplane_decoder.out_chans,
                affine=False,
                track_running_stats=True)  # simply do the normalization
        self.normalize_feat = normalize_feat

    def triplane_decode(self, latent_normalized, c):

        latent_denormalized = self.superresolution['conv2'](latent_normalized)

        # * triplane rendering
        ret_dict = self.triplane_decoder(latent_denormalized,
                                         c)  # triplane latent -> imgs
        ret_dict.update({
            'latent_normalized': latent_normalized,
            'latent_denormalized': latent_denormalized
        })

        return ret_dict

    def vit_decode(self, latent, img_size):

        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * smooth convolution before triplane
        latent = self.superresolution['conv1'](latent)

        if self.normalize_feat:
            # latent = (latent - 10) / (3 * 50)
            latent_normalized = self.IN(latent).clamp(
                -50, 50)  # normalize to mean 0, std 1
        else:
            latent_normalized = latent.clamp(-50, 50)
            # latent = latent.clamp(-1, 1)
            # latent = (latent / 3. ).clamp(-1, 1)  # contain the 3-sigma range, clip to [-1,1]

        return latent_normalized

    def forward(self, latent, c, img_size):

        latent_normalized = self.vit_decode(latent, img_size)
        return self.triplane_decode(latent_normalized, c)


class ViTTriplaneDecomposed_SR_v1_centralizefeats(ViTTriplaneDecomposed):
    """v1, add basic convolution layer before triplane decoder
    """

    def __init__(
        self,
        vit_decoder: VisionTransformer,
        triplane_decoder: Triplane,
        cls_token,
        normalize_feat=True,
    ) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token)
        self.superresolution = nn.Conv2d(triplane_decoder.out_chans,
                                         triplane_decoder.out_chans,
                                         3,
                                         padding=1)

        self.normalize_feat = normalize_feat
        # if normalize_feat:
        # self.BN_norm = nn.BatchNorm2d(vit_decoder.embed_dim, affine=False) # simply do the normalization

    def forward(self, latent, c, img_size):
        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * smooth convolution before triplane
        latent = self.superresolution(latent)

        if self.normalize_feat:
            # latent = (latent - 10) / (3 * 50)
            latent = latent - latent.mean(dim=1, keepdim=True)  # B C H W
            latent = latent.clamp(-50, 50)
            # latent = latent.clamp(-1, 1)
            # latent = self.BN_norm(latent) # normalize to mean 0, std 1
            # latent = (latent / 3. ).clamp(-1, 1)  # contain the 3-sigma range, clip to [-1,1]

        # * triplane rendering
        ret_dict = self.triplane_decoder(latent, c)  # triplane latent -> imgs
        ret_dict.update({'latent': latent})

        return ret_dict


class ViTTriplaneDecomposed_SR_v1_ldmvae(ViTTriplaneDecomposed):
    """v1, add basic convolution layer before triplane decoder
    """

    def __init__(
        self,
        vit_decoder: VisionTransformer,
        triplane_decoder: Triplane,
        cls_token,
        normalize_feat=True,
    ) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token)

        ldm_z_channels = triplane_decoder.out_chans
        # ldm_embed_dim = 16 # https://github.com/CompVis/latent-diffusion/blob/e66308c7f2e64cb581c6d27ab6fbeb846828253b/models/first_stage_models/kl-f16/config.yaml
        ldm_embed_dim = triplane_decoder.out_chans

        self.superresolution = nn.ModuleDict({
            'after_vit_conv':
            nn.Conv2d(
                triplane_decoder.out_chans,
                triplane_decoder.out_chans * 2,  # for vae features
                3,
                padding=1),
            'quant_conv':
            torch.nn.Conv2d(2 * ldm_z_channels, 2 * ldm_embed_dim, 1),
            'post_quant_conv':
            torch.nn.Conv2d(ldm_embed_dim, ldm_z_channels, 1),
        })
        # self.logvar = torch.nn.Parameter(torch.ones(size=()) * 0) # TODO, add to optimizer

        self.normalize_feat = normalize_feat

    def vae_encoder(self, h):
        # * smooth convolution before triplane
        h = self.superresolution['after_vit_conv'](
            h)  # TODO ablations needed of nor

        moments = self.superresolution['quant_conv'](h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def vae_decode(self, z, c):
        z_triplane = self.superresolution['post_quant_conv'](z)

        ret_dict = self.triplane_decoder(z_triplane,
                                         c)  # triplane latent -> imgs
        ret_dict.update({'latent': z_triplane})

        return ret_dict

    def forward(self, latent, c, img_size, sample_posterior=True):
        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * KL-reg in LDM
        # https://github.com/CompVis/latent-diffusion/blob/e66308c7f2e64cb581c6d27ab6fbeb846828253b/ldm/models/autoencoder.py
        posterior = self.vae_encoder(latent)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        ret_dict = self.vae_decode(z, c)
        ret_dict.update({
            'posterior': posterior,
            # 'logvar': self.logvar
        })

        return ret_dict


class ViTTriplaneDecomposed_SR_v1_dualConv(ViTTriplaneDecomposed):
    # if normalize_feat:
    # self.BN_norm = nn.BatchNorm2d(vit_decoder.embed_dim, affine=False) # simply do the normalization
    """v1, add basic convolution layer before triplane decoder
    observation: novel view fails. 
    """

    def __init__(
        self,
        vit_decoder: VisionTransformer,
        triplane_decoder: Triplane,
        cls_token,
        normalize_feat=True,
    ) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token)
        # self.superresolution = nn.Conv2d(triplane_decoder.out_chans,
        #                                  triplane_decoder.out_chans,
        #                                  3,
        #                                  padding=1)
        self.superresolution = Conv3x3TriplaneTransformation(
            triplane_decoder.out_chans, triplane_decoder.out_chans)

        self.normalize_feat = normalize_feat
        # if normalize_feat:
        # self.BN_norm = nn.BatchNorm2d(vit_decoder.embed_dim, affine=False) # simply do the normalization

    def forward(self, latent, c, img_size):
        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * smooth convolution before triplane
        latent = self.superresolution(latent)

        if self.normalize_feat:
            # latent = (latent - 10) / (3 * 50)
            latent = latent.clamp(-50, 50)
            # latent = latent.clamp(-1, 1)
            # latent = self.BN_norm(latent) # normalize to mean 0, std 1
            # latent = (latent / 3. ).clamp(-1, 1)  # contain the 3-sigma range, clip to [-1,1]

        # * triplane rendering
        ret_dict = self.triplane_decoder(latent, c)  # triplane latent -> imgs
        ret_dict.update({'latent': latent})

        return ret_dict


class ViTTriplaneDecomposed_SR_v1_ablaconv(ViTTriplaneDecomposed):
    """v1, add basic convolution layer before triplane decoder
    """

    def __init__(
        self,
        vit_decoder: VisionTransformer,
        triplane_decoder: Triplane,
        cls_token,
        normalize_feat=True,
    ) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token)
        # self.superresolution = nn.Conv2d(triplane_decoder.out_chans,
        #                                  triplane_decoder.out_chans,
        #                                  3,
        #                                  padding=1)

        self.normalize_feat = normalize_feat
        # if normalize_feat:
        # self.BN_norm = nn.BatchNorm2d(vit_decoder.embed_dim, affine=False) # simply do the normalization

    def forward(self, latent, c, img_size):
        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * smooth convolution before triplane
        # latent = self.superresolution(latent)

        if self.normalize_feat:
            # latent = (latent - 10) / (3 * 50)
            latent = latent.clamp(-50, 50)
            # latent = latent.clamp(-1, 1)
            # latent = self.BN_norm(latent) # normalize to mean 0, std 1
            # latent = (latent / 3. ).clamp(-1, 1)  # contain the 3-sigma range, clip to [-1,1]

        # * triplane rendering
        ret_dict = self.triplane_decoder(latent, c)  # triplane latent -> imgs
        ret_dict.update({'latent': latent})

        return ret_dict


class ViTTriplaneDecomposed_SR_v1_withResidual(ViTTriplaneDecomposed):
    """v1, add basic convolution layer before triplane decoder
    observation: training view improves; novel view fails.
    Ablations/Reconstruction/l2_loss_SR_v1_resblk/runs
    """

    def __init__(
        self,
        vit_decoder: VisionTransformer,
        triplane_decoder: Triplane,
        cls_token,
        normalize_feat=True,
    ) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token)
        self.superresolution = nn.Conv2d(triplane_decoder.out_chans,
                                         triplane_decoder.out_chans,
                                         3,
                                         padding=1)

        self.normalize_feat = normalize_feat
        # if normalize_feat:
        # self.BN_norm = nn.BatchNorm2d(vit_decoder.embed_dim, affine=False) # simply do the normalization

    def forward(self, latent, c, img_size):
        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * smooth convolution before triplane
        latent = self.superresolution(latent) + latent

        if self.normalize_feat:
            # latent = (latent - 10) / (3 * 50)
            latent = latent.clamp(-50, 50)
            # latent = latent.clamp(-1, 1)
            # latent = self.BN_norm(latent) # normalize to mean 0, std 1
            # latent = (latent / 3. ).clamp(-1, 1)  # contain the 3-sigma range, clip to [-1,1]

        # * triplane rendering
        ret_dict = self.triplane_decoder(latent, c)  # triplane latent -> imgs
        ret_dict.update({'latent': latent})

        return ret_dict


class ViTTriplaneDecomposed_SR_v1_3DViT(ViTTriplaneDecomposed):
    """v1 + vit on the triplane
    observation: even worse than resblk baseline. Worse under both train and test performance.
    """

    def __init__(
        self,
        vit_decoder: VisionTransformer,
        triplane_decoder: Triplane,
        cls_token,
        normalize_feat=True,
    ) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            decoder_pred_size=(vit_decoder.patch_embed.patch_size)**2 *
            (triplane_decoder.out_chans // 3),
            unpatchify_out_chans=triplane_decoder.out_chans // 3,
        )

        self.superresolution = nn.Conv2d(triplane_decoder.out_chans,
                                         triplane_decoder.out_chans,
                                         3,
                                         padding=1)

        self.normalize_feat = normalize_feat

        self.decoder_pred_3d = nn.Linear(self.vit_decoder.embed_dim,
                                         self.vit_decoder.embed_dim * 3,
                                         bias=True)

        self.transformer_3D_blk = Block(
            dim=self.vit_decoder.embed_dim,
            num_heads=12,
            mlp_ratio=4,
        )  # vit_base cfg

        # if normalize_feat:
        # self.BN_norm = nn.BatchNorm2d(vit_decoder.embed_dim, affine=False) # simply do the normalization

    def unpatchify_triplane(self, x, p=None, unpatchify_out_chans=None):
        """
        x: (N, L, patch_size**2 * self.out_chans)
        imgs: (N, self.out_chans, H, W)
        """
        if unpatchify_out_chans is None:
            unpatchify_out_chans = self.unpatchify_out_chans
        # p = self.vit_decoder.patch_size
        if self.cls_token:  # TODO, how to better use cls token
            x = x[:, 1:]

        if p is None:  # assign upsample patch size
            p = self.patch_size
        h = w = int((x.shape[1] // 3)**.5)
        assert h * w * 3 == x.shape[1]

        x = x.reshape(shape=(x.shape[0], 3, h, w, p, p, unpatchify_out_chans))
        x = torch.einsum('ndhwpqc->ndchpwq',
                         x)  # nplanes, C order in the renderer.py
        triplanes = x.reshape(shape=(x.shape[0], unpatchify_out_chans * 3,
                                     h * p, h * p))
        return triplanes

    def forward(self, latent, c, img_size):
        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        # ! change here, add 3D transformer
        latent = self.decoder_pred_3d(latent)

        # unpatchify 3D dimention
        if self.cls_token:  # TODO, how to better use cls token
            latent = latent[:, 1:]

        # h = w = int(latent.shape[1]**.5)
        # assert h * w == latent.shape[1]
        latent = latent.reshape(latent.shape[0], latent.shape[1],
                                self.vit_decoder.embed_dim,
                                3).permute(0, 3, 1, 2)  # B 3 L C
        latent = latent.reshape(latent.shape[0], -1,
                                self.vit_decoder.embed_dim)  # B 3*L, C

        latent = self.transformer_3D_blk(latent)  # B 3*L C

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(latent)  # B 3 L H W 8
        latent = self.unpatchify_triplane(
            latent)  # spatial_vit_latent, B, C, H, W

        # * smooth convolution before triplane
        latent = self.superresolution(latent)

        if self.normalize_feat:
            latent = latent.clamp(-50, 50)

        # * triplane rendering
        ret_dict = self.triplane_decoder(latent, c)  # triplane latent -> imgs
        ret_dict.update({'latent': latent})

        return ret_dict


class ViTTriplaneDecomposed_SR_v3(ViTTriplaneDecomposed):
    """v3, imagen unshuffle SR the triplane with imagen upsapmle
    """

    def __init__(self, vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane, cls_token) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token)
        self.superresolution = PixelShuffleUpsample(
            triplane_decoder.out_chans,
            triplane_decoder.out_chans // 2)  # TODO
        # self.triplane_decoder.unshuffle_upsample1 = PixelShuffleUpsample(triplane_decoder.out_chans // 2, triplane_decoder.out_chans // 4)

    def forward(self, latent, c):
        latent = self.forward_vit_decoder(latent, False)  # pred_vit_latent

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * smooth convolution before triplane
        latent = self.superresolution(latent)  # B C//2, H*2, W*2
        # latent = self.triplane_decoder.unshuffle_upsample0(latent)

        # TODO 2D convolutions -> Triplane
        # * triplane rendering
        ret_dict = self.triplane_decoder(latent, c)  # triplane latent -> imgs

        return ret_dict


class ViTTriplaneDecomposed_SR_v4(ViTTriplaneDecomposed):
    """v4, imagen unshuffle SR final image
    """

    def __init__(self, vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane, cls_token) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token)

        self.superresolution0 = PixelShuffleUpsample(
            triplane_decoder.decoder.output_dim, 128)
        self.superresolution1 = PixelShuffleUpsample(128, 3)

    def forward(self, latent, c):

        # ViT decoder
        latent = self.forward_vit_decoder(latent, False)  # pred_vit_latent
        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # TODO 2D convolutions -> Triplane
        # * triplane rendering
        ret_dict = self.triplane_decoder(latent, c)  # triplane latent -> imgs

        # N = latent.shape[0]
        # H = W = self.triplane_decoder.neural_rendering_resolution
        # feature_image = ret_dict['feature_samples'].permute(0, 2, 1).reshape(
        #     N, ret_dict['feature_samples'].shape[-1], H, W).contiguous()  # B 32 H W
        feature_image = ret_dict['feature_image']

        feature_image = self.superresolution0(feature_image)
        sr_image = self.superresolution1(feature_image)
        # latent = self.triplane_decoder.unshuffle_upsample0(latent)
        ret_dict.update({'sr_image': sr_image})
        return ret_dict


class ViTTriplaneDecomposed_SR_pixelunshuffle_Upsampler(ViTTriplaneDecomposed):
    """v4, imagen unshuffle SR final image
    """

    def __init__(self, vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane) -> None:
        super().__init__(vit_decoder, triplane_decoder)

        # https://github.com/JingyunLiang/SwinIR/blob/6545850fbf8df298df73d81f3e8cba638787c8bd/models/network_swinir.py#L740
        # if self.upsampler == 'pixelshuffle':
        # for classical SR

        num_feat = 128
        num_out_ch = 3

        # self.conv_after_body = nn.Conv2d(triplane_decoder.decoder.output_dim, triplane_decoder.decoder.output_dim, 3, 1, 1)
        # self.conv_before_upsample = nn.Sequential(nn.Conv2d(triplane_decoder.decoder.output_dim, num_feat, 3, 1, 1),
        #                                             nn.LeakyReLU(inplace=True))
        # self.upsample = Upsample(4, num_feat) # 4 time SR
        # self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.superresolution = PixelUnshuffleUpsample(
            triplane_decoder.decoder.output_dim, num_feat, num_out_ch)

    def forward(self, latent, c):

        # ViT decoder
        latent = self.forward_vit_decoder(latent, False)  # pred_vit_latent
        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # TODO 2D convolutions -> Triplane
        # * triplane rendering
        ret_dict = self.triplane_decoder(latent, c)  # triplane latent -> imgs

        # N = latent.shape[0]
        # H = W = self.triplane_decoder.neural_rendering_resolution
        # feature_image = ret_dict['feature_samples'].permute(0, 2, 1).reshape(
        #     N, ret_dict['feature_samples'].shape[-1], H, W).contiguous()  # B 32 H W
        x = self.superresolution(ret_dict['feature_image'])

        ret_dict.update({'image_sr': x})
        return ret_dict


class ViTTriplaneDecomposed_SR_pixelunshuffle_Upsampler_withTriplaneSmooth(
        ViTTriplaneDecomposed):

    def __init__(self, vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane) -> None:
        super().__init__(vit_decoder, triplane_decoder)
        triplane_superresolution = nn.Conv2d(triplane_decoder.out_chans,
                                             triplane_decoder.out_chans,
                                             3,
                                             padding=1)
        self.superresolution = nn.ModuleDict({
            'triplane':
            triplane_superresolution,
            'pixelunshuffle':
            PixelUnshuffleUpsample(triplane_decoder.decoder.output_dim, 128, 3)
        })

    def forward(self, latent, c):

        # ViT decoder
        latent = self.forward_vit_decoder(latent, False)  # pred_vit_latent
        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * tripane smooth
        latent = self.superresolution['triplane'](latent)  # B C//2, H*2, W*2

        # TODO 2D convolutions -> Triplane
        # * triplane rendering
        ret_dict = self.triplane_decoder(latent, c)  # triplane latent -> imgs

        # N = latent.shape[0]
        # H = W = self.triplane_decoder.neural_rendering_resolution
        # feature_image = ret_dict['feature_samples'].permute(0, 2, 1).reshape(
        #     N, ret_dict['feature_samples'].shape[-1], H, W).contiguous()  # B 32 H W
        x = self.superresolution['pixelunshuffle'](ret_dict['feature_image'])

        ret_dict.update({'image_sr': x})
        return ret_dict


class ViTTriplaneDecomposed_pixelunshuffle_upsample_triplane(
        ViTTriplaneDecomposed):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token=False) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token)
        triplane_superresolution = nn.Sequential(
            PixelUnshuffleUpsample(self.vit_decoder.embed_dim, 256, 192),
            PixelUnshuffleUpsample(192, 128, 96),
        )
        self.superresolution = nn.ModuleDict({
            'triplane':
            triplane_superresolution,
            # 'pixelunshuffle':
            # PixelUnshuffleUpsample(triplane_decoder.decoder.output_dim, 128, 3)
        })

        self.decoder_pred = nn.Identity()

    def forward(self, latent, c, img_size):

        # ViT decoder
        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent
        # ViT decoder projection, from MAE

        # latent = self.decoder_pred(
        #     latent)  # pred_vit_latent -> patch or original size
        batch_size, num_patches, _ = latent.shape

        h = w = int(num_patches**.5)
        assert h * w == num_patches

        latent = latent.reshape(batch_size, h, w, -1).permute(0, 3, 1,
                                                              2)  # B H W C

        # pixelunshuffle unsample
        # logger.log(latent.shape)
        latent = self.superresolution['triplane'](latent)  # B 96, H*16, W*16
        # logger.log(latent.shape)

        # latent = self.unpatchify(
        #     latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # TODO 2D convolutions -> Triplane
        # * triplane rendering
        ret_dict = self.triplane_decoder(latent, c)  # triplane latent -> imgs

        # N = latent.shape[0]
        # H = W = self.triplane_decoder.neural_rendering_resolution
        # feature_image = ret_dict['feature_samples'].permute(0, 2, 1).reshape(
        #     N, ret_dict['feature_samples'].shape[-1], H, W).contiguous()  # B 32 H W
        # x = self.superresolution['pixelunshuffle'](ret_dict['feature_image'])

        # ret_dict.update({'image_sr': x})
        return ret_dict


class ViTTriplaneDecomposed_pixelunshuffle_upsample_triplane_hybrid(
        ViTTriplaneDecomposed):

    def __init__(
        self,
        vit_decoder: VisionTransformer,
        triplane_decoder: Triplane,
        cls_token=False,
        normalize_feat=True,
        # normalize_std_mean=(51, 10),
    ) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            decoder_pred_size=(vit_decoder.patch_embed.patch_size // 4)**2 *
            (triplane_decoder.out_chans * 2),
            unpatchify_out_chans=triplane_decoder.out_chans * 2)
        triplane_superresolution = nn.Sequential(
            # PixelUnshuffleUpsample(self.vit_decoder.embed_dim, 256, 192),
            PixelUnshuffleUpsample(output_dim=triplane_decoder.out_chans * 2,
                                   num_feat=int(triplane_decoder.out_chans *
                                                1.5),
                                   num_out_ch=triplane_decoder.out_chans), )
        self.superresolution = nn.ModuleDict({
            'triplane':
            triplane_superresolution,
            # 'pixelunshuffle':
            # PixelUnshuffleUpsample(triplane_decoder.decoder.output_dim, 128, 3)
        })

        self.normalize_feat = normalize_feat
        # if normalize_feat:
        #     self.BN_norm = nn.BatchNorm2d(triplane_decoder.out_chans * 2)
        # self.normalize_feat = normalize_feat
        # self.normalize_mean_std = normalize_mean_std  # for diffusion, (-1,1) range

        # self.decoder_pred = nn.Identity()

    def forward(self, latent, c, img_size):

        # ViT decoder
        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent
        # ViT decoder projection, from MAE

        # logger.log('latent after vit deocder', latent.shape)
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        # logger.log('latent after decoder_pred', latent.shape)
        # batch_size, num_patches, _ = latent.shape

        latent = self.unpatchify(
            latent, p=4)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)
        # logger.log('unpachified latent', latent.shape)

        # h = w = int(num_patches**.5)
        # assert h * w == num_patches

        # latent = latent.reshape(batch_size, h, w, -1).permute(0, 3, 1,
        #                                                       2)  # B H W C

        # pixelunshuffle unsample
        # logger.log(latent.shape)

        # if self.normalize_feat:
        #     latent = self.BN_norm(latent) # normalize to mean 0, std 1

        latent = self.superresolution['triplane'](latent)  # B 96, H*16, W*16

        if self.normalize_feat:
            latent = latent.clamp(-50, 50)

        # logger.log('SR latent', latent.shape)
        # logger.log(latent.shape)

        #     latent = (latent / 3. ).clamp(-1, 1)  # contain the 3-sigma range, clip to [-1,1]
        #     latent = (latent - self.normalize_mean_std[1]) / (
        #         4 * self.normalize_mean_std[1])
        #     latent = torch.clip(latent, 0, 1)

        # TODO 2D convolutions -> Triplane
        # * triplane rendering

        ret_dict = self.triplane_decoder(latent, c)  # triplane latent -> imgs
        ret_dict.update({'latent': latent})

        # N = latent.shape[0]
        # H = W = self.triplane_decoder.neural_rendering_resolution
        # feature_image = ret_dict['feature_samples'].permute(0, 2, 1).reshape(
        #     N, ret_dict['feature_samples'].shape[-1], H, W).contiguous()  # B 32 H W
        # x = self.superresolution['pixelunshuffle'](ret_dict['feature_image'])

        # ret_dict.update({'image_sr': x})
        return ret_dict


# class ViTTriplaneDecomposed_pixelunshuffle_upsample_triplane_hybrid(
#         ViTTriplaneDecomposed):


class ViTTriplaneDecomposed_pixelunshuffle_upsample_triplane_hybrid_MLPAblation(
        ViTTriplaneDecomposed):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token=False,
                 normalize_feat=True) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            decoder_pred_size=(vit_decoder.patch_embed.patch_size // 4)**2 *
            (triplane_decoder.out_chans * 2),
            unpatchify_out_chans=triplane_decoder.out_chans * 2,
        )
        #  normalize_feat=normalize_feat)

        self.normalize_feat = normalize_feat

        triplane_superresolution = nn.Sequential(
            nn.Linear(triplane_decoder.out_chans * 2,
                      triplane_decoder.out_chans,
                      bias=True)  # decoder to pat
        )
        self.superresolution = nn.ModuleDict({
            'triplane':
            triplane_superresolution,
            # 'pixelunshuffle':
            # PixelUnshuffleUpsample(triplane_decoder.decoder.output_dim, 128, 3)
        })

    def forward(self, latent, c, img_size):

        # ViT decoder
        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify(
            latent, p=4)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)
        # if self.normalize_feat:

        latent = self.superresolution['triplane'](
            latent.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2).contiguous(
        )  # simply dim up using MLP, ablation convolution upsample (same dimention, smaller resolution)

        # st()

        if self.normalize_feat:
            latent = latent.clamp(-50, 50)

        ret_dict = self.triplane_decoder(latent, c)  # triplane latent -> imgs
        ret_dict.update({'latent': latent})

        return ret_dict


class ViTTriplaneDecomposed_pixelunshuffle_upsample_triplane_two_upsample_mlp(
        ViTTriplaneDecomposed):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token=False,
                 normalize_feat=True) -> None:
        Mlp_dim_mult = 8

        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            decoder_pred_size=(vit_decoder.patch_embed.patch_size // 4)**2 *
            (triplane_decoder.out_chans * Mlp_dim_mult),
            unpatchify_out_chans=triplane_decoder.out_chans * Mlp_dim_mult,
        )
        #  normalize_feat=normalize_feat)

        self.normalize_feat = normalize_feat

        triplane_superresolution = ResMlp(
            triplane_decoder.out_chans * Mlp_dim_mult,
            size_out=(4)**2 * triplane_decoder.out_chans,
            drop=0.1)
        self.superresolution = nn.ModuleDict({
            'triplane_upsample_mlp':
            triplane_superresolution,
        })

    def forward(self, latent, c, img_size):

        # ViT decoder
        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify(
            latent,
            p=4)  # spatial_vit_latent, B, C, H, W (B, 24*Mlp_dim_mult, 56, 56)

        latent = self.superresolution['triplane_upsample_mlp'](latent.permute(
            0, 2, 3, 1))  # .permute(0, 3, 1, 2).contiguous() # B H W 24*16
        latent = latent.reshape(
            latent.shape[0], -1,
            latent.shape[-1])  # merge H*W -> L for unpachify

        latent = self.unpatchify(
            latent, p=4, unpatchify_out_chans=self.triplane_decoder.out_chans
        )  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        if self.normalize_feat:
            latent = latent.clamp(-50, 50)

        ret_dict = self.triplane_decoder(latent, c)  # triplane latent -> imgs
        ret_dict.update({'latent': latent})

        return ret_dict


class ViTTriplaneDecomposed_pixelunshuffle_upsample_triplane_two_upsample_mlp_triplane448(
        ViTTriplaneDecomposed):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token=False,
                 normalize_feat=True) -> None:
        self.Mlp_dim_mult1 = 8  # TODO, change to 12 after validating larger triplane works
        self.Mlp_dim_mult2 = 6

        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            decoder_pred_size=(vit_decoder.patch_embed.patch_size // 4)**2 *
            (triplane_decoder.out_chans * self.Mlp_dim_mult1),
            unpatchify_out_chans=triplane_decoder.out_chans *
            self.Mlp_dim_mult1,
        )
        #  normalize_feat=normalize_feat)

        self.normalize_feat = normalize_feat

        triplane_superresolution_56to224 = ResMlp(
            triplane_decoder.out_chans * self.Mlp_dim_mult1,
            size_out=(4)**2 * triplane_decoder.out_chans * self.Mlp_dim_mult2,
            drop=0.1)

        triplane_superresolution_224to448 = ResMlp(
            triplane_decoder.out_chans * self.Mlp_dim_mult2,
            size_out=(2)**2 * triplane_decoder.out_chans,
            drop=0.1)

        self.superresolution = nn.ModuleDict({
            'triplane_upsample_mlp1':
            triplane_superresolution_56to224,
            'triplane_upsample_mlp2':
            triplane_superresolution_224to448,
        })

    def forward(self, latent, c, img_size):

        # ViT decoder
        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify(
            latent,
            p=4,
            unpatchify_out_chans=self.triplane_decoder.out_chans *
            self.Mlp_dim_mult1
        )  # spatial_vit_latent, B, C, H, W (B, 24*Mlp_dim_mult, 56, 56)

        # ======== upsample triplane 56 -> 224 =========
        latent = self.superresolution['triplane_upsample_mlp1'](latent.permute(
            0, 2, 3, 1))
        latent = latent.reshape(
            latent.shape[0], -1,
            latent.shape[-1])  # merge H*W -> L for unpachify

        latent = self.unpatchify(
            latent,
            p=4,
            unpatchify_out_chans=self.triplane_decoder.out_chans *
            self.Mlp_dim_mult2
        )  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # ======== upsample triplane 224 -> 448 =========

        latent = self.superresolution['triplane_upsample_mlp2'](latent.permute(
            0, 2, 3, 1))
        latent = latent.reshape(
            latent.shape[0], -1,
            latent.shape[-1])  # merge H*W -> L for unpachify

        latent = self.unpatchify(
            latent, p=2, unpatchify_out_chans=self.triplane_decoder.out_chans
        )  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # upsample down

        if self.normalize_feat:
            latent = latent.clamp(-50, 50)

        ret_dict = self.triplane_decoder(latent, c)  # triplane latent -> imgs
        ret_dict.update({'latent': latent})

        return ret_dict


# =============== Ablations dim & two transformations ==================


class ViTTriplaneDecomposed_pixelunshuffle_upsample_triplane_two_upsample_mlp_abladim_and_two_transformations(
        ViTTriplaneDecomposed):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token=False,
                 normalize_feat=True) -> None:
        Mlp_dim_mult = 2

        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            decoder_pred_size=(3)**2 *
            (triplane_decoder.out_chans * Mlp_dim_mult),
            unpatchify_out_chans=triplane_decoder.out_chans * Mlp_dim_mult,
        )
        #  normalize_feat=normalize_feat)

        self.normalize_feat = normalize_feat

        triplane_superresolution = ResMlp(
            triplane_decoder.out_chans * Mlp_dim_mult,
            size_out=(2)**2 * triplane_decoder.out_chans,
            drop=0.1)

        self.superresolution = nn.ModuleDict({
            'triplane_upsample_mlp':
            triplane_superresolution,
        })

    def forward(self, latent, c, img_size):

        # ViT decoder
        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify(
            latent,
            p=3)  # spatial_vit_latent, B, C, H, W (B, 24*Mlp_dim_mult, 56, 56)

        latent = self.superresolution['triplane_upsample_mlp'](latent.permute(
            0, 2, 3, 1))  # .permute(0, 3, 1, 2).contiguous() # B H W 24*16
        latent = latent.reshape(
            latent.shape[0], -1,
            latent.shape[-1])  # merge H*W -> L for unpachify

        latent = self.unpatchify(
            latent, p=2, unpatchify_out_chans=self.triplane_decoder.out_chans
        )  # spatial_vit_latent, B, C, H, W (B, 24, 252, 252)

        if self.normalize_feat:
            latent = latent.clamp(-50, 50)

        ret_dict = self.triplane_decoder(latent, c)  # triplane latent -> imgs
        ret_dict.update({'latent': latent})

        return ret_dict


class ViTTriplaneDecomposed_pixelunshuffle_upsample_triplane_two_upsample_mlp_abladim_and_two_transformations_126(
        ViTTriplaneDecomposed):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token=False,
                 normalize_feat=True) -> None:
        Mlp_dim_mult = 1

        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            decoder_pred_size=(3)**2 *
            (triplane_decoder.out_chans * Mlp_dim_mult),
            unpatchify_out_chans=triplane_decoder.out_chans * Mlp_dim_mult,
        )
        #  normalize_feat=normalize_feat)

        self.normalize_feat = normalize_feat

        # triplane_superresolution = ResMlp(
        #     triplane_decoder.out_chans * Mlp_dim_mult,
        #     size_out=(2)**2 * triplane_decoder.out_chans,
        #     drop=0.1)

        # self.superresolution = nn.ModuleDict({
        #     'triplane_upsample_mlp':
        #     triplane_superresolution,
        # })

    def forward(self, latent, c, img_size):

        # ViT decoder
        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify(
            latent,
            p=3)  # spatial_vit_latent, B, C, H, W (B, 24*Mlp_dim_mult, 56, 56)

        # latent = self.superresolution['triplane_upsample_mlp'](latent.permute(
        #     0, 2, 3, 1))  # .permute(0, 3, 1, 2).contiguous() # B H W 24*16
        # latent = latent.reshape(
        #     latent.shape[0], -1,
        #     latent.shape[-1])  # merge H*W -> L for unpachify

        # latent = self.unpatchify(
        #     latent, p=2, unpatchify_out_chans=self.triplane_decoder.out_chans
        # )  # spatial_vit_latent, B, C, H, W (B, 24, 252, 252)

        if self.normalize_feat:
            latent = latent.clamp(-50, 50)

        ret_dict = self.triplane_decoder(latent, c)  # triplane latent -> imgs
        ret_dict.update({'latent': latent})

        return ret_dict


class ViTTriplane(VisionTransformer):

    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,  # Check ViT encoder dim
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer='nn.LayerNorm',
        out_chans=96,
        cls_token=True,
        # * class TriPlaneGenerator(nn.Module)
        c_dim=25,  # Conditioning label (C) dimensionality.
        img_resolution=128,  # Output resolution.
        img_channels=3,  # Number of output color channels.
        rendering_kwargs={},
        # **synthesis_kwargs,  # Arguments for SynthesisNetwork.
    ):

        super().__init__(img_size,
                         patch_size,
                         in_chans,
                         num_classes,
                         embed_dim,
                         depth,
                         num_heads,
                         mlp_ratio,
                         qkv_bias,
                         qk_scale,
                         drop_rate,
                         attn_drop_rate,
                         drop_path_rate,
                         norm_layer,
                         patch_embedding=False,
                         cls_token=cls_token)

        self.out_chans = out_chans
        self.patch_size = patch_size

        # * hyper params
        # * =================== copied from triplane =================== *
        self.c_dim = c_dim
        self.img_resolution = img_resolution  # TODO
        self.img_channels = img_channels

        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()
        self.decoder = OSGDecoder(
            32, {
                'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1),
                'decoder_output_dim': 32
            })

        self.neural_rendering_resolution = 128  # TODO
        self.rendering_kwargs = rendering_kwargs
        # self._last_planes = None
        # self.pool_256 = torch.nn.AdaptiveAvgPool2d((256, 256))

        # *=================== token to spatial feature map =================== *
        # * from MAE, decoder token projection to patch
        # TODO, needs embed PE, norm and decoder_pred projection below
        self.decoder_pred = nn.Linear(embed_dim,
                                      patch_size**2 * out_chans,
                                      bias=True)  # decoder to patch
        # TODO convolution upsample block

    def forward_vit_decoder(self, latent, unpachify=False):
        # latent: (N, L, C) from DINO/CLIP ViT encoder
        # latent = self.prepare_tokens(latent)

        # B, nc, w, h = latent.shape

        # add positional encoding to each token
        latent = latent + self.interpolate_pos_encoding(
            latent, self.img_size, self.img_size)  # B, L, C

        for blk in self.blocks:
            latent = blk(latent)
        latent = self.norm(latent)  # spatial feature map w/o cls token
        if self.cls_token is not None:
            latent = latent[:, 1:]

        if unpachify:
            latent = self.unpatchify(latent)  # spatial_vit_latent

        return latent

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * self.out_chans)
        imgs: (N, self.out_chans, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.out_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.out_chans, h * p, h * p))
        return imgs

    # * pure reconstruction
    def forward_triplane_decoder(self,
                                 planes,
                                 c,
                                 neural_rendering_resolution=None,
                                 update_emas=False,
                                 cache_backbone=False,
                                 use_cached_backbone=False,
                                 return_meta=False,
                                 return_raw_only=False,
                                 **synthesis_kwargs):

        return_sampling_details_flag = self.rendering_kwargs.get(
            'return_sampling_details_flag', False)

        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        # cam2world_matrix = torch.eye(4, device=c.device).unsqueeze(0).repeat_interleave(c.shape[0], dim=0)
        # c[:, :16] = cam2world_matrix.view(-1, 16)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        H = W = self.neural_rendering_resolution
        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(
            cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2],
                             planes.shape[-1])  # BS 96 256 256

        # Perform volume rendering
        rendering_details = self.renderer(planes,
                                          self.decoder,
                                          ray_origins,
                                          ray_directions,
                                          self.rendering_kwargs,
                                          return_meta=return_meta)

        feature_samples, depth_samples, weights_samples = (
            rendering_details[k]
            for k in ['feature_samples', 'depth_samples', 'weights_samples'])

        if return_sampling_details_flag:
            shape_synthesized = rendering_details['shape_synthesized']
        else:
            shape_synthesized = None

        # Reshape into 'raw' neural-rendered image
        feature_image = feature_samples.permute(0, 2, 1).reshape(
            N, feature_samples.shape[-1], H, W).contiguous()  # B 32 H W
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]  # B 3 H W
        sr_image = rgb_image

        ret_dict = {
            'image': sr_image,
            'image_raw': rgb_image,
            'image_depth': depth_image,
            'weights_samples': weights_samples,
            'shape_synthesized': shape_synthesized
        }
        if return_meta:
            ret_dict.update({
                'feature_volume':
                rendering_details['feature_volume'],
                'all_coords':
                rendering_details['all_coords'],
                'weights':
                rendering_details['weights'],
            })

        return ret_dict

    def forward(self, latent, c):
        latent = self.forward_vit_decoder(latent, False)  # pred_vit_latent

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # TODO 2D convolutions -> Triplane
        # * triplane rendering
        ret_dict = self.forward_triplane_decoder(latent,
                                                 c)  # triplane latent -> imgs

        return ret_dict


# class ViTTriplaneDecomposed_SR_v2(ViTTriplaneDecomposed):
#     """v2, add eg3d SR decoder
#     32 -> ? how to set. No need to add this.
#     """

#     def __init__(self, vit_decoder: VisionTransformer,
#                  triplane_decoder: Triplane) -> None:
#         super().__init__(vit_decoder, triplane_decoder)
#         self.superresolution = SuperresolutionHybrid4X()

#     def forward(self, latent, c):
#         latent = self.forward_vit_decoder(latent, False)  # pred_vit_latent

#         # ViT decoder projection, from MAE
#         latent = self.decoder_pred(
#             latent)  # pred_vit_latent -> patch or original size
#         latent = self.unpatchify(
#             latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

#         # * smooth convolution before triplane
#         st()
#         ret_dict = self.forward_triplane_decoder(latent,
#                                                  c)  # triplane latent -> imgs

#         st()

#         N = latent.shape[0]
#         H = W = self.triplane_decoder.neural_rendering_resolution
#         feature_image = ret_dict['feature_samples'].permute(0, 2, 1).reshape(
#             N, ret_dict['feature_samples'].shape[-1], H, W).contiguous()  # B 32 H W

#         sr_image = self.superresolution(feature_image)
#         ret_dict.update({'sr_image': sr_image})
#         return ret_dict


class ViTTriplaneDecomposed_SR_v1_GN_twoconv_affine_div6(
        ViTTriplaneDecomposed_SR_v1_IN_twoconv_affine_true_track_mean):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat)

        self.IN = nn.Identity()

        self.superresolution.update({
            'norm':
            GroupNorm2dDecopmosed(triplane_decoder.out_chans, group_divider=6)
        })
        # {'norm': GroupNorm2dDecopmosed(triplane_decoder.out_chans, group_divider=2)})
        self.register_buffer('w_avg',
                             torch.zeros([512]))  # will replace externally

    def vit_decode_backbone(self, latent, img_size):
        return self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

    def vit_decode_postprocess(
        self,
        latent_from_vit,
    ):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit)  # pred_vit_latent -> patch or original size
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # * smooth convolution before triplane
        latent = self.superresolution['conv1'](latent)

        if self.normalize_feat:
            latent_normalized = self.superresolution['norm'].norm_wo_affine(
                latent)  # type: ignore
        else:
            latent_normalized = latent

        return dict(latent_normalized=latent_normalized,
                    cls_token=cls_token,
                    latent_from_vit=latent_from_vit)

    def vit_decode(self, latent, img_size):

        latent_from_vit = self.vit_decode_backbone(latent, img_size)

        return self.vit_decode_postprocess(latent_from_vit)

    def triplane_decode(self,
                        vit_decode_out,
                        c,
                        return_raw_only=False,
                        **kwargs):

        if isinstance(vit_decode_out, dict):
            latent_normalized, sr_w_code = (vit_decode_out.get(k, None)
                                            for k in ('latent_normalized',
                                                      'sr_w_code'))

            latent_normalized = vit_decode_out['latent_normalized']

        else:
            latent_normalized = vit_decode_out
            sr_w_code = None
            vit_decode_out = dict(latent_normalized=latent_normalized
                                  )  # for later dict update compatability

        latent_denormalized = self.superresolution['norm'].norm_affine(
            latent_normalized)  # type: ignore
        latent_denormalized = self.superresolution['conv2'](
            latent_denormalized)

        # * triplane rendering

        ret_dict = self.triplane_decoder(latent_denormalized,
                                         c,
                                         ws=sr_w_code,
                                         return_raw_only=return_raw_only,
                                         **kwargs)  # triplane latent -> imgs
        ret_dict.update({
            # 'latent_normalized': latent_normalized,
            'latent_denormalized': latent_denormalized,
            **vit_decode_out
        })

        return ret_dict


# ! VAE from here
class VAE_V1(ViTTriplaneDecomposed_SR_v1_GN_twoconv_affine):

    def __init__(
        self,
        vit_decoder: VisionTransformer,
        triplane_decoder: Triplane,
        cls_token,
        decoder_pred_size=-1,
        unpatchify_out_chans=-1,
    ) -> None:
        """observation: cano PSNR 25 after one night; nvs collapses (~16PSNR)
        """

        self.reparameterization_soft_clamp = False

        if decoder_pred_size == -1:

            patch_size = vit_decoder.patch_embed.patch_size  # type: ignore
            if isinstance(patch_size, tuple):
                patch_size = patch_size[0]

            decoder_pred_size = (patch_size //
                                 2)**2 * (triplane_decoder.out_chans * 3)
        if unpatchify_out_chans == -1:
            unpatchify_out_chans = triplane_decoder.out_chans * 3

        super().__init__(vit_decoder,
                         triplane_decoder,
                         cls_token,
                         decoder_pred_size=decoder_pred_size,
                         unpatchify_out_chans=unpatchify_out_chans)

        ldm_z_channels = triplane_decoder.out_chans
        # ldm_embed_dim = 16 # https://github.com/CompVis/latent-diffusion/blob/e66308c7f2e64cb581c6d27ab6fbeb846828253b/models/first_stage_models/kl-f16/config.yaml
        ldm_embed_dim = triplane_decoder.out_chans

        self.superresolution.update({
            'conv1':
            nn.Identity(),
            'norm':
            nn.Identity(),
            'after_vit_conv':
            nn.Conv2d(
                int(triplane_decoder.out_chans * 2),
                triplane_decoder.out_chans * 2,  # for vae features
                3,
                padding=1),
            'quant_conv':
            torch.nn.Conv2d(2 * ldm_z_channels, 2 * ldm_embed_dim, 1),
            # 'post_quant_conv':
            # torch.nn.Conv2d(ldm_embed_dim, ldm_z_channels, 1),
            'proj_upsample_224':
            PixelUnshuffleUpsample(
                output_dim=triplane_decoder.out_chans * 3,
                num_feat=int(triplane_decoder.out_chans * 2.5),
                num_out_ch=int(triplane_decoder.out_chans * 2),
                sr_ratio=2),  # 4x SR too heavy to converge. used for diffusion
        })

    def vit_decode_backbone(self, latent, img_size):
        return self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

    def vit_decode_postprocess(self, latent_from_vit, sample_posterior=True):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify(
            latent, p=7)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        latent = self.superresolution['proj_upsample_224'](
            latent, input_skip_connection=False)  # upsample triplane

        posterior = self.vae_encode(latent)

        # ! do VAE here
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        # ret_dict = self.vae_decode(z, c)
        # ret_dict.update({
        # 'posterior': posterior,
        # 'logvar': self.logvar
        # })

        return dict(
            latent_normalized=z,
            cls_token=cls_token,
            latent_from_vit=latent_from_vit,
            posterior=posterior,
        )

    def vit_decode(self, latent, img_size):

        latent_from_vit = self.vit_decode_backbone(latent, img_size)

        return self.vit_decode_postprocess(latent_from_vit)

    def triplane_decode(self, vit_decode_out, c, return_raw_only=False):

        if isinstance(vit_decode_out, dict):
            latent_after_vit, sr_w_code = (vit_decode_out.get(k, None)
                                           for k in ('latent_after_vit',
                                                     'sr_w_code'))

            # latent_after_vit = vit_decode_out[
            #     'latent_normalized']  # vae q(z0|x)

        else:
            latent_after_vit = vit_decode_out
            sr_w_code = None
            vit_decode_out = dict(latent_normalized=latent_after_vit
                                  )  # for later dict update compatability

        # latent_denormalized = self.superresolution['norm'].norm_affine(
        #     latent_normalized)  # type: ignore

        latent_denormalized = self.superresolution['conv2'](
            latent_after_vit)  # post_quant_conv

        # * triplane rendering

        ret_dict = self.triplane_decoder(
            latent_denormalized,
            c,
            ws=sr_w_code,
            return_raw_only=return_raw_only)  # triplane latent -> imgs
        ret_dict.update({
            'latent_denormalized': latent_denormalized,
            **vit_decode_out
        })

        return ret_dict

    # VAE APIs
    def vae_encode(self, h):
        # * smooth convolution before triplane
        h = self.superresolution['after_vit_conv'](h)

        moments = self.superresolution['quant_conv'](h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior


class VAE_V2_noConvSR(VAE_V1):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane,
            cls_token,
            normalize_feat=True,
            #  sr_ratio=2,
            decoder_pred_size=-1,
            unpatchify_out_chans=-1,
            **kwargs) -> None:

        if decoder_pred_size == -1:
            decoder_pred_size = (14 // 1)**2 * (triplane_decoder.out_chans * 2)
        if unpatchify_out_chans == -1:
            unpatchify_out_chans = triplane_decoder.out_chans * 2

        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            # decoder_pred_size=(14 // 2)**2 * (triplane_decoder.out_chans * 2),
            decoder_pred_size=decoder_pred_size,
            unpatchify_out_chans=unpatchify_out_chans)

        ldm_z_channels = ldm_embed_dim = triplane_decoder.out_chans

        self.superresolution.update({
            'proj_upsample_224':
            nn.Identity(),
            'after_vit_conv':
            nn.Identity(),
            # nn.Conv2d(
            #     int(triplane_decoder.out_chans * 2),
            #     triplane_decoder.out_chans * 2,  # for vae features
            #     3,
            #     padding=1),
            'quant_conv':
            torch.nn.Conv2d(2 * ldm_z_channels, 2 * ldm_embed_dim, 1),
            # 'post_quant_conv':
            # torch.nn.Conv2d(ldm_embed_dim, ldm_z_channels, 1),
        })

    def vit_decode_postprocess(self, latent_from_vit, sample_posterior=True):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify(
            latent, p=14)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        assert latent.shape[-1] == 224

        # latent = self.superresolution['proj_upsample_224'](
        #     latent, input_skip_connection=False) # upsample triplane

        posterior = self.vae_encode(latent)

        # ! do VAE here
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        return dict(
            latent_normalized=z,  # could be used for diffusion prior learning 
            latent_after_vit=z,
            cls_token=cls_token,
            latent_from_vit=latent_from_vit,
            posterior=posterior,
        )


class VAE_LDM_V1(VAE_V2_noConvSR):

    def __init__(
        self,
        vit_decoder: VisionTransformer,
        triplane_decoder: Triplane,
        cls_token,
        normalize_feat=True,
        sr_ratio=2,
        decoder_pred_size=-1,
        vae_dit_token_size=16,
        **kwargs,
    ) -> None:
        if decoder_pred_size == -1:
            decoder_pred_size = (14 // 1)**2 * (triplane_decoder.out_chans * 1)
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            unpatchify_out_chans=triplane_decoder.out_chans * 1,
            decoder_pred_size=decoder_pred_size,
            **kwargs,
        )
        """do VAE in the middle of two ViT; 
        use LDM f=4 setting (resolution 64*64*3), though should abide by 56 here. If works, lower the dimention? 
        """

        self.ldm_z_channels = 3  # 56*56*3, hard coded
        self.ldm_embed_dim = 3
        self.vae_p = 4  # resolution = 4 * 16
        self.token_size = vae_dit_token_size  # use dino-v2 dim tradition here
        self.vae_res = self.vae_p * self.token_size

        self.superresolution.update(
            dict(
                # ldm_downsample=nn.Linear(384,
                ldm_downsample=nn.Linear(vit_decoder.embed_dim,
                                         self.vae_p * self.vae_p *
                                         self.ldm_z_channels * 2,
                                         bias=True),
                ldm_upsample=nn.Linear(self.vae_p * self.vae_p *
                                       self.ldm_z_channels * 1,
                                       vit_decoder.embed_dim,
                                       bias=True),  # ? too high dim upsample
                after_vit_conv=nn.Identity(),
                quant_conv=torch.nn.Conv2d(2 * self.ldm_z_channels,
                                           2 * self.ldm_embed_dim, 1),
            ))

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify(
            latent, p=14)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        ret_dict.update(
            dict(
                cls_token=cls_token,
                # latent_from_vit=latent_from_vit,
                latent_after_vit=latent))

        return ret_dict

    def vit_decode(self, latent, img_size, sample_posterior=True):

        # ! first downsample for VAE
        latent = self.superresolution['ldm_downsample'](latent)
        latent = self.unpatchify(
            latent, p=self.vae_p, unpatchify_out_chans=self.ldm_z_channels *
            2)  # B 6 64 64, follow stable diffusion dimentions

        # ! do VAE here
        posterior = self.vae_encode(latent)

        if sample_posterior:
            latent = posterior.sample()
        else:
            latent = posterior.mode()

        ret_dict = dict(
            latent_normalized=latent,
            posterior=posterior,
        )

        # z = z.permute(0, 3,1,2) # B 4 H W

        # ! upsample back to ViT-L spatial latent size, with cls
        latent = latent.reshape(
            shape=(latent.shape[0], latent.shape[1], self.token_size,
                   self.vae_p, self.token_size, self.vae_p))  # B C h p w q
        latent = torch.einsum('nchpwq->nhwpqc', latent)  # B 16 16 p p C
        # latent = latent.flatten(2).transpose(1, 2) # merge hpwp ->
        latent = latent.reshape(latent.shape[0], self.token_size**2,
                                -1)  # B N C
        latent = self.superresolution['ldm_upsample'](latent)

        latent_from_vit = self.vit_decode_backbone(latent, img_size)
        return self.vit_decode_postprocess(latent_from_vit, ret_dict)


class VAE_LDM_V2(VAE_LDM_V1):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, **kwargs)
        """no pretrain
        """

        self.superresolution.update(
            dict(ldm_downsample=nn.Linear(384,
                                          self.vae_p * self.vae_p *
                                          self.ldm_z_channels * 2,
                                          bias=True), ))


class VAE_LDM_V3_noConvQuant(VAE_LDM_V2):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, **kwargs)
        """no pretrin; 
        replace the quant_conv with MLP 6->3
        """

        self.superresolution.update(
            dict(
                # ldm_downsample=nn.Linear(384,
                #                          self.vae_p * self.vae_p *
                #                          self.ldm_z_channels * 2,
                #                          bias=True),
                quant_conv=nn.Identity(),
                quant_mlp=Mlp(2 * self.ldm_z_channels,
                              out_features=2 * self.ldm_embed_dim),
                after_vit_conv=nn.Identity(),
            ))

    def vae_encode(self, h):
        # * smooth convolution before triplane
        # h = self.superresolution['after_vit_conv'](h)
        h = h.permute(0, 2, 3, 1)  # B 64 64 6
        moments = self.superresolution['quant_mlp'](h)
        moments = moments.permute(0, 3, 1, 2)  # B 6 64 64
        posterior = DiagonalGaussianDistribution(moments)
        return posterior


class VAE_LDM_V1_noConvQuant(VAE_LDM_V1):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 decoder_pred_size=-1,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, decoder_pred_size, **kwargs)

        self.superresolution.update(
            dict(
                # ldm_downsample=nn.Linear(384,
                #                          self.vae_p * self.vae_p *
                #                          self.ldm_z_channels * 2,
                #                          bias=True),
                quant_conv=nn.Identity(),
                quant_mlp=Mlp(2 * self.ldm_z_channels,
                              out_features=2 * self.ldm_embed_dim),
                after_vit_conv=nn.Identity(),
            ))

    def vae_encode(self, h):
        # * smooth convolution before triplane
        # h = self.superresolution['after_vit_conv'](h)
        h = h.permute(0, 2, 3, 1)  # B 64 64 6
        moments = self.superresolution['quant_mlp'](h)
        moments = moments.permute(0, 3, 1, 2)  # B 6 64 64
        posterior = DiagonalGaussianDistribution(moments)
        return posterior


class VAE_LDM_V1_noConvQuantNoPT(VAE_LDM_V1_noConvQuant):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 decoder_pred_size=-1,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, decoder_pred_size, **kwargs)

        self.superresolution.update(
            dict(ldm_downsample=nn.Linear(384,
                                          self.vae_p * self.vae_p *
                                          self.ldm_z_channels * 2,
                                          bias=True), ))


class VAE_LDM_V4_vit3D(VAE_LDM_V3_noConvQuant):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            decoder_pred_size=-1,
            #  ldm 3d params
            vae_p=1,
            ldm_z_channels=4,
            ldm_embed_dim=4,
            **kwargs) -> None:

        if decoder_pred_size == -1:
            decoder_pred_size = (14 // 1)**2 * (
                triplane_decoder.out_chans // 3 * 1
            )  # ! no need to triple the dimension in the final MLP

        super().__init__(vit_decoder,
                         triplane_decoder,
                         cls_token,
                         normalize_feat=normalize_feat,
                         sr_ratio=sr_ratio,
                         decoder_pred_size=decoder_pred_size,
                         **kwargs)
        """no pretrin; 
        replace the quant_conv with MLP 6->3
        """

        # assert self.cls_token
        # self.decoder_pred_size = decoder_pred_size

        # self.ldm_z_channels = 4  # 56*56*3, hard coded
        # self.ldm_embed_dim = 4
        # self.vae_p = 1  # for latent 3D, 1*16

        self.ldm_z_channels = ldm_z_channels
        self.ldm_embed_dim = ldm_embed_dim
        self.vae_p = vae_p
        self.plane_n = 3

        self.superresolution.update(
            dict(
                ldm_downsample=nn.Linear(
                    384,
                    # vit_decoder.embed_dim,
                    self.vae_p * self.vae_p * 3 * self.ldm_z_channels *
                    2,  # 48
                    bias=True),
                # quant_conv=nn.Identity(),
                quant_mlp=Mlp(2 * self.ldm_z_channels,
                              out_features=2 * self.ldm_embed_dim),
                after_vit_conv=nn.Identity(),
                ldm_upsample=nn.Linear(self.ldm_z_channels,
                                       vit_decoder.embed_dim,
                                       bias=True),  # ? too high dim upsample
            ))
        # self.vit_decoder.pos_embed = nn.Parameter(
        #     torch.zeros(1, 3 * 16 * 16 + 1, vit_decoder.embed_dim))

        has_token = bool(self.cls_token)
        self.vit_decoder.pos_embed = nn.Parameter(
            torch.zeros(1, self.plane_n * 16 * 16 + has_token,
                        vit_decoder.embed_dim))

    def vae_encode(self, h):
        # * smooth convolution before triplane
        # h = self.superresolution['after_vit_conv'](h)
        # h = h.permute(0, 2, 3, 1)  # B 64 64 6
        moments = self.superresolution['quant_mlp'](
            h)  # B 3 L self.ldm_z_channels

        # moments = moments.permute(0, 4, 1, 2,
        #                           3)  # B 4 3 16 16, for vae sampling, for unpachify3D

        # moments: B L 3 C
        moments = moments.permute(0, 3, 1, 2)  # B 3 L C -> B C 3 L

        posterior = DiagonalGaussianDistribution(
            moments, soft_clamp=self.reparameterization_soft_clamp)
        return posterior

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.vit_decoder.pos_embed.shape[1] - 1  # type: ignore
        # if npatch == N and w == h:
        # assert npatch == N and w == h
        return self.vit_decoder.pos_embed

        # pos_embed = self.vit_decoder.pos_embed.float()
        # return pos_embed
        class_pos_embed = pos_embed[:, 0]  # type: ignore
        patch_pos_embed = pos_embed[:, 1:]  # type: ignore
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        # patch_pos_embed = nn.functional.interpolate(
        #     patch_pos_embed.reshape(1, 3, int(math.sqrt(N//3)), int(math.sqrt(N//3)), dim).permute(0, 4, 1, 2, 3),
        #     scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        #     mode="bicubic",
        # ) # ! no interpolation needed, just add, since the resolution shall match

        # assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed),
                         dim=1).to(previous_dtype)

    def unpatchify3D(self, x, p, unpatchify_out_chans, plane_n=3):
        """
        x: (N, L, patch_size**2 * self.out_chans)
        return: 3D latents
        """

        if self.cls_token:  # TODO, how to better use cls token
            x = x[:, 1:]

        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, plane_n,
                             unpatchify_out_chans))

        x = torch.einsum(
            'nhwpqdc->ndhpwqc', x
        )  # nplanes, C little endian tradiition, as defined in the renderer.py

        latents3D = x.reshape(shape=(x.shape[0], plane_n, h * p, h * p,
                                     unpatchify_out_chans))
        return latents3D

    def vae_reparameterization(self, latent, sample_posterior):
        """input: latent from ViT encoder
        """
        # ! first downsample for VAE
        latents3D = self.superresolution['ldm_downsample'](latent)  # B L 24

        if self.vae_p > 1:
            latents3D = self.unpatchify3D(
                latents3D,
                p=self.vae_p,
                unpatchify_out_chans=self.ldm_z_channels *
                2)  # B 3 H W unpatchify_out_chans, H=W=16 now
            latents3D = latents3D.reshape(
                latents3D.shape[0], self.plane_n, -1, latents3D.shape[-1]
            )  # B 3 H*W C (H=self.vae_p*self.token_size)
        else:
            latents3D = latents3D.reshape(latents3D.shape[0],
                                          latents3D.shape[1], 3,
                                          2 * self.ldm_z_channels)  # B L 3 C
            latents3D = latents3D.permute(0, 2, 1, 3)  # B 3 L C

        # ! maintain the cls token here
        # latent3D = latent.reshape()

        # ! do VAE here
        posterior = self.vae_encode(latents3D)  # B self.ldm_z_channels 3 L

        if sample_posterior:
            latent = posterior.sample()
        else:
            latent = posterior.mode()  # B C 3 L

        log_q = posterior.log_p(latent)  # same shape as latent

        # latent = latent.permute(0, 2, 3, 4,
        #                         1)  # C to the last dim, B 3 16 16 4, for unpachify 3D

        # ! for LSGM KL code
        latent_normalized_2Ddiffusion = latent.reshape(
            latent.shape[0], -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)  # B, 3*4, 16 16
        log_q_2Ddiffusion = log_q.reshape(
            latent.shape[0], -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)  # B, 3*4, 16 16
        latent = latent.permute(0, 2, 3, 1)  # B C 3 L -> B 3 L C

        latent = latent.reshape(latent.shape[0], -1,
                                latent.shape[-1])  # B 3*L C

        ret_dict = dict(
            normal_entropy=posterior.normal_entropy(),
            latent_normalized=latent,
            latent_normalized_2Ddiffusion=latent_normalized_2Ddiffusion,  # 
            log_q_2Ddiffusion=log_q_2Ddiffusion,
            log_q=log_q,
            posterior=posterior,
            latent_name=
            'latent_normalized'  # for which latent to decode; could be modified externally
        )

        return ret_dict

    def vit_decode(self, latent, img_size, sample_posterior=True):

        ret_dict = self.vae_reparameterization(latent, sample_posterior)
        # latent = ret_dict['latent_normalized']
        assert isinstance(ret_dict, dict)
        latent_name = ret_dict['latent_name']
        assert isinstance(latent_name, str)
        latent = ret_dict[latent_name]

        if latent_name == 'latent_normalized_2Ddiffusion':  # denoised from diffusion
            # latent_normalized_2Ddiffusion = latent.reshape(
            #     latent.shape[0], -1, self.token_size * self.vae_p,
            #     self.token_size * self.vae_p)  # B, 3*4, 16 16
            latent = latent.reshape(latent.shape[0], latent.shape[1] // 3, 3,
                                    -1)  # B C 3 L
            latent = latent.permute(0, 2, 3, 1)  # B C 3 L -> B 3 L C

            latent = latent.reshape(latent.shape[0], -1,
                                    latent.shape[-1])  # B 3*L C

        latent = self.vit_decode_backbone(latent, img_size)
        return self.vit_decode_postprocess(latent, ret_dict)

    def vit_decode_backbone(self, latent, img_size):
        # assert x.ndim == 3  # N L C
        if isinstance(latent, dict):
            if 'latent_normalized' not in latent:
                latent = latent[
                    'latent_normalized_2Ddiffusion']  # B, C*3, H, W
            else:
                latent = latent[
                    'latent_normalized']  # TODO, just for compatability now

        if latent.ndim != 3:  # B 3*4 16 16
            latent = latent.reshape(latent.shape[0], latent.shape[1] // 3, 3,
                                    (self.vae_p * self.token_size)**2).permute(
                                        0, 2, 3, 1)  # B C 3 L => B 3 L C
            latent = latent.reshape(latent.shape[0], -1,
                                    latent.shape[-1])  # B 3*L C

        assert latent.shape == (
            # latent.shape[0], 3 * (self.token_size**2),
            latent.shape[0],
            3 * ((self.vae_p * self.token_size)**2),
            self.ldm_z_channels), f'latent.shape: {latent.shape}'

        latent = self.superresolution['ldm_upsample'](latent)

        return super().vit_decode_backbone(
            latent, img_size)  # torch.Size([8, 3072, 768])

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify_triplane(
            latent, p=14, unpatchify_out_chans=self.unpatchify_out_chans //
            3)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        sr_w_code = self.w_avg
        assert sr_w_code is not None
        ret_dict.update(
            dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
                latent.shape[0], 0), ))  # type: ignore

        return ret_dict

    def forward_vit_decoder(self, x, img_size=None):
        # latent: (N, L, C) from DINO/CLIP ViT encoder

        # * also dino ViT
        # add positional encoding to each token
        if img_size is None:
            img_size = self.img_size

        # if self.cls_token:
        # st()
        x = x + self.interpolate_pos_encoding(x, img_size,
                                              img_size)[:, :]  # B, L, C
        # else:
        #     x = x + self.interpolate_pos_encoding(x, img_size,
        #                                           img_size)[:, :]  # B, L, C

        for blk in self.vit_decoder.blocks:
            x = blk(x)
        x = self.vit_decoder.norm(x)

        return x

    def unpatchify_triplane(self, x, p=None, unpatchify_out_chans=None):
        """
        x: (N, L, patch_size**2 * self.out_chans)
        imgs: (N, self.out_chans, H, W)
        """
        if unpatchify_out_chans is None:
            unpatchify_out_chans = self.unpatchify_out_chans // 3
        # p = self.vit_decoder.patch_size
        if self.cls_token:  # TODO, how to better use cls token
            x = x[:, 1:]

        if p is None:  # assign upsample patch size
            p = self.patch_size
        h = w = int((x.shape[1] // 3)**.5)
        assert h * w * 3 == x.shape[1]

        x = x.reshape(shape=(x.shape[0], 3, h, w, p, p, unpatchify_out_chans))
        x = torch.einsum('ndhwpqc->ndchpwq',
                         x)  # nplanes, C order in the renderer.py
        triplanes = x.reshape(shape=(x.shape[0], unpatchify_out_chans * 3,
                                     h * p, h * p))
        return triplanes


class VAE_LDM_V4_vit3D_fusion(VAE_LDM_V4_vit3D):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=4,
            #  cross_attention_blk=CLSCrossAttentionBlock,
            fusion_blk=TriplaneFusionBlock,
            fusion_blk_start=0,  # appy fusion blk start with?
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, **kwargs)
        """
        1. separate self attention within each triplane
        2. test cross attentio nfusion
        """

        # TODO, register modules in the optimizers, how

        token_size = 224 // self.patch_size
        logger.log('token_size: {}', token_size)

        # st()
        self.vit_decoder.pos_embed = nn.Parameter(
            torch.zeros(1, self.plane_n * (token_size**2 + self.cls_token),
                        vit_decoder.embed_dim))

        # nh = self.vit_decoder.num_heads

        self.fusion_blk_start = fusion_blk_start
        self.create_fusion_blks(fusion_blk_depth, use_fusion_blk, fusion_blk)

        # self.vit_decoder.cls_token = self.vit_decoder.cls_token.clone().repeat_interleave(3, dim=0) # each plane has a separate cls token
        # translate

    # !
    def create_fusion_blks(self, fusion_blk_depth, use_fusion_blk, fusion_blk):

        vit_decoder_blks = self.vit_decoder.blocks
        assert len(vit_decoder_blks) == 12, 'ViT-B by default'

        nh = self.vit_decoder.blocks[0].attn.num_heads
        dim = self.vit_decoder.embed_dim

        fusion_blk_start = self.fusion_blk_start
        triplane_fusion_vit_blks = nn.ModuleList()

        if fusion_blk_start != 0:
            for i in range(0, fusion_blk_start):
                triplane_fusion_vit_blks.append(
                    vit_decoder_blks[i])  # append all vit blocks in the front

        for i in range(fusion_blk_start, len(vit_decoder_blks),
                       fusion_blk_depth):
            vit_blks_group = vit_decoder_blks[i:i +
                                              fusion_blk_depth]  # moduleList
            triplane_fusion_vit_blks.append(
                # TriplaneFusionBlockv2(vit_blks_group, nh, dim, use_fusion_blk))
                fusion_blk(vit_blks_group, nh, dim, use_fusion_blk))

        self.vit_decoder.blocks = triplane_fusion_vit_blks

    def forward_vit_decoder(self, x, img_size=None):
        # latent: (N, L, C) from DINO/CLIP ViT encoder

        # * also dino ViT
        # add positional encoding to each token
        if img_size is None:
            img_size = self.img_size

        # if self.cls_token:
        x = x + self.interpolate_pos_encoding(x, img_size,
                                              img_size)[:, :]  # B, L, C
        # else:
        #     x = x + self.interpolate_pos_encoding(x, img_size,
        #                                           img_size)[:, 1:]  # B, L, C

        B, L, C = x.shape  # has [cls] token in N
        x = x.view(B, 3, L // 3, C)

        # for blk in self.vit_decoder.blocks:
        for blk in self.vit_decoder.blocks[self.fusion_blk_start:]:
            x = blk(x)  # B 3 N C
        x = self.vit_decoder.norm(x)

        # post process shape
        x = x.view(B, L, C)
        return x

    def unpatchify_triplane(self, x, p=None, unpatchify_out_chans=None):
        """separate triplane version; x shape: B (3*257) 768
        """
        if unpatchify_out_chans is None:
            unpatchify_out_chans = self.unpatchify_out_chans // 3
        # p = self.vit_decoder.patch_size

        B, L, C = x.shape
        x = x.reshape(B, 3, L // 3, C)

        if self.cls_token:  # TODO, how to better use cls token
            x = x[:, :, 1:]  # B 3 256 C

        if p is None:  # assign upsample patch size
            p = self.patch_size

        h = w = int((x.shape[2])**.5)
        assert h * w == x.shape[2]

        x = x.reshape(shape=(B, 3, h, w, p, p, unpatchify_out_chans))
        x = torch.einsum('ndhwpqc->ndchpwq',
                         x)  # nplanes, C order in the renderer.py
        x = x.reshape(shape=(B, 3 * unpatchify_out_chans, h * p, h * p))
        return x


class VAE_LDM_V4_vit3D_v2_ablanofusion(VAE_LDM_V4_vit3D_fusion):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 **kwargs) -> None:
        super().__init__(vit_decoder,
                         triplane_decoder,
                         cls_token,
                         normalize_feat,
                         sr_ratio,
                         use_fusion_blk=False,
                         **kwargs)
        """no cross attention fusion, jsut split the self attention channels.
        """
        # TODO, define the forward function with split attention tokens


# class VAE_LDM_V4_vit3D(VAE_LDM_V3_noConvQuant):


class VAE_LDM_V4_vit3D_v3_conv3D(VAE_LDM_V4_vit3D_fusion):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=4,
            fusion_blk=TriplaneFusionBlockv2,  # type: ignore
            **kwargs) -> None:
        super().__init__(vit_decoder,
                         triplane_decoder,
                         cls_token,
                         normalize_feat,
                         sr_ratio,
                         fusion_blk=fusion_blk,
                         **kwargs)
        """no cross attention fusion, jsut split the self attention channels.
        """


class VAE_LDM_V4_vit3D_v3_conv3D_depth2(VAE_LDM_V4_vit3D_fusion):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv2,  # type: ignore
            **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            fusion_blk=fusion_blk,  # type: ignore
            use_fusion_blk=use_fusion_blk,
            fusion_blk_depth=fusion_blk_depth,
            **kwargs)
        """no cross attention fusion, jsut split the self attention channels.
        """


class VAE_LDM_V4_vit3D_v3_conv3D_depth1(VAE_LDM_V4_vit3D_fusion):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk=TriplaneFusionBlockv2,  # type: ignore
            fusion_blk_depth=1,
            **kwargs) -> None:
        super().__init__(vit_decoder,
                         triplane_decoder,
                         cls_token,
                         normalize_feat,
                         sr_ratio,
                         use_fusion_blk=use_fusion_blk,
                         fusion_blk=fusion_blk,
                         fusion_blk_depth=fusion_blk_depth,
                         **kwargs)
        """no cross attention fusion, jsut split the self attention channels.
        """


class VAE_LDM_V4_vit3D_v3_conv3D_depth3(VAE_LDM_V4_vit3D_fusion):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=3,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            fusion_blk=TriplaneFusionBlockv2,  # type: ignore
            use_fusion_blk=use_fusion_blk,
            fusion_blk_depth=fusion_blk_depth,
            **kwargs)
        """no cross attention fusion, jsut split the self attention channels.
        """


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha(VAE_LDM_V4_vit3D_fusion):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 fusion_blk=TriplaneFusionBlockv3,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            fusion_blk=fusion_blk,  # type: ignore
            use_fusion_blk=use_fusion_blk,
            fusion_blk_depth=fusion_blk_depth,
            **kwargs)
        """no cross attention fusion, jsut split the self attention channels.
        """


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_trunc_normal(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, **kwargs)
        """init pos_embed != 0, also learnable.
            """
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.vit_decoder.pos_embed, std=0.02)
        logger.log('init pos_embed with trunc_normal_')

    #     nn.init.normal_(self.cls_token, std=1e-6)
    #     named_apply(init_weights_vit_timm, self)


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, **kwargs)
        """init pos_embed != 0, follow DiT sincos
            """
        self.init_weights()

    def init_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        p = self.token_size
        D = self.vit_decoder.pos_embed.shape[-1]
        grid_size = (3 * p, p)
        pos_embed = get_2d_sincos_pos_embed(D,
                                            grid_size).reshape(3 * p * p,
                                                               D)  # H*W, D
        self.vit_decoder.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))
        logger.log('init pos_embed with sincos')


# class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_split_vitB(
#         VAE_LDM_V4_vit3D_fusion):

#     def __init__(
#         self,
#         vit_decoder: VisionTransformer,
#         triplane_decoder: Triplane,
#         cls_token,
#         normalize_feat=True,
#         sr_ratio=2,
#         use_fusion_blk=True,
#         fusion_blk_depth=1,  # for the last 6 vitb blks, all replace with fusion blk
#         **kwargs) -> None:
#         super().__init__(
#             vit_decoder,
#             triplane_decoder,
#             cls_token,
#             normalize_feat,
#             sr_ratio,
#             fusion_blk=TriplaneFusionBlockv3,  # type: ignore
#             use_fusion_blk=use_fusion_blk,
#             fusion_blk_depth=fusion_blk_depth,
#             fusion_blk_start=6,
#             **kwargs)
#         """split fusion blocks into 6+6 for ViT-B
#         """

#         self.superresolution.update(
#             dict(
#                 ldm_downsample=nn.Linear(
#                     vit_decoder.embed_dim,
#                     self.vae_p * self.vae_p * 3 * self.ldm_z_channels *
#                     2,  # 48
#                     bias=True), ))

#     def vit_decode(self, latent, img_size, sample_posterior=True):

#         # * add pre-3D vit blocks
#         for blk in self.vit_decoder.blocks[:self.fusion_blk_start]:
#             latent = blk(latent)  # B 3 N C

#         return super().vit_decode(latent, img_size, sample_posterior)


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos):
    """add long skip connection within ViT to improve the reconstruction performance, decoder only for now
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, **kwargs)
        """init pos_embed != 0, follow DiT sincos
            """
        self.init_weights()
        self.reparameterization_soft_clamp = True  # some instability in training VAE

        # create skip linear
        logger.log(
            f'length of vit_decoder.blocks: {len(self.vit_decoder.blocks)}')
        # for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks)//2+1:]:
        for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks) // 2:]:
            blk.skip_linear = nn.Linear(2 * self.vit_decoder.embed_dim,
                                        self.vit_decoder.embed_dim)

            # trunc_normal_(blk.skip_linear.weight, std=.02)
            nn.init.constant_(blk.skip_linear.weight, 0)
            if isinstance(blk.skip_linear,
                          nn.Linear) and blk.skip_linear.bias is not None:
                nn.init.constant_(blk.skip_linear.bias, 0)

    def forward_vit_decoder(self, x, img_size=None):
        # latent: (N, L, C) from DINO/CLIP ViT encoder

        # * also dino ViT
        # add positional encoding to each token
        if img_size is None:
            img_size = self.img_size

        # if self.cls_token:
        x = x + self.interpolate_pos_encoding(x, img_size,
                                              img_size)[:, :]  # B, L, C

        B, L, C = x.shape  # has [cls] token in N
        x = x.view(B, 3, L // 3, C)

        skips = [x]
        assert self.fusion_blk_start == 0

        # in blks
        for blk in self.vit_decoder.blocks[0:len(self.vit_decoder.blocks) //
                                           2 - 1]:
            x = blk(x)  # B 3 N C
            skips.append(x)

        # mid blks
        # for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks)//2-1:len(self.vit_decoder.blocks)//2+1]:
        for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks) // 2 -
                                           1:len(self.vit_decoder.blocks) //
                                           2]:
            x = blk(x)  # B 3 N C

        # out blks
        for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks) // 2:]:
            x = x + blk.skip_linear(torch.cat([x, skips.pop()],
                                              dim=-1))  # long skip connections
            x = blk(x)  # B 3 N C

        x = self.vit_decoder.norm(x)

        # post process shape
        x = x.view(B, L, C)
        return x


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_after_vit_conv_abc(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, **kwargs)
        """add a conv layer after ViT decoder, since raw ViT feature may not be suitable for direct reconstruction
        if works, switch to larger conv models, e.g., 
        1. conv with triplane roll out
        2. res blk roll out? 3*3 or 1*3*1
        """

        self.superresolution.update(
            dict(before_rendering_conv=nn.Conv2d(triplane_decoder.out_chans,
                                                 triplane_decoder.out_chans,
                                                 3,
                                                 padding=1), ))


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_after_vit_conv(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_after_vit_conv_abc
):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, **kwargs)
        """add a conv layer after ViT decoder, since raw ViT feature may not be suitable for direct reconstruction
        if works, switch to larger conv models, e.g., 
        1. conv with triplane roll out
        2. res blk roll out? 3*3 or 1*3*1
        """

    def unpatchify_triplane(self, x, p=None, unpatchify_out_chans=None):
        """
        x: (N, L, patch_size**2 * self.out_chans)
        imgs: (N, self.out_chans, H, W)
        """
        triplanes = super().unpatchify_triplane(
            x, p, unpatchify_out_chans=unpatchify_out_chans
        )  # shape=(B, 3 * unpatchify_out_chans, h * p, h * p)
        triplanes = self.superresolution['before_rendering_conv'](triplanes)
        return triplanes


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_after_vit_conv_with_resconnection(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_after_vit_conv_abc
):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, **kwargs)

        nn.init.constant_(self.superresolution['before_rendering_conv'].weight,
                          0)
        nn.init.constant_(self.superresolution['before_rendering_conv'].bias,
                          0)

    def unpatchify_triplane(self, x, p=None, unpatchify_out_chans=None):
        """
        x: (N, L, patch_size**2 * self.out_chans)
        imgs: (N, self.out_chans, H, W)
        """
        triplanes = super().unpatchify_triplane(
            x, p, unpatchify_out_chans=unpatchify_out_chans
        )  # shape=(B, 3 * unpatchify_out_chans, h * p, h * p)
        triplanes = self.superresolution['before_rendering_conv'](
            triplanes) + triplanes
        return triplanes


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_after_vit_conv_with_resconnection_blk(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_after_vit_conv
):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, **kwargs)

        self.superresolution.update(
            dict(before_rendering_conv=ResidualBlock(
                triplane_decoder.out_chans,
                triplane_decoder.out_chans,
            )))


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit):
    """ 7*2 SR
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            decoder_pred_size=(7 // 1)**2 *
            (triplane_decoder.out_chans // 3 *
             1),  # ! no need to triple the dimension in the final MLP
            **kwargs)
        self.superresolution.update(
            dict(conv_sr=RodinRollOutConv3DSR2X(triplane_decoder.out_chans)))

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify_triplane(
            latent, p=7, unpatchify_out_chans=self.unpatchify_out_chans //
            3)  # spatial_vit_latent, B, C, H, W (B, 96, 112, 112)

        # 2X SR with Rodin Conv 3D
        latent = self.superresolution['conv_sr'](latent)  # still B 3C H W

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        sr_w_code = self.w_avg
        assert sr_w_code is not None
        ret_dict.update(
            dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
                latent.shape[0], 0), ))  # type: ignore

        return ret_dict


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            decoder_pred_size=(4 // 1)**2 * int(
                triplane_decoder.out_chans // 3 * 1.5
            ),  # 144 -> 96 after x4 RodinSR ,  # ! no need to triple the dimension in the final MLP
            **kwargs)
        # self.superresolution.update(
        #     dict(conv_sr=RodinConv3DPixelUnshuffleUpsample(
        #         int(1.5 *
        #             triplane_decoder.out_chans), num_out_ch=triplane_decoder.out_chans))) # not converging

        self.superresolution.update(
            dict(conv_sr=RodinRollOutConv3DSR_FlexibleChannels(
                int(1.5 * triplane_decoder.out_chans),
                num_out_ch=triplane_decoder.out_chans)))  #

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify_triplane(
            latent,
            p=4,
            unpatchify_out_chans=int(
                self.unpatchify_out_chans // 3 *
                1.5))  # spatial_vit_latent, B, C, H, W (B, 96, 112, 112)

        # 2X SR with Rodin Conv 3D
        latent = self.superresolution['conv_sr'](latent)  # still B 3C H W

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        sr_w_code = self.w_avg
        assert sr_w_code is not None
        ret_dict.update(
            dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
                latent.shape[0], 0), ))  # type: ignore

        return ret_dict


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4lite(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit):
    """ 4*4 SR
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            decoder_pred_size=(4 // 1)**2 *
            (triplane_decoder.out_chans // 3 *
             1),  # ! no need to triple the dimension in the final MLP
            **kwargs)
        self.superresolution.update(
            dict(conv_sr=RodinRollOutConv3DSR4X_lite(
                triplane_decoder.out_chans)))

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify_triplane(
            latent, p=4, unpatchify_out_chans=self.unpatchify_out_chans //
            3)  # spatial_vit_latent, B, C, H, W (B, 96, 112, 112)

        # 2X SR with Rodin Conv 3D
        latent = self.superresolution['conv_sr'](latent)  # still B 3C H W

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        sr_w_code = self.w_avg
        assert sr_w_code is not None
        ret_dict.update(
            dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
                latent.shape[0], 0), ))  # type: ignore

        return ret_dict


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4lite_v2(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4lite
):
    """ 4*4 SR, 32.0 GB
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            # decoder_pred_size=(4 // 1)**2 *
            # (triplane_decoder.out_chans // 3 *
            #  1),  # ! no need to triple the dimension in the final MLP
            **kwargs)

        self.superresolution.update(
            dict(conv_sr=RodinConv3DPixelUnshuffleUpsample_improvedVersion(
                triplane_decoder.out_chans)))  # 64 -> 256


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4lite_v3(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4lite
):
    """ 4*4 SR, 32.0 GB
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            # decoder_pred_size=(4 // 1)**2 *
            # (triplane_decoder.out_chans // 3 *
            #  1),  # ! no need to triple the dimension in the final MLP
            **kwargs)

        self.superresolution.update(
            dict(conv_sr=RodinConv3DPixelUnshuffleUpsample_improvedVersion2(
                triplane_decoder.out_chans,
                triplane_decoder.out_chans)))  # 64 -> 256


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_both(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit):
    """ 4*4 SR
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 **kwargs) -> None:
        super().__init__(vit_decoder,
                         triplane_decoder,
                         cls_token,
                         normalize_feat,
                         sr_ratio,
                         use_fusion_blk,
                         fusion_blk_depth,
                         decoder_pred_size=(1 // 1)**2 *
                         (triplane_decoder.out_chans // 3 * 1),
                         **kwargs)
        self.superresolution.update(
            dict(conv_sr_16_64=RodinRollOutConv3DSR4X_lite(
                triplane_decoder.out_chans, input_resolutiopn=64),
                 conv_sr_64_256=RodinRollOutConv3DSR4X_lite(
                     triplane_decoder.out_chans, input_resolutiopn=256)))

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify_triplane(
            latent,
            p=1,
            unpatchify_out_chans=1 * self.unpatchify_out_chans //
            3)  # spatial_vit_latent, B, C, H, W (B, 96*1, 16, 16)

        # 4X SR with Rodin Conv 3D
        latent = self.superresolution['conv_sr_16_64'](
            latent)  # still B 3C 64 64
        latent = self.superresolution['conv_sr_64_256'](
            latent)  # still B 3C 256 256

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        sr_w_code = self.w_avg
        assert sr_w_code is not None
        ret_dict.update(
            dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
                latent.shape[0], 0), ))  # type: ignore

        return ret_dict


# class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_channeldown(
class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_channeldown_big(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit):
    """ 4*4 SR
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            decoder_pred_size=(4 // 1)**2 *
            #  int(triplane_decoder.out_chans // 3 * 1.5),
            int(triplane_decoder.out_chans // 3 * 2),
            **kwargs)
        self.superresolution.update(
            dict(conv_sr=RodinRollOutConv3DSR_FlexibleChannels(
                int(triplane_decoder.out_chans *
                    2), int(triplane_decoder.out_chans * 1))))
        # 1.5), int(triplane_decoder.out_chans * 1))))

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify_triplane(
            latent,
            p=4,
            unpatchify_out_chans=int(
                # 1.5 * self.unpatchify_out_chans //
                2 * self.unpatchify_out_chans //
                3))  # spatial_vit_latent, B, C, H, W (B, 96*2, 16, 16)

        # 4X SR with Rodin Conv 3D
        latent = self.superresolution['conv_sr'](latent)  # still B 3C H W

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        sr_w_code = self.w_avg
        assert sr_w_code is not None
        ret_dict.update(
            dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
                latent.shape[0], 0), ))  # type: ignore

        return ret_dict


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_channeldown_abla(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit):
    """ 4*4 SR, with no channel down, but use flexible conv. ablate why better than lite version?
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            decoder_pred_size=(4 // 1)**2 *
            #  int(triplane_decoder.out_chans // 3 * 1.5),
            int(triplane_decoder.out_chans // 3 * 1),
            **kwargs)
        self.superresolution.update(
            dict(conv_sr=RodinRollOutConv3DSR_FlexibleChannels(
                int(triplane_decoder.out_chans *
                    1), int(triplane_decoder.out_chans * 1))))
        # 1.5), int(triplane_decoder.out_chans * 1))))

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify_triplane(
            latent,
            p=4,
            unpatchify_out_chans=int(
                # 1.5 * self.unpatchify_out_chans //
                # 2 * self.unpatchify_out_chans //
                1 * self.unpatchify_out_chans //
                3))  # spatial_vit_latent, B, C, H, W (B, 96*2, 16, 16)

        # 4X SR with Rodin Conv 3D
        latent = self.superresolution['conv_sr'](latent)  # still B 3C H W

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        sr_w_code = self.w_avg
        assert sr_w_code is not None
        ret_dict.update(
            dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
                latent.shape[0], 0), ))  # type: ignore

        return ret_dict


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_2x3x2(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit):
    """ 4*4 SR
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            decoder_pred_size=(2 // 1)**2 *
            (triplane_decoder.out_chans // 3 * 2),  # 768 -> 192 -> 144 -> 96; 
            **kwargs)
        self.superresolution.update(
            dict(
                conv_sr_32_96=RodinRollOutConv3DSR_FlexibleChannels(
                    triplane_decoder.out_chans * 2,
                    int(triplane_decoder.out_chans * 1.5),
                    input_resolutiopn=96),
                conv_sr_96_192=RodinRollOutConv3DSR_FlexibleChannels(
                    int(triplane_decoder.out_chans * 1.5),
                    triplane_decoder.out_chans * 1,
                    input_resolutiopn=192)))  # go with 192 size triplane here.

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify_triplane(
            latent,
            p=2,
            unpatchify_out_chans=2 * self.unpatchify_out_chans //
            3)  # spatial_vit_latent, B, C, H, W (B, 96*2, 32, 32)

        # 3X SR with Rodin Conv 3D
        latent = self.superresolution['conv_sr_32_96'](
            latent)  # still B 1.5*3*C 64 64

        # 2X SR with Rodin Conv 3D
        latent = self.superresolution['conv_sr_96_192'](
            latent)  # still B 1*3*C 192 192

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        sr_w_code = self.w_avg
        assert sr_w_code is not None
        ret_dict.update(
            dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
                latent.shape[0], 0), ))  # type: ignore

        return ret_dict


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_3x2x2(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit):
    """ 4*4 SR
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            decoder_pred_size=(3 // 1)**2 *
            (triplane_decoder.out_chans // 3 * 2),  # 768 -> 192 -> 144 -> 96; 
            **kwargs)
        self.superresolution.update(
            dict(
                conv_sr_48_96=RodinRollOutConv3DSR_FlexibleChannels(
                    triplane_decoder.out_chans * 2,
                    int(triplane_decoder.out_chans * 1.5),
                    input_resolutiopn=96),
                conv_sr_96_192=RodinRollOutConv3DSR_FlexibleChannels(
                    int(triplane_decoder.out_chans * 1.5),
                    triplane_decoder.out_chans * 1,
                    input_resolutiopn=192)))  # go with 192 size triplane here.

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify_triplane(
            latent,
            p=3,
            unpatchify_out_chans=2 * self.unpatchify_out_chans //
            3)  # spatial_vit_latent, B, C, H, W (B, 96*2, 32, 32)

        # 3X SR with Rodin Conv 3D
        latent = self.superresolution['conv_sr_48_96'](
            latent)  # still B 1.5*3*C 64 64

        # 2X SR with Rodin Conv 3D
        latent = self.superresolution['conv_sr_96_192'](
            latent)  # still B 1*3*C 192 192

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        sr_w_code = self.w_avg
        assert sr_w_code is not None
        ret_dict.update(
            dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
                latent.shape[0], 0), ))  # type: ignore

        return ret_dict


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x2x2(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit):
    """ 4*4 SR
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            decoder_pred_size=(3 // 1)**2 *
            (triplane_decoder.out_chans // 3 * 2),  # 768 -> 192 -> 144 -> 96; 
            **kwargs)
        self.superresolution.update(
            dict(
                conv_sr_48_96=RodinRollOutConv3DSR_FlexibleChannels(
                    triplane_decoder.out_chans * 2,
                    int(triplane_decoder.out_chans * 1.5),
                    input_resolutiopn=96),
                conv_sr_96_192=RodinRollOutConv3DSR_FlexibleChannels(
                    int(triplane_decoder.out_chans * 1.5),
                    triplane_decoder.out_chans * 1,
                    input_resolutiopn=192)))  # go with 192 size triplane here.

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify_triplane(
            latent,
            p=3,
            unpatchify_out_chans=2 * self.unpatchify_out_chans //
            3)  # spatial_vit_latent, B, C, H, W (B, 96*2, 32, 32)

        # 3X SR with Rodin Conv 3D
        latent = self.superresolution['conv_sr_48_96'](
            latent)  # still B 1.5*3*C 64 64

        # 2X SR with Rodin Conv 3D
        latent = self.superresolution['conv_sr_96_192'](
            latent)  # still B 1*3*C 192 192

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        sr_w_code = self.w_avg
        assert sr_w_code is not None
        ret_dict.update(
            dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
                latent.shape[0], 0), ))  # type: ignore

        return ret_dict


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_channeldown(
        # class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_channeldown_big(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit):
    """ 4*4 SR
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            decoder_pred_size=(4 // 1)**2 *
            int(triplane_decoder.out_chans // 3 * 1.5),
            #  int(triplane_decoder.out_chans // 3 * 2),
            **kwargs)
        self.superresolution.update(
            dict(conv_sr=RodinRollOutConv3DSR4X(  # ! old worked version
                int(triplane_decoder.out_chans *
                    # 2), int(triplane_decoder.out_chans * 1))))
                    1.5), )))
        # int(triplane_decoder.out_chans * 1))))

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify_triplane(
            latent,
            p=4,
            unpatchify_out_chans=int(
                1.5 * self.unpatchify_out_chans //
                # 2 * self.unpatchify_out_chans //
                3))  # spatial_vit_latent, B, C, H, W (B, 96*2, 16, 16)

        # 4X SR with Rodin Conv 3D
        latent = self.superresolution['conv_sr'](latent)  # still B 3C H W

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        sr_w_code = self.w_avg
        assert sr_w_code is not None
        ret_dict.update(
            dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
                latent.shape[0], 0), ))  # type: ignore

        return ret_dict


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit):
    """ 4*4 SR
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 channel_multiplier=1.5,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            decoder_pred_size=(4 // 1)**2 *
            int(triplane_decoder.out_chans // 3 * channel_multiplier),
            #  int(triplane_decoder.out_chans // 3 * 2),
            **kwargs)
        self.channel_multiplier = channel_multiplier
        self.superresolution.update(
            dict(conv_sr=RodinConv3D4X_lite_mlp_as_residual(
                int(triplane_decoder.out_chans *
                    channel_multiplier), int(triplane_decoder.out_chans * 1))))

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit
        )  # pred_vit_latent -> patch or original size; B 768 384

        latent = self.unpatchify_triplane(
            latent,
            p=4,
            unpatchify_out_chans=int(
                self.channel_multiplier * self.unpatchify_out_chans //
                3))  # spatial_vit_latent, B, C, H, W (B, 96*2, 16, 16)

        # 4X SR with Rodin Conv 3D
        latent = self.superresolution['conv_sr'](latent)  # still B 3C H W

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        sr_w_code = self.w_avg
        assert sr_w_code is not None
        ret_dict.update(
            dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
                latent_from_vit.shape[0], 0), ))  # type: ignore

        return ret_dict


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_2XC(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle
):
    """ 4*4 SR with 2X channels
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 channel_multiplier=2,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            channel_multiplier=channel_multiplier,

            #  decoder_pred_size=(4 // 1)**2 *
            #  int(triplane_decoder.out_chans // 3 * channel_multiplier),
            **kwargs)


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle
):
    """ 4*4 SR with 2X channels
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            channel_multiplier=channel_multiplier,

            #  decoder_pred_size=(4 // 1)**2 *
            #  int(triplane_decoder.out_chans // 3 * channel_multiplier),
            **kwargs)


# ! adopted by the current baselines.
class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC_final(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle
):
    """ 4*4 SR with 2X channels
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder,
                         triplane_decoder,
                         cls_token,
                         normalize_feat,
                         sr_ratio,
                         use_fusion_blk,
                         fusion_blk_depth,
                         channel_multiplier=channel_multiplier,
                         **kwargs)

        for k in [
                'conv1', 'conv2', 'norm', 'quant_conv', 'proj_upsample_224',
                'after_vit_conv'
        ]:
            del self.superresolution[k]  # lint unused modules
        del self.IN

    def triplane_decode(self,
                        vit_decode_out,
                        c,
                        return_raw_only=False,
                        **kwargs):

        if isinstance(vit_decode_out, dict):
            latent_after_vit, sr_w_code = (vit_decode_out.get(k, None)
                                           for k in ('latent_after_vit',
                                                     'sr_w_code'))

        else:
            latent_after_vit = vit_decode_out
            sr_w_code = None
            vit_decode_out = dict(latent_normalized=latent_after_vit
                                  )  # for later dict update compatability

        # * triplane rendering
        ret_dict = self.triplane_decoder(latent_after_vit,
                                         c,
                                         ws=sr_w_code,
                                         return_raw_only=return_raw_only,
                                         **kwargs)  # triplane latent -> imgs
        ret_dict.update({
            'latent_after_vit': latent_after_vit,
            **vit_decode_out
        })

        return ret_dict


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC_final_GCPatchEmbed(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC_final
):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 channel_multiplier=4,
                 ldm_z_channels=4,
                 ldm_embed_dim=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder,
                         triplane_decoder,
                         cls_token,
                         normalize_feat,
                         sr_ratio,
                         use_fusion_blk,
                         fusion_blk_depth,
                         channel_multiplier,
                         ldm_z_channels=ldm_z_channels,
                         ldm_embed_dim=ldm_embed_dim,
                         **kwargs)

        self.superresolution.update(
            dict(ldm_upsample=PatchEmbedTriplane(
                self.vae_p * self.token_size,
                self.vae_p,
                3 * self.ldm_embed_dim,  # B 3 L C
                vit_decoder.embed_dim,
                bias=True)))

        # assert self.cls_token, 'requires [cls] token to project to bcg_w'

    def vit_decode(self, latent, img_size, sample_posterior=True, c=None):

        ret_dict = self.vae_reparameterization(latent, sample_posterior)
        # latent = ret_dict['latent_normalized']

        latent = self.vit_decode_backbone(ret_dict, img_size)
        # st()
        return self.vit_decode_postprocess(latent, ret_dict)

    def vit_decode_backbone(self, latent, img_size):
        # assert x.ndim == 3  # N L C
        if isinstance(latent, dict):
            latent = latent['latent_normalized_2Ddiffusion']  # B, C*3, H, W

        # assert latent.shape == (
        #     latent.shape[0], 3 * (self.token_size * self.vae_p)**2,
        #     self.ldm_z_channels), f'latent.shape: {latent.shape}'

        # st()
        latent = self.superresolution['ldm_upsample'](  # ! B 768 (3*256) 768 
            latent)  # torch.Size([8, 12, 32, 32]) => torch.Size([8, 256, 768])
        # latent: torch.Size([8, 768, 768])
        # st()

        # ! directly feed to vit_decoder
        return self.forward_vit_decoder(latent, img_size)  # pred_vit_latent


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC_final_GCPatchEmbed_PatchEmbedTriplaneRodin(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC_final_GCPatchEmbed
):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, channel_multiplier, **kwargs)
        self.superresolution.update(
            dict(ldm_upsample=PatchEmbedTriplaneRodin(
                self.vae_p * self.token_size,
                self.vae_p,
                3 * self.ldm_embed_dim,  # B 3 L C
                vit_decoder.embed_dim,
                bias=True)))


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC_final_GCPatchEmbed_LiteViT3D(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC_final_GCPatchEmbed
):
    """
    1. reuse attention proj layer from dino
    2. reuse attention; first self then 3D cross attention
    """

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            #  fusion_blk=TriplaneFusionBlockv4,
            fusion_blk=TriplaneFusionBlockv4_nested,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder,
                         triplane_decoder,
                         cls_token,
                         normalize_feat,
                         sr_ratio,
                         use_fusion_blk,
                         fusion_blk_depth,
                         channel_multiplier,
                         fusion_blk=fusion_blk,
                         **kwargs)

        self.superresolution.update(
            dict(conv_sr=RodinConv3D4X_lite_mlp_as_residual_lite(
                int(triplane_decoder.out_chans *
                    channel_multiplier), int(triplane_decoder.out_chans * 1))))


class RodinSR_192(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC_final_GCPatchEmbed_LiteViT3D
):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 fusion_blk=TriplaneFusionBlockv4_nested,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        self.superresolution.update(
            dict(conv_sr=RodinConv3D4X_lite_mlp_as_residual(
                int(triplane_decoder.out_chans * channel_multiplier),
                int(triplane_decoder.out_chans * 1),
                input_resolution=192,
            )))


# ==========================================================================================


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC_final_GCPatchEmbed_f4(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC_final_GCPatchEmbed
):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, channel_multiplier, **kwargs)


# study whether replacing the quant_mlp with quant_embedder


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC_final_GCPatchEmbed_LiteViT3D_convQuant(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC_final_GCPatchEmbed_LiteViT3D
):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 fusion_blk=TriplaneFusionBlockv4_nested,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)
        self.plane_n = 3

        for k in ['quant_mlp']:
            del self.superresolution[k]  # lint unused modules

        # https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/models/autoencoder.py#L302
        self.superresolution.update(
            dict(quant_conv=nn.Conv2d(2 * 3 * self.ldm_z_channels,
                                      2 * self.ldm_embed_dim * 3,
                                      kernel_size=1,
                                      groups=3)))
        # dict(quant_conv=nn.Conv2d(2 * self.ldm_z_channels,
        #                           2 * self.ldm_embed_dim,
        #                           kernel_size=1,
        #                           groups=1)))

    def vae_encode(self, h):
        # * smooth convolution before triplane
        # h = self.superresolution['after_vit_conv'](h)
        # h = h.permute(0, 2, 3, 1)  # B 64 64 6
        B, C, H, W = h.shape  # C=24 here
        # st()
        moments = self.superresolution['quant_conv'](
            h)  # Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), groups=3)

        moments = moments.reshape(
            B,
            # moments.shape[1] // 3,
            moments.shape[1] // self.plane_n,
            # 3,
            self.plane_n,
            H,
            W,
        )  # B C 3 H W

        moments = moments.flatten(-2)  # B C 3 L

        posterior = DiagonalGaussianDistribution(
            moments, soft_clamp=self.reparameterization_soft_clamp)
        return posterior

    def vae_reparameterization(self, latent, sample_posterior):
        """input: latent from ViT encoder
        """
        # ! first downsample for VAE
        # st() # latent: B 256 384
        latents3D = self.superresolution['ldm_downsample'](
            latent)  # latents3D: B 256 96

        assert self.vae_p > 1
        latents3D = self.unpatchify3D(
            latents3D,
            p=self.vae_p,
            unpatchify_out_chans=self.ldm_z_channels *
            2)  # B 3 H W unpatchify_out_chans, H=W=16 now
        #     latents3D = latents3D.reshape(
        #         latents3D.shape[0], 3, -1, latents3D.shape[-1]
        #     )  # B 3 H*W C (H=self.vae_p*self.token_size)
        # else:
        #     latents3D = latents3D.reshape(latents3D.shape[0],
        #                                   latents3D.shape[1], 3,
        #                                   2 * self.ldm_z_channels)  # B L 3 C
        #     latents3D = latents3D.permute(0, 2, 1, 3)  # B 3 L C

        B, _, H, W, C = latents3D.shape
        latents3D = latents3D.permute(0, 1, 4, 2, 3).reshape(B, -1, H,
                                                             W)  # B 3C H W

        # ! do VAE here
        posterior = self.vae_encode(latents3D)  # B self.ldm_z_channels 3 L

        if sample_posterior:
            latent = posterior.sample()
        else:
            latent = posterior.mode()  # B C 3 L

        log_q = posterior.log_p(latent)  # same shape as latent

        # ! for LSGM KL code
        latent_normalized_2Ddiffusion = latent.reshape(
            latent.shape[0], -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)  # B, 3*4, 16 16
        log_q_2Ddiffusion = log_q.reshape(
            latent.shape[0], -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)  # B, 3*4, 16 16
        # st()
        latent = latent.permute(0, 2, 3, 1)  # B C 3 L -> B 3 L C

        latent = latent.reshape(latent.shape[0], -1,
                                latent.shape[-1])  # B 3*L C

        ret_dict = dict(
            normal_entropy=posterior.normal_entropy(),
            latent_normalized=latent,
            latent_normalized_2Ddiffusion=latent_normalized_2Ddiffusion,  # 
            log_q_2Ddiffusion=log_q_2Ddiffusion,
            log_q=log_q,
            posterior=posterior,
        )

        return ret_dict


# todo, study replacing sr mlp with roll out conv


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC_final_bcg(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC_final
):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, channel_multiplier, **kwargs)
        self.superresolution.update(
            dict(
                # bg_quant_mlp=Mlp(2 * self.ldm_z_channels,
                #               out_features=2*512),
                # bg_ldm_downsample=nn.Linear(
                #     384, 512*2, bias=True),
                bg_ldm_downsample=nn.Linear(
                    384, 256, bias=True
                ),  # no vae for this version, just do the synthesis with SG backbone
            ))
        # assert self.cls_token, 'requires [cls] token to project to bcg_w'

    def vit_decode(self, latent, img_size, sample_posterior=True):
        # add bg_latents here
        assert latent.shape[1] == 257, 'requires [cls] for w_bcg here'
        latents_bg = self.superresolution['bg_ldm_downsample'](
            latent[:, 0])  # B 768 -> B 512, no vae required. Bx256
        # latent[:, 0:1])  # B 768 -> B 512
        # return super().vit_decode(latent, img_size, sample_posterior)

        return {
            'z_bcg': latents_bg,
            **super().vit_decode(latent[:, 1:], img_size, sample_posterior),
        }

    def triplane_decode(self, vit_decode_out, c, return_raw_only=False):
        assert isinstance(vit_decode_out, dict)

        # * triplane rendering
        ret_dict = self.triplane_decoder(
            vit_decode_out['latent_after_vit'],
            c,
            z_bcg=vit_decode_out['z_bcg'],
            return_raw_only=return_raw_only)  # triplane latent -> imgs
        ret_dict.update({
            'latent_after_vit': vit_decode_out['latent_after_vit'],
            **vit_decode_out
        })

        return ret_dict

    '''
    def vae_encode_bg(self, h):
        # * smooth convolution before triplane
        moments = self.superresolution['bg_quant_mlp'](
            h)  # B 1 self.ldm_z_channels, must has C in the second dimension

        posterior = DiagonalGaussianDistribution(
            moments, soft_clamp=self.reparameterization_soft_clamp)
        return posterior

    def vae_reparameterization(self, latent, sample_posterior):
        """input: latent from ViT encoder
        """
        # 2D bg branch
        # cls_latent = latent[:, 0] # https://github.com/facebookresearch/dinov2/blob/c3c2683a13cde94d4d99f523cf4170384b00c34c/dinov2/models/vision_transformer.py#L232C32-L232C44
        latents_bg = self.superresolution['bg_ldm_downsample'](latent[:, 0:1])  # B 768 -> B 512
        # ! do VAE here
        posterior_bg = self.vae_encode_bg(latents_bg)  

        if sample_posterior:
            latents_bg = posterior_bg.sample()
        else:
            latents_bg = posterior_bg.mode()  
        
        latents_bg = latents_bg

        log_q_bg = posterior_bg.log_p(latents_bg)  # same shape as latent

        ret_dict = dict(
            latents_bg=latents_bg,
            log_q_bg=log_q_bg,
            posterior_bg=posterior_bg,
        )

        # ! processes 3D latents
        ret_dict.update(
            **super().vae_reparameterization(latent, sample_posterior)
        ) 

        return ret_dict
    '''


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC_improved(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC
):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, channel_multiplier, **kwargs)

        self.superresolution.update(
            dict(
                conv_sr=
                #  RodinConv3D4X_lite_mlp_as_residual_improved( # not converging
                RodinConv3D4X_lite_improved_lint_withresidual(
                    int(triplane_decoder.out_chans * channel_multiplier),
                    int(triplane_decoder.out_chans *
                        2), int(triplane_decoder.out_chans * 1))))


class VAE_SR_4XC_mimic2D(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC
):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, channel_multiplier, **kwargs)

        for k in [
                'conv1', 'conv2', 'norm', 'quant_conv', 'proj_upsample_224',
                'after_vit_conv'
        ]:
            del self.superresolution[k]
        del self.IN

        self.superresolution.update(
            dict(conv_sr=
                 RodinConv3D4X_lite_mlp_as_residual_improved(  # not converging
                     # RodinConv3D4X_lite_improved_lint_withresidual(
                     int(triplane_decoder.out_chans * channel_multiplier),
                     int(triplane_decoder.out_chans * 2),
                     int(triplane_decoder.out_chans * 1))))

    def triplane_decode(self, vit_decode_out, c, return_raw_only=False):

        if isinstance(vit_decode_out, dict):
            latent_after_vit, sr_w_code = (vit_decode_out.get(k, None)
                                           for k in ('latent_after_vit',
                                                     'sr_w_code'))

        else:
            latent_after_vit = vit_decode_out
            sr_w_code = None
            vit_decode_out = dict(latent_normalized=latent_after_vit
                                  )  # for later dict update compatability

        # * triplane rendering
        ret_dict = self.triplane_decoder(
            latent_after_vit, c, ws=sr_w_code,
            return_raw_only=return_raw_only)  # triplane latent -> imgs
        ret_dict.update({
            'latent_after_vit': latent_after_vit,
            **vit_decode_out
        })

        return ret_dict


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_more_SR_BLKs(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit):
    """ 4*4 SR
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 channel_multiplier=2,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            decoder_pred_size=(4 // 1)**2 *
            int(triplane_decoder.out_chans // 3 * channel_multiplier),
            **kwargs)
        assert channel_multiplier >= 2  # 2 1.5 1
        self.channel_multiplier = channel_multiplier
        self.superresolution.update(
            dict(conv_sr_64_128=RodinConv3D4X_lite_mlp_as_residual(
                int(triplane_decoder.out_chans * channel_multiplier),
                int(triplane_decoder.out_chans *
                    max(1.5, channel_multiplier // 2)),
                input_resolution=128),
                 conv_sr_128_256=RodinConv3D4X_lite_mlp_as_residual(
                     int(triplane_decoder.out_chans *
                         max(1.5, channel_multiplier // 2)),
                     int(triplane_decoder.out_chans * 1),
                     input_resolution=256)))

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify_triplane(
            latent,
            p=4,
            unpatchify_out_chans=int(
                self.channel_multiplier * self.unpatchify_out_chans //
                3))  # spatial_vit_latent, B, C, H, W (B, 96*2, 16, 16)

        # 2X SR with Rodin Conv 3D, 64->128
        latent = self.superresolution['conv_sr_64_128'](
            latent)  # still B 3C H W

        # 2X SR with Rodin Conv 3D, 128 -> 256
        latent = self.superresolution['conv_sr_128_256'](
            latent)  # still B 3C H W

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        sr_w_code = self.w_avg
        assert sr_w_code is not None
        ret_dict.update(
            dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
                latent.shape[0], 0), ))  # type: ignore

        return ret_dict


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_more_SR_BLKs_X4C(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_more_SR_BLKs
):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder,
                         triplane_decoder,
                         cls_token,
                         normalize_feat,
                         sr_ratio,
                         use_fusion_blk,
                         fusion_blk_depth,
                         channel_multiplier=channel_multiplier,
                         **kwargs)


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_more_SR_BLKs_X3C(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_more_SR_BLKs
):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 channel_multiplier=3,
                 **kwargs) -> None:
        super().__init__(vit_decoder,
                         triplane_decoder,
                         cls_token,
                         normalize_feat,
                         sr_ratio,
                         use_fusion_blk,
                         fusion_blk_depth,
                         channel_multiplier=channel_multiplier,
                         **kwargs)


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_more_SR_BLKs_332(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit):
    """ 4*4 SR
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 channel_multiplier=2,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            decoder_pred_size=(3 // 1)**2 *
            int(triplane_decoder.out_chans // 3 * channel_multiplier),
            **kwargs)
        assert channel_multiplier == 2  # 2 1.5 1
        self.channel_multiplier = channel_multiplier
        self.superresolution.update(
            dict(conv_sr_48_144=RodinConv3D4X_lite_mlp_as_residual(
                int(triplane_decoder.out_chans * channel_multiplier),
                int(triplane_decoder.out_chans * 1.5),
                input_resolution=128),
                 conv_sr_144_256=RodinConv3D4X_lite_mlp_as_residual(
                     int(triplane_decoder.out_chans * 1.5),
                     int(triplane_decoder.out_chans * 1),
                     input_resolution=256)))

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify_triplane(
            latent,
            p=3,
            unpatchify_out_chans=int(
                self.channel_multiplier * self.unpatchify_out_chans //
                3))  # spatial_vit_latent, B, C, H, W (B, 96*2, 16, 16)

        # 2X SR with Rodin Conv 3D, 64->128
        latent = self.superresolution['conv_sr_48_144'](
            latent)  # still B 3C H W

        # 2X SR with Rodin Conv 3D, 128 -> 256
        latent = self.superresolution['conv_sr_144_256'](
            latent)  # still B 3C H W

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        sr_w_code = self.w_avg
        assert sr_w_code is not None
        ret_dict.update(
            dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
                latent.shape[0], 0), ))  # type: ignore

        return ret_dict


# ============= just for debugging ==============


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_2DVAE(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit):
    """ simply remove the volume rendering, triplane directly for reconstruction. check why the reconstruction sometimes fails...
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            decoder_pred_size=(4 // 1)**2 *
            (triplane_decoder.out_chans // 3 *
             1),  # ! no need to triple the dimension in the final MLP
            **kwargs)
        self.triplane_decoder = nn.Identity()
        self.superresolution.update(
            dict(conv2=nn.Conv2d(triplane_decoder.out_chans, 3, 3, padding=1)))

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify_triplane(
            latent, p=4, unpatchify_out_chans=self.unpatchify_out_chans //
            3)  # spatial_vit_latent, B, C, H, W (B, 96, 112, 112)

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        sr_w_code = self.w_avg
        assert sr_w_code is not None
        ret_dict.update(
            dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
                latent.shape[0], 0), ))  # type: ignore

        return ret_dict

    def triplane_decode(self, vit_decode_out, c, return_raw_only=False):

        if isinstance(vit_decode_out, dict):
            latent_after_vit, sr_w_code = (vit_decode_out.get(k, None)
                                           for k in ('latent_after_vit',
                                                     'sr_w_code'))

        else:
            latent_after_vit = vit_decode_out
            sr_w_code = None
            vit_decode_out = dict(latent_normalized=latent_after_vit
                                  )  # for later dict update compatability

        latent_denormalized = self.superresolution['conv2'](
            latent_after_vit)  # post_quant_conv
        return {'image_raw': latent_denormalized, **vit_decode_out}

        # * triplane rendering

        # ret_dict = self.triplane_decoder(
        #     latent_denormalized,
        #     c,
        #     ws=sr_w_code,
        #     return_raw_only=return_raw_only)  # triplane latent -> imgs
        # ret_dict.update({
        #     'latent_denormalized': latent_denormalized,
        #     **vit_decode_out
        # })

        # return ret_dict


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_3DVAE(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit):
    """ simply remove the volume rendering, triplane directly for reconstruction. check why the reconstruction sometimes fails...
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            decoder_pred_size=(4 // 1)**2 *
            (triplane_decoder.out_chans // 3 *
             1),  # ! no need to triple the dimension in the final MLP
            **kwargs)

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify_triplane(
            latent, p=4, unpatchify_out_chans=self.unpatchify_out_chans //
            3)  # spatial_vit_latent, B, C, H, W (B, 96, 112, 112)

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        sr_w_code = self.w_avg
        assert sr_w_code is not None
        ret_dict.update(
            dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
                latent.shape[0], 0), ))  # type: ignore

        return ret_dict

    def triplane_decode(self, vit_decode_out, c, return_raw_only=False):

        if isinstance(vit_decode_out, dict):
            latent_after_vit, sr_w_code = (vit_decode_out.get(k, None)
                                           for k in ('latent_after_vit',
                                                     'sr_w_code'))

        else:
            latent_after_vit = vit_decode_out
            sr_w_code = None
            vit_decode_out = dict(latent_normalized=latent_after_vit
                                  )  # for later dict update compatability

        latent_denormalized = self.superresolution['conv2'](
            latent_after_vit)  # post_quant_conv
        # return {
        #     'image_raw': latent_denormalized,
        #     **vit_decode_out
        # }

        # * triplane rendering

        ret_dict = self.triplane_decoder(
            latent_denormalized,
            c,
            ws=sr_w_code,
            return_raw_only=return_raw_only)  # triplane latent -> imgs
        ret_dict.update({
            'latent_denormalized': latent_denormalized,
            **vit_decode_out
        })

        return ret_dict


class VAE_SR_224(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit):
    """ 4*4 SR
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 channel_multiplier=4,
                 p=2,
                 **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            decoder_pred_size=(p // 1)**2 *
            int(triplane_decoder.out_chans // 3 * channel_multiplier),
            **kwargs)
        self.channel_multiplier = channel_multiplier
        self.p = p

        # remove unused modules
        for k in [
                'conv1', 'conv2', 'norm', 'quant_conv', 'proj_upsample_224',
                'after_vit_conv'
        ]:
            del self.superresolution[k]
        del self.IN

        self.superresolution.update(
            dict(conv_sr_32_64=RodinConv3D4X_lite_mlp_as_residual(
                int(triplane_decoder.out_chans * channel_multiplier),
                int(triplane_decoder.out_chans * channel_multiplier),
                input_resolution=64,
            ),
                 conv_sr_64_256=RodinConv3D4X_lite_mlp_as_residual(
                     int(triplane_decoder.out_chans * channel_multiplier),
                     int(triplane_decoder.out_chans * 1))))

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit)  # pred_vit_latent -> patch or original size

        latent = self.unpatchify_triplane(
            latent,
            p=self.p,
            unpatchify_out_chans=int(
                self.channel_multiplier * self.unpatchify_out_chans //
                3))  # spatial_vit_latent, B, C, H, W (B, 96*2, 16, 16)

        # 4X SR with Rodin Conv 3D
        latent = self.superresolution['conv_sr_32_64'](
            latent)  # still B 3C H W
        latent = self.superresolution['conv_sr_64_256'](
            latent)  # still B 3C H W

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        sr_w_code = self.w_avg
        assert sr_w_code is not None
        ret_dict.update(
            dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
                latent.shape[0], 0), ))  # type: ignore

        return ret_dict

    def triplane_decode(self, vit_decode_out, c, return_raw_only=False):

        if isinstance(vit_decode_out, dict):
            latent_after_vit, sr_w_code = (vit_decode_out.get(k, None)
                                           for k in ('latent_after_vit',
                                                     'sr_w_code'))

        else:
            latent_after_vit = vit_decode_out
            sr_w_code = None
            vit_decode_out = dict(latent_normalized=latent_after_vit
                                  )  # for later dict update compatability

        # * triplane rendering
        ret_dict = self.triplane_decoder(
            latent_after_vit, c, ws=sr_w_code,
            return_raw_only=return_raw_only)  # triplane latent -> imgs
        ret_dict.update({
            'latent_after_vit': latent_after_vit,
            **vit_decode_out
        })

        return ret_dict


class RodinSR_192(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC_final_GCPatchEmbed_LiteViT3D
):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 fusion_blk=TriplaneFusionBlockv4_nested,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        self.superresolution.update(
            dict(conv_sr=RodinConv3D4X_lite_mlp_as_residual(
                int(triplane_decoder.out_chans * channel_multiplier),
                int(triplane_decoder.out_chans * 1),
                input_resolution=192,
            )))


class RodinSR_192_fusionv5(RodinSR_192):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 fusion_blk=TriplaneFusionBlockv5_ldm_addCA,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)


class RodinSR_192_fusionv5_dualCA(RodinSR_192):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 fusion_blk=TriplaneFusionBlockv5_ldm_add_dualCA,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)


class RodinSR_256_fusionv5(RodinSR_192):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 fusion_blk=TriplaneFusionBlockv5_ldm_addCA,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        self.superresolution.update(
            dict(conv_sr=RodinConv3D4X_lite_mlp_as_residual(
                int(triplane_decoder.out_chans * channel_multiplier),
                int(triplane_decoder.out_chans * 1),
                input_resolution=256,
            )))


# class LiteSR_192_fusionv5(
class RodinSR_192_fusionv5_ConvQuant(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC_final_GCPatchEmbed_LiteViT3D_convQuant
):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 fusion_blk=TriplaneFusionBlockv5_ldm_addCA,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        # ! must use rodin conv or no 3D?
        self.superresolution.update(
            dict(conv_sr=RodinConv3D4X_lite_mlp_as_residual(
                int(triplane_decoder.out_chans * channel_multiplier),
                int(triplane_decoder.out_chans * 1),
                input_resolution=192,
            )))


class RodinSR_256_fusionv5_ConvQuant(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC_final_GCPatchEmbed_LiteViT3D_convQuant
):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 fusion_blk=TriplaneFusionBlockv5_ldm_addCA,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        # ! must use rodin conv or no 3D?
        self.superresolution.update(
            dict(conv_sr=RodinConv3D4X_lite_mlp_as_residual(
                int(triplane_decoder.out_chans * channel_multiplier),
                int(triplane_decoder.out_chans * 1),
                input_resolution=256,
            )))


class RodinSR_256_fusionv5_ConvQuant_fg_bgtriplane(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC_final_GCPatchEmbed_LiteViT3D_convQuant
):
    """add a background tri-plane as in nerf++/stylenerf.
    """

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 fusion_blk=TriplaneFusionBlockv5_ldm_addCA,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        # new renderer with fg-bg support
        self.triplane_decoder.renderer = ImportanceRendererfg_bg(
        )  # support fg/bg composition

        # add a background triplane prediction layer.
        self.superresolution.update(
            dict(conv_sr=RodinConv3D4X_lite_mlp_as_residual(
                int(triplane_decoder.out_chans * channel_multiplier),
                int(triplane_decoder.out_chans * 1),
                input_resolution=256,
                bcg_triplane=True)))

    # def vit_decode(self, latent, img_size, sample_posterior=True):
    #     return super().vit_decode(latent, img_size, sample_posterior)

    # def triplane_decode(self, vit_decode_out, c, return_raw_only=False):

    #     assert isinstance(vit_decode_out, dict)
    #     latent_after_vit, sr_w_code = (vit_decode_out.get(k, None)
    #                                     for k in ('latent_after_vit',
    #                                                 'sr_w_code'))
    #     assert isinstance(latent_after_vit, tuple), 'x, x_bcg'

    #     # * triplane rendering
    #     ret_dict = self.triplane_decoder(
    #         latent_after_vit,
    #         c, ws=sr_w_code,
    #         return_raw_only=return_raw_only)  # triplane latent -> imgs
    #     ret_dict.update({
    #         'latent_after_vit': latent_after_vit,
    #         **vit_decode_out
    #     })

    #     return ret_dict


# ! unused class
class RodinSR_256_fusionv5_ConvQuant_fg_bgtriplane_sddecoder(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC_final_GCPatchEmbed_LiteViT3D_convQuant
):
    # the vae decoder output an extra plane as the 2D decoder latent output.
    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane_fg_bg_plane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 fusion_blk=TriplaneFusionBlockv4_nested,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)
        self.plane_n = 4

        self.superresolution.update(
            dict(
                ldm_downsample=nn.Linear(
                    384,
                    self.vae_p * self.vae_p * self.plane_n *
                    self.ldm_z_channels * 2,  # 3->4, add a bg code
                    bias=True),
                quant_conv=nn.Conv2d(2 * self.plane_n * self.ldm_z_channels,
                                     2 * self.ldm_embed_dim * self.plane_n,
                                     kernel_size=1,
                                     groups=self.plane_n)))

    def vae_reparameterization(self, latent, sample_posterior):
        """input: latent from ViT encoder
        """
        # ! first downsample for VAE
        latents3D = self.superresolution['ldm_downsample'](latent)  # B L 24
        # batch_size = latent.shape[0]

        assert self.vae_p > 1
        latents3D = self.unpatchify3D(
            latents3D,
            p=self.vae_p,
            unpatchify_out_chans=self.ldm_z_channels * 2,
            plane_n=self.plane_n)  # B 3 H W unpatchify_out_chans, H=W=16 now

        B, _, H, W, C = latents3D.shape
        latents3D = latents3D.permute(0, 1, 4, 2, 3).reshape(B, -1, H,
                                                             W)  # B 3C H W

        # ! do VAE here
        posterior = self.vae_encode(latents3D)  # B self.ldm_z_channels 3 L

        if sample_posterior:
            latent = posterior.sample()
        else:
            latent = posterior.mode()  # B C 3 L

        log_q = posterior.log_p(latent)  # same shape as latent

        # ! for LSGM KL code
        # latent_normalized_2Ddiffusion = latent.reshape(
        latent = latent.reshape(  # ! reuse var name
            B, -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)  # B, self.plane_n*4, 16 16
        # ! split for two branch
        latent_normalized_2Ddiffusion, bg_plane = torch.split(
            latent, [self.plane_n * 3, self.plane_n * 1], dim=1)
        # st()

        log_q_2Ddiffusion = log_q.reshape(
            B, -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)  # B, self.plane_n*4, 16 16

        # latent = latent.permute(0, 2, 3, 1)  # B C 3 L -> B 3 L C
        # fg_latent, bg_latent = torch.split(latent, [3, 1], dim=1)

        # latent = latent.reshape(latent.shape[0], -1,
        #                         latent.shape[-1])  # B 3*L C

        # fg_latent = fg_latent.reshape(fg_latent.shape[0], -1,
        #                               fg_latent.shape[-1])  # B 3*L C

        # unpachify bg_latent, Bx32x32x4
        # bg_latent = bg_latent.squeeze(1).reshape(bg_latent.shape[0],
        #                                          self.token_size * self.vae_p,
        #                                          self.token_size * self.vae_p,
        #                                          bg_latent.shape[-1]).permute(0,3,1,2) # B C H W

        ret_dict = dict(
            # latent_normalized=fg_latent,
            # latent_normalized=None, # ! not required?
            bg_plane=bg_plane,
            latent_normalized_2Ddiffusion=latent_normalized_2Ddiffusion,  # 
            log_q_2Ddiffusion=log_q_2Ddiffusion,
            log_q=log_q,
            posterior=posterior,
            # normal_entropy=posterior.normal_entropy(),
        )

        return ret_dict

    def triplane_decode(self, vit_decode_out, c, return_raw_only=False):

        assert isinstance(vit_decode_out, dict)
        # latent_after_vit, bg_plane = (vit_decode_out.get(k, None)
        #                                 for k in ('latent_after_vit',
        #                                             'bg_plane'))

        # * triplane rendering
        ret_dict = self.triplane_decoder(
            vit_decode_out['latent_after_vit'],
            vit_decode_out['bg_plane'],
            c,
            return_raw_only=return_raw_only)  # triplane latent -> imgs
        # st()
        ret_dict.update({
            # 'latent_after_vit': vit_decode_out['latent_after_vit'],
            **vit_decode_out
        })

        return ret_dict


class RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn(
        VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC_final_GCPatchEmbed_LiteViT3D_convQuant
):
    # lite version, no sd-bg, use TriplaneFusionBlockv4_nested_init_from_dino
    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane_fg_bg_plane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        # self.superresolution.update(
        #     dict(conv_sr=RodinConv3D4X_lite_mlp_as_residual_litev2(
        #         int(triplane_decoder.out_chans * channel_multiplier),
        #         int(triplane_decoder.out_chans * 1),
        #         input_resolution=256,
        #         bcg_triplane=False)))
    def vit_decode_backbone(self, latent, img_size):
        return super().vit_decode_backbone(latent, img_size)


class RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn_liteSR(
        RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane_fg_bg_plane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        self.superresolution.update(
            dict(conv_sr=RodinConv3D4X_lite_mlp_as_residual_litev2(
                int(triplane_decoder.out_chans * channel_multiplier),
                int(triplane_decoder.out_chans * 1),
                input_resolution=256,
                bcg_triplane=False)))


# class RodinSR_256_fusionv5_ConvQuant_fg_bgtriplane_sddecoder_liteSR_dinoInit3DAttn(RodinSR_256_fusionv5_ConvQuant_fg_bgtriplane_sddecoder):
#     # lite version, with sd-bg, use TriplaneFusionBlockv4_nested_init_from_dino
#     def __init__(self, vit_decoder: VisionTransformer, triplane_decoder: Triplane_fg_bg_plane, cls_token, normalize_feat=True, sr_ratio=2, use_fusion_blk=True, fusion_blk_depth=2, fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino, channel_multiplier=4, **kwargs) -> None:
#         super().__init__(vit_decoder, triplane_decoder, cls_token, normalize_feat, sr_ratio, use_fusion_blk, fusion_blk_depth, fusion_blk, channel_multiplier, **kwargs)

#         self.superresolution.update(
#             dict(conv_sr=RodinConv3D4X_lite_mlp_as_residual_litev2(
#                 int(triplane_decoder.out_chans * channel_multiplier),
#                 int(triplane_decoder.out_chans * 1),
#                 input_resolution=256,
#                 bcg_triplane=False)))


# ! SD version, encoder only
class RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn_SD(
        RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane_fg_bg_plane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        for k in [
                'ldm_downsample',
                # 'conv_sr'
        ]:
            del self.superresolution[k]

    def vae_reparameterization(self, latent, sample_posterior):
        # latent: B 24 32 32
        # latent = latent.float()
        with torch.cuda.amp.autocast(
                enabled=False, dtype=torch.bfloat16, cache_enabled=True
        ):  # only handles the execusion, not data type

            assert self.vae_p >= 1
            # latents3D = self.unpatchify3D(
            #     latents3D,
            #     p=self.vae_p,
            #     unpatchify_out_chans=self.ldm_z_channels *
            #     2)  # B 3 H W unpatchify_out_chans, H=W=16 now
            # B, C3, H, W = latent.shape
            # latents3D = latent.reshape(B, 3, C3//3, H, W)

            # latents3D = latents3D.permute(0, 1, 4, 2, 3).reshape(B, -1, H,
            #                                                      W)  # B 3C H W

            # st()

            # ! do VAE
            posterior = self.vae_encode(latent.float())
            # st()

            if sample_posterior:
                latent = posterior.sample(
                )  # # B self.ldm_z_channels 3 L(HW), self.ldm_z_channels=4 here
            else:
                latent = posterior.mode()  # B C 3 L

            log_q = posterior.log_p(latent)  # same shape as latent

            # st()
            vae_spatial_latent_size = int(latent.shape[-1]**0.5)

            # ! for LSGM KL code
            # latent_normalized_2Ddiffusion = latent.reshape(
            #     latent.shape[0], -1, self.token_size * self.vae_p,
            #     self.token_size * self.vae_p)  # B, 4*3, 16 16
            # log_q_2Ddiffusion = log_q.reshape(
            #     latent.shape[0], -1, self.token_size * self.vae_p,
            #     self.token_size * self.vae_p)  # B, 4*3, 16 16

            latent_normalized_2Ddiffusion = latent.reshape(
                latent.shape[0], -1, vae_spatial_latent_size,
                vae_spatial_latent_size)  # B, 4*3, 16 16
            log_q_2Ddiffusion = log_q.reshape(
                latent.shape[0], -1, vae_spatial_latent_size,
                vae_spatial_latent_size)  # B, 4*3, 16 16

            # st()

            latent = latent.permute(0, 2, 3, 1)  # B C 3 L -> B 3 L C

            latent = latent.reshape(latent.shape[0], -1,
                                    latent.shape[-1])  # B 3*L C

            ret_dict = dict(
                normal_entropy=posterior.normal_entropy(),
                latent_normalized=latent,
                latent_normalized_2Ddiffusion=latent_normalized_2Ddiffusion,  # 
                log_q_2Ddiffusion=log_q_2Ddiffusion,
                log_q=log_q,
                posterior=posterior,
            )

            return ret_dict


class RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn_SD_D(
        RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn_SD):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane_fg_bg_plane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        self.decoder_pred = None  # directly un-patchembed

        self.superresolution.update(
            dict(
                conv_sr=Decoder(  # serve as Deconv
                    resolution=128,
                    # resolution=256,
                    in_channels=3,
                    # ch=64,
                    ch=32,
                    # ch=16,
                    ch_mult=[1, 2, 2, 4],
                    # ch_mult=[1, 1, 2, 2],
                    # num_res_blocks=2,
                    # ch_mult=[1,2,4],
                    # num_res_blocks=0,
                    num_res_blocks=1,
                    dropout=0.0,
                    attn_resolutions=[],
                    out_ch=32,
                    # z_channels=vit_decoder.embed_dim//4,
                    # z_channels=vit_decoder.embed_dim,
                    z_channels=vit_decoder.embed_dim // 2,
                ),
                after_vit_upsampler=Upsample2D(
                    channels=vit_decoder.embed_dim,
                    use_conv=True,
                    use_conv_transpose=False,
                    out_channels=vit_decoder.embed_dim // 2)))

    # ''' # for SD Decoder, verify encoder first
    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        def unflatten_token(x, p=None):
            B, L, C = x.shape
            x = x.reshape(B, 3, L // 3, C)

            if self.cls_token:  # TODO, how to better use cls token
                x = x[:, :, 1:]  # B 3 256 C

            h = w = int((x.shape[2])**.5)
            assert h * w == x.shape[2]

            if p is None:
                x = x.reshape(shape=(B, 3, h, w, -1))
                x = rearrange(
                    x, 'b n h w c->(b n) c h w'
                )  # merge plane into Batch and prepare for rendering
            else:
                x = x.reshape(shape=(B, 3, h, w, p, p, -1))
                x = rearrange(
                    x, 'b n h w p1 p2 c->(b n) c (h p1) (w p2)'
                )  # merge plane into Batch and prepare for rendering

            return x

        latent = unflatten_token(
            latent_from_vit)  # B 3 h w vit_decoder.embed_dim

        # ! x2 upsapmle, 16 -32 before sending into SD Decoder
        latent = self.superresolution['after_vit_upsampler'](
            latent)  # B*3 192 32 32

        # latent = unflatten_token(latent_from_vit, p=2)

        # ! SD SR
        latent = self.superresolution['conv_sr'](latent)  # still B 3C H W
        latent = rearrange(latent, '(b n) c h w->b (n c) h w', n=3)

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        # sr_w_code = self.w_avg
        # assert sr_w_code is not None
        # ret_dict.update(
        #     dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
        #         latent_from_vit.shape[0], 0), ))  # type: ignore

        return ret_dict

    # '''


# ! optimized 3D Decoder
class RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_lite3DAttn(
        RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn_SD):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane_fg_bg_plane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)
        # 1. convert output plane token to B L 3 C//3 shape
        # 2. change vit decoder fusion arch (fusion block)
        # 3. output follow B L 3 C//3 with decoder input dim C//3
        # TODO: ablate basic decoder design, on the metrics (input/novelview both)
        self.decoder_pred = nn.Linear(self.vit_decoder.embed_dim // 3,
                                      2048,
                                      bias=True)  # decoder to patch

        # st()
        self.superresolution.update(
            dict(ldm_upsample=PatchEmbedTriplaneRodin(
                self.vae_p * self.token_size,
                self.vae_p,
                3 * self.ldm_embed_dim,  # B 3 L C
                vit_decoder.embed_dim // 3,
                bias=True)))

        # ! original pos_embed
        has_token = bool(self.cls_token)
        self.vit_decoder.pos_embed = nn.Parameter(
            torch.zeros(1, 16 * 16 + has_token, vit_decoder.embed_dim))

    def vae_reparameterization(self, latent, sample_posterior):
        # latent: B 24 32 32

        assert self.vae_p > 1

        # ! do VAE here
        # st()
        posterior = self.vae_encode(latent)  # B self.ldm_z_channels 3 L

        if sample_posterior:
            latent = posterior.sample()
        else:
            latent = posterior.mode()  # B C 3 L

        log_q = posterior.log_p(latent)  # same shape as latent

        # ! for LSGM KL code
        latent_normalized_2Ddiffusion = latent.reshape(
            latent.shape[0], -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)  # B, 3*4, 16 16
        log_q_2Ddiffusion = log_q.reshape(
            latent.shape[0], -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)  # B, 3*4, 16 16

        # TODO, add a conv_after_quant

        # ! reshape for ViT decoder
        latent = latent.permute(0, 3, 1, 2)  # B C 3 L -> B L C 3
        latent = latent.reshape(*latent.shape[:2], -1)  # B L C3

        ret_dict = dict(
            normal_entropy=posterior.normal_entropy(),
            latent_normalized=latent,
            latent_normalized_2Ddiffusion=latent_normalized_2Ddiffusion,  # 
            log_q_2Ddiffusion=log_q_2Ddiffusion,
            log_q=log_q,
            posterior=posterior,
        )

        return ret_dict

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        B, N, C = latent_from_vit.shape
        latent_from_vit = latent_from_vit.reshape(B, N, C // 3, 3).permute(
            0, 3, 1, 2)  # -> B 3 N C//3

        # ! remaining unchanged

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit
        )  # pred_vit_latent -> patch or original size; B 768 384

        latent = latent.reshape(B, 3 * N, -1)  # B L C

        latent = self.unpatchify_triplane(
            latent,
            p=4,
            unpatchify_out_chans=int(
                self.channel_multiplier * self.unpatchify_out_chans //
                3))  # spatial_vit_latent, B, C, H, W (B, 96*2, 16, 16)

        # 4X SR with Rodin Conv 3D
        latent = self.superresolution['conv_sr'](latent)  # still B 3C H W

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        sr_w_code = self.w_avg
        assert sr_w_code is not None
        ret_dict.update(
            dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
                latent_from_vit.shape[0], 0), ))  # type: ignore

        return ret_dict

    def vit_decode_backbone(self, latent, img_size):
        # assert x.ndim == 3  # N L C
        if isinstance(latent, dict):
            latent = latent['latent_normalized_2Ddiffusion']  # B, C*3, H, W

        # assert latent.shape == (
        #     latent.shape[0], 3 * (self.token_size * self.vae_p)**2,
        #     self.ldm_z_channels), f'latent.shape: {latent.shape}'

        latent = self.superresolution['ldm_upsample'](  # ! B 768 (3*256) 768 
            latent)  # torch.Size([8, 12, 32, 32]) => torch.Size([8, 256, 768])
        # latent: torch.Size([8, 768, 768])

        B, N3, C = latent.shape
        latent = latent.reshape(B, 3, N3 // 3,
                                C).permute(0, 2, 3, 1)  # B 3HW C -> B HW C 3
        latent = latent.reshape(*latent.shape[:2], -1)  # B HW C3

        # ! directly feed to vit_decoder
        return self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

    def forward_vit_decoder(self, x, img_size=None):
        # latent: (N, L, C) from DINO/CLIP ViT encoder

        # * also dino ViT
        # add positional encoding to each token
        if img_size is None:
            img_size = self.img_size

        # if self.cls_token:
        x = x + self.interpolate_pos_encoding(x, img_size,
                                              img_size)[:, :]  # B, L, C

        B, L, C = x.shape  # has [cls] token in N

        # ! no need to reshape here
        # x = x.view(B, 3, L // 3, C)

        skips = [x]
        assert self.fusion_blk_start == 0

        # in blks
        for blk in self.vit_decoder.blocks[0:len(self.vit_decoder.blocks) //
                                           2 - 1]:
            x = blk(x)  # B 3 N C
            skips.append(x)

        # mid blks
        # for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks)//2-1:len(self.vit_decoder.blocks)//2+1]:
        for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks) // 2 -
                                           1:len(self.vit_decoder.blocks) //
                                           2]:
            x = blk(x)  # B 3 N C

        # out blks
        for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks) // 2:]:
            x = x + blk.skip_linear(torch.cat([x, skips.pop()],
                                              dim=-1))  # long skip connections
            x = blk(x)  # B 3 N C

        x = self.vit_decoder.norm(x)

        # post process shape
        x = x.view(B, L, C)
        return x

    def create_fusion_blks(self, fusion_blk_depth, use_fusion_blk, fusion_blk):

        vit_decoder_blks = self.vit_decoder.blocks
        # assert len(vit_decoder_blks) == 12, 'ViT-B by default'

        # nh = self.vit_decoder.blocks[
        #     0].attn.num_heads // 3  # ! lighter, actually divisible by 4
        # dim = self.vit_decoder.embed_dim // 3  # ! separate

        nh = self.vit_decoder.blocks[
            0].attn.num_heads // 1  # ! lighter, actually divisible by 4
        dim = self.vit_decoder.embed_dim // 1  # ! separate

        fusion_blk_start = self.fusion_blk_start
        triplane_fusion_vit_blks = nn.ModuleList()

        if fusion_blk_start != 0:
            for i in range(0, fusion_blk_start):
                triplane_fusion_vit_blks.append(
                    vit_decoder_blks[i])  # append all vit blocks in the front

        for i in range(fusion_blk_start, len(vit_decoder_blks),
                       fusion_blk_depth):
            vit_blks_group = vit_decoder_blks[i:i +
                                              fusion_blk_depth]  # moduleList
            triplane_fusion_vit_blks.append(
                # TriplaneFusionBlockv2(vit_blks_group, nh, dim, use_fusion_blk))
                fusion_blk(vit_blks_group, nh, dim, use_fusion_blk))

        self.vit_decoder.blocks = triplane_fusion_vit_blks


class RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_lite3DAttn_merge(
        RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_lite3DAttn):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)


class RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_lite3DAttn_merge_add3DAttn(
        RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn_SD):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_add3DAttn,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)


class RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C(
        RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn_SD):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)


class RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout(
        RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn_SD):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)


# final version, above + SD-Decoder
class RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D(
        RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout
):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        self.decoder_pred = None  # directly un-patchembed
        self.superresolution.update(
            dict(
                conv_sr=Decoder(  # serve as Deconv
                    resolution=128,
                    # resolution=256,
                    in_channels=3,
                    # ch=64,
                    ch=32,
                    # ch=16,
                    ch_mult=[1, 2, 2, 4],
                    # ch_mult=[1, 1, 2, 2],
                    # num_res_blocks=2,
                    # ch_mult=[1,2,4],
                    # num_res_blocks=0,
                    num_res_blocks=1,
                    dropout=0.0,
                    attn_resolutions=[],
                    out_ch=32,
                    # z_channels=vit_decoder.embed_dim//4,
                    z_channels=vit_decoder.embed_dim,
                    # z_channels=vit_decoder.embed_dim//2,
                ),
                # after_vit_upsampler=Upsample2D(channels=vit_decoder.embed_dim,use_conv=True, use_conv_transpose=False, out_channels=vit_decoder.embed_dim//2)
            ))
        self.D_roll_out_input = False

    # ''' # for SD Decoder
    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        def unflatten_token(x, p=None):
            B, L, C = x.shape
            x = x.reshape(B, 3, L // 3, C)

            if self.cls_token:  # TODO, how to better use cls token
                x = x[:, :, 1:]  # B 3 256 C

            h = w = int((x.shape[2])**.5)
            assert h * w == x.shape[2]

            if p is None:
                x = x.reshape(shape=(B, 3, h, w, -1))
                if not self.D_roll_out_input:
                    x = rearrange(
                        x, 'b n h w c->(b n) c h w'
                    )  # merge plane into Batch and prepare for rendering
                else:
                    x = rearrange(
                        x, 'b n h w c->b c h (n w)'
                    )  # merge plane into Batch and prepare for rendering
            else:
                x = x.reshape(shape=(B, 3, h, w, p, p, -1))
                if self.D_roll_out_input:
                    x = rearrange(
                        x, 'b n h w p1 p2 c->b c (h p1) (n w p2)'
                    )  # merge plane into Batch and prepare for rendering
                else:
                    x = rearrange(
                        x, 'b n h w p1 p2 c->(b n) c (h p1) (w p2)'
                    )  # merge plane into Batch and prepare for rendering

            return x

        latent = unflatten_token(
            latent_from_vit)  # B 3 h w vit_decoder.embed_dim

        # ! x2 upsapmle, 16 -32 before sending into SD Decoder
        # latent = self.superresolution['after_vit_upsampler'](latent) # B*3 192 32 32

        # latent = unflatten_token(latent_from_vit, p=2)

        # ! SD SR
        latent = self.superresolution['conv_sr'](latent)  # still B 3C H W
        if not self.D_roll_out_input:
            latent = rearrange(latent, '(b n) c h w->b (n c) h w', n=3)
        else:
            latent = rearrange(latent, 'b c h (n w)->b (n c) h w', n=3)

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        # sr_w_code = self.w_avg
        # assert sr_w_code is not None
        # ret_dict.update(
        #     dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
        #         latent_from_vit.shape[0], 0), ))  # type: ignore

        return ret_dict

    # '''


class RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder(
        RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D
):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        # del skip_lienar
        for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks) // 2:]:
            del blk.skip_linear

    @torch.inference_mode()
    def forward_points(self,
                       planes,
                       points: torch.Tensor,
                       chunk_size: int = 2**16):
        # planes: (N, 3, D', H', W')
        # points: (N, P, 3)
        N, P = points.shape[:2]
        if planes.ndim == 4:
            planes = planes.reshape(
                len(planes),
                3,
                -1,  # ! support background plane
                planes.shape[-2],
                planes.shape[-1])  # BS 96 256 256

        # query triplane in chunks
        outs = []
        for i in trange(0, points.shape[1], chunk_size):
            chunk_points = points[:, i:i + chunk_size]

            # query triplane
            # st()
            chunk_out = self.triplane_decoder.renderer._run_model(  # type: ignore
                planes=planes,
                decoder=self.triplane_decoder.decoder,
                sample_coordinates=chunk_points,
                sample_directions=torch.zeros_like(chunk_points),
                options=self.rendering_kwargs,
            )
            # st()

            outs.append(chunk_out)
            torch.cuda.empty_cache()

        # st()

        # concatenate the outputs
        point_features = {
            k: torch.cat([out[k] for out in outs], dim=1)
            for k in outs[0].keys()
        }
        return point_features

    def triplane_decode_grid(self,
                             vit_decode_out,
                             grid_size,
                             aabb: torch.Tensor = None,
                             **kwargs):
        # planes: (N, 3, D', H', W')
        # grid_size: int

        assert isinstance(vit_decode_out, dict)
        planes = vit_decode_out['latent_after_vit']

        # aabb: (N, 2, 3)
        if aabb is None:
            if 'sampler_bbox_min' in self.rendering_kwargs:
                aabb = torch.tensor([
                    [self.rendering_kwargs['sampler_bbox_min']] * 3,
                    [self.rendering_kwargs['sampler_bbox_max']] * 3,
                ],
                                    device=planes.device,
                                    dtype=planes.dtype).unsqueeze(0).repeat(
                                        planes.shape[0], 1, 1)
            else:  # shapenet dataset, follow eg3d
                aabb = torch.tensor(
                    [  # https://github.com/NVlabs/eg3d/blob/7cf1fd1e99e1061e8b6ba850f91c94fe56e7afe4/eg3d/gen_samples.py#L188
                        [-self.rendering_kwargs['box_warp'] / 2] * 3,
                        [self.rendering_kwargs['box_warp'] / 2] * 3,
                    ],
                    device=planes.device,
                    dtype=planes.dtype).unsqueeze(0).repeat(
                        planes.shape[0], 1, 1)

        assert planes.shape[0] == aabb.shape[
            0], "Batch size mismatch for planes and aabb"
        N = planes.shape[0]

        # create grid points for triplane query
        grid_points = []
        for i in range(N):
            grid_points.append(
                torch.stack(torch.meshgrid(
                    torch.linspace(aabb[i, 0, 0],
                                   aabb[i, 1, 0],
                                   grid_size,
                                   device=planes.device),
                    torch.linspace(aabb[i, 0, 1],
                                   aabb[i, 1, 1],
                                   grid_size,
                                   device=planes.device),
                    torch.linspace(aabb[i, 0, 2],
                                   aabb[i, 1, 2],
                                   grid_size,
                                   device=planes.device),
                    indexing='ij',
                ),
                            dim=-1).reshape(-1, 3))
        cube_grid = torch.stack(grid_points, dim=0).to(planes.device)  # 1 N 3
        # st()

        features = self.forward_points(planes, cube_grid)

        # reshape into grid
        features = {
            k: v.reshape(N, grid_size, grid_size, grid_size, -1)
            for k, v in features.items()
        }

        # st()

        return features

    def create_fusion_blks(self, fusion_blk_depth, use_fusion_blk, fusion_blk):
        # no need to fuse anymore
        pass

    def forward_vit_decoder(self, x, img_size=None):
        # st()
        return self.vit_decoder(x)

    def vit_decode_backbone(self, latent, img_size):
        return super().vit_decode_backbone(latent, img_size)

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):
        return super().vit_decode_postprocess(latent_from_vit, ret_dict)

    # def vae_reparameterization(self, latent, sample_posterior):
    #     return super().vae_reparameterization(latent, sample_posterior)


class RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder_L(
        RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder
):
    # larger D
    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        # self.superresolution.update(
        #     dict(conv_sr=Decoder(  # serve as Deconv
        #         resolution=128,
        #         # resolution=256,
        #         in_channels=3,
        #         # ch=64,
        #         ch=32,
        #         # ch=16,
        #         ch_mult=[1, 2, 2, 4],
        #         # ch_mult=[1, 1, 2, 2],
        #         num_res_blocks=2,
        #         # ch_mult=[1,2,4],
        #         # num_res_blocks=0,
        #         # num_res_blocks=1,
        #         dropout=0.0,
        #         attn_resolutions=[],
        #         out_ch=32,
        #         # z_channels=vit_decoder.embed_dim//4,
        #         z_channels=vit_decoder.embed_dim, # 768 or 1024, depends on the VIT arch
        #         # z_channels=vit_decoder.embed_dim//2,
        #     ),
        #     # after_vit_upsampler=Upsample2D(channels=vit_decoder.embed_dim,use_conv=True, use_conv_transpose=False, out_channels=vit_decoder.embed_dim//2)
        #     ))
        self.D_roll_out_input = True

    def triplane_decode(self,
                        vit_decode_out,
                        c,
                        return_raw_only=False,
                        **kwargs):
        return super().triplane_decode(vit_decode_out, c, return_raw_only,
                                       **kwargs)


class RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder_S(
        RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder_L
):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)
        self.D_roll_out_input = False

        # self.rendering_kwargs['sampler_bbox_min']
        # self.rendering_kwargs['sampler_bbox_max']

        # self.superresolution.update(
        #     dict(conv_sr=NearestConvSR(  # serve as Deconv
        #         output_dim=vit_decoder.embed_dim,
        #         num_out_ch=32,
        #     )))

    # def vae_reparameterization(self, latent, sample_posterior):
    #     return super().vae_reparameterization(latent, sample_posterior)


# ! flexicube port


class flexicube_finetune(
        RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder_S
):
    # add flexicube rendering pipeline
    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        # ! TODO, change FOV to the right value
        # g_buffer_objv_FoVy = 0.6911112401860215 * 180
        g_buffer_focal = 1.3888887
        self.init_flexicubes_geometry(dist_util.dev(),
                                      fovy=None,
                                      focal=g_buffer_focal)  # ! TODO, fovy
        self.grid_res = 128
        self.grid_scale = 2.1  # what this for?
        self.synthesizer = self.triplane_decoder  # for compat issue

    def init_flexicubes_geometry(self, device, fovy=50.0, focal=None):
        camera = PerspectiveCamera(fovy=fovy, focal=focal, device=device)
        renderer = NeuralRender(device, camera_model=camera)
        self.geometry = FlexiCubesGeometry(
            grid_res=self.grid_res,
            scale=self.grid_scale,  # TODO also requires update.
            renderer=renderer,
            render_type='neural_render',
            device=dist_util.dev(),
        )

    def get_sdf_deformation_prediction(self, planes):
        '''
        Predict SDF and deformation for tetrahedron vertices
        :param planes: triplane feature map for the geometry
        '''
        init_position = self.geometry.verts.unsqueeze(0).expand(
            planes.shape[0], -1, -1)

        # Step 1: predict the SDF and deformation
        sdf, deformation, weight = torch.utils.checkpoint.checkpoint(
            self.synthesizer.get_geometry_prediction,
            planes,
            init_position,
            self.geometry.indices,
            use_reentrant=False,
        )

        # Step 2: Normalize the deformation to avoid the flipped triangles.
        deformation = 1.0 / (self.grid_res * self.deformation_multiplier
                             ) * torch.tanh(deformation)
        sdf_reg_loss = torch.zeros(sdf.shape[0],
                                   device=sdf.device,
                                   dtype=torch.float32)

        ####
        # Step 3: Fix some sdf if we observe empty shape (full positive or full negative)
        sdf_bxnxnxn = sdf.reshape((sdf.shape[0], self.grid_res + 1,
                                   self.grid_res + 1, self.grid_res + 1))
        sdf_less_boundary = sdf_bxnxnxn[:, 1:-1, 1:-1,
                                        1:-1].reshape(sdf.shape[0], -1)
        pos_shape = torch.sum((sdf_less_boundary > 0).int(), dim=-1)
        neg_shape = torch.sum((sdf_less_boundary < 0).int(), dim=-1)
        zero_surface = torch.bitwise_or(pos_shape == 0, neg_shape == 0)
        if torch.sum(zero_surface).item() > 0:
            update_sdf = torch.zeros_like(sdf[0:1])
            max_sdf = sdf.max()
            min_sdf = sdf.min()
            update_sdf[:,
                       self.geometry.center_indices] += (1.0 - min_sdf
                                                         )  # greater than zero
            update_sdf[:, self.geometry.boundary_indices] += (
                -1 - max_sdf)  # smaller than zero
            new_sdf = torch.zeros_like(sdf)
            for i_batch in range(zero_surface.shape[0]):
                if zero_surface[i_batch]:
                    new_sdf[i_batch:i_batch + 1] += update_sdf
            update_mask = (new_sdf == 0).float()
            # Regulraization here is used to push the sdf to be a different sign (make it not fully positive or fully negative)
            sdf_reg_loss = torch.abs(sdf).mean(dim=-1).mean(dim=-1)
            sdf_reg_loss = sdf_reg_loss * zero_surface.float()
            sdf = sdf * update_mask + new_sdf * (1 - update_mask)

        # Step 4: Here we remove the gradient for the bad sdf (full positive or full negative)
        final_sdf = []
        final_def = []
        for i_batch in range(zero_surface.shape[0]):
            if zero_surface[i_batch]:
                final_sdf.append(sdf[i_batch:i_batch + 1].detach())
                final_def.append(deformation[i_batch:i_batch + 1].detach())
            else:
                final_sdf.append(sdf[i_batch:i_batch + 1])
                final_def.append(deformation[i_batch:i_batch + 1])
        sdf = torch.cat(final_sdf, dim=0)
        deformation = torch.cat(final_def, dim=0)
        return sdf, deformation, sdf_reg_loss, weight

    def get_geometry_prediction(self, planes=None):
        '''
        Function to generate mesh with give triplanes
        :param planes: triplane features
        '''
        # Step 1: first get the sdf and deformation value for each vertices in the tetrahedon grid.
        sdf, deformation, sdf_reg_loss, weight = self.get_sdf_deformation_prediction(
            planes)
        v_deformed = self.geometry.verts.unsqueeze(dim=0).expand(
            sdf.shape[0], -1, -1) + deformation
        tets = self.geometry.indices
        n_batch = planes.shape[0]
        v_list = []
        f_list = []
        flexicubes_surface_reg_list = []

        # Step 2: Using marching tet to obtain the mesh
        for i_batch in range(n_batch):
            verts, faces, flexicubes_surface_reg = self.geometry.get_mesh(
                v_deformed[i_batch],
                sdf[i_batch].squeeze(dim=-1),
                with_uv=False,
                indices=tets,
                weight_n=weight[i_batch].squeeze(dim=-1),
                is_training=self.training,
            )
            flexicubes_surface_reg_list.append(flexicubes_surface_reg)
            v_list.append(verts)
            f_list.append(faces)

        flexicubes_surface_reg = torch.cat(flexicubes_surface_reg_list).mean()
        flexicubes_weight_reg = (weight**2).mean()

        return v_list, f_list, sdf, deformation, v_deformed, (
            sdf_reg_loss, flexicubes_surface_reg, flexicubes_weight_reg)

    def get_texture_prediction(self, planes, tex_pos, hard_mask=None):
        '''
        Predict Texture given triplanes
        :param planes: the triplane feature map
        :param tex_pos: Position we want to query the texture field
        :param hard_mask: 2D silhoueete of the rendered image
        '''
        tex_pos = torch.cat(tex_pos, dim=0)
        if not hard_mask is None:
            tex_pos = tex_pos * hard_mask.float()
        batch_size = tex_pos.shape[0]
        tex_pos = tex_pos.reshape(batch_size, -1, 3)
        ###################
        # We use mask to get the texture location (to save the memory)
        if hard_mask is not None:
            n_point_list = torch.sum(hard_mask.long().reshape(
                hard_mask.shape[0], -1),
                                     dim=-1)
            sample_tex_pose_list = []
            max_point = n_point_list.max()
            expanded_hard_mask = hard_mask.reshape(batch_size, -1,
                                                   1).expand(-1, -1, 3) > 0.5
            for i in range(tex_pos.shape[0]):
                tex_pos_one_shape = tex_pos[i][expanded_hard_mask[i]].reshape(
                    1, -1, 3)
                if tex_pos_one_shape.shape[1] < max_point:
                    tex_pos_one_shape = torch.cat([
                        tex_pos_one_shape,
                        torch.zeros(1,
                                    max_point - tex_pos_one_shape.shape[1],
                                    3,
                                    device=tex_pos_one_shape.device,
                                    dtype=torch.float32)
                    ],
                                                  dim=1)
                sample_tex_pose_list.append(tex_pos_one_shape)
            tex_pos = torch.cat(sample_tex_pose_list, dim=0)

        tex_feat = torch.utils.checkpoint.checkpoint(
            self.synthesizer.get_texture_prediction,
            planes,
            tex_pos,
            use_reentrant=False,
        )

        if hard_mask is not None:
            final_tex_feat = torch.zeros(planes.shape[0],
                                         hard_mask.shape[1] *
                                         hard_mask.shape[2],
                                         tex_feat.shape[-1],
                                         device=tex_feat.device)
            expanded_hard_mask = hard_mask.reshape(
                hard_mask.shape[0], -1,
                1).expand(-1, -1, final_tex_feat.shape[-1]) > 0.5
            for i in range(planes.shape[0]):
                final_tex_feat[i][expanded_hard_mask[i]] = tex_feat[
                    i][:n_point_list[i]].reshape(-1)
            tex_feat = final_tex_feat

        return tex_feat.reshape(planes.shape[0], hard_mask.shape[1],
                                hard_mask.shape[2], tex_feat.shape[-1])

    def render_mesh(self, mesh_v, mesh_f, cam_mv, render_size=256):
        '''
        Function to render a generated mesh with nvdiffrast
        :param mesh_v: List of vertices for the mesh
        :param mesh_f: List of faces for the mesh
        :param cam_mv:  4x4 rotation matrix
        :return:
        '''
        return_value_list = []
        for i_mesh in range(len(mesh_v)):
            return_value = self.geometry.render_mesh(mesh_v[i_mesh],
                                                     mesh_f[i_mesh].int(),
                                                     cam_mv[i_mesh],
                                                     resolution=render_size,
                                                     hierarchical_mask=False)
            return_value_list.append(return_value)

        return_keys = return_value_list[0].keys()
        return_value = dict()
        for k in return_keys:
            value = [v[k] for v in return_value_list]
            return_value[k] = value

        mask = torch.cat(return_value['mask'], dim=0)
        hard_mask = torch.cat(return_value['hard_mask'], dim=0)
        tex_pos = return_value['tex_pos']
        depth = torch.cat(return_value['depth'], dim=0)
        normal = torch.cat(return_value['normal'], dim=0)
        return mask, hard_mask, tex_pos, depth, normal

    def forward_geometry(self, planes, render_cameras, render_size=256):
        '''
        Main function of our Generator. It first generate 3D mesh, then render it into 2D image
        with given `render_cameras`.
        :param planes: triplane features
        :param render_cameras: cameras to render generated 3D shape
        '''
        B, NV = render_cameras.shape[:2]

        # Generate 3D mesh first
        mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(
            planes)

        st()

        # Render the mesh into 2D image (get 3d position of each image plane)
        cam_mv = render_cameras
        run_n_view = cam_mv.shape[1]
        antilias_mask, hard_mask, tex_pos, depth, normal = self.render_mesh(
            mesh_v, mesh_f, cam_mv, render_size=render_size)

        tex_hard_mask = hard_mask
        tex_pos = [
            torch.cat([pos[i_view:i_view + 1] for i_view in range(run_n_view)],
                      dim=2) for pos in tex_pos
        ]
        tex_hard_mask = torch.cat([
            torch.cat([
                tex_hard_mask[i * run_n_view + i_view:i * run_n_view + i_view +
                              1] for i_view in range(run_n_view)
            ],
                      dim=2) for i in range(planes.shape[0])
        ],
                                  dim=0)

        # Querying the texture field to predict the texture feature for each pixel on the image
        tex_feat = self.get_texture_prediction(planes, tex_pos, tex_hard_mask)
        background_feature = torch.ones_like(tex_feat)  # white background

        # Merge them together
        img_feat = tex_feat * tex_hard_mask + background_feature * (
            1 - tex_hard_mask)

        # We should split it back to the original image shape
        img_feat = torch.cat([
            torch.cat([
                img_feat[i:i + 1, :, render_size * i_view:render_size *
                         (i_view + 1)] for i_view in range(run_n_view)
            ],
                      dim=0) for i in range(len(tex_pos))
        ],
                             dim=0)

        img = img_feat.clamp(0, 1).permute(0, 3, 1, 2).unflatten(0, (B, NV))
        antilias_mask = antilias_mask.permute(0, 3, 1, 2).unflatten(0, (B, NV))
        depth = -depth.permute(0, 3, 1, 2).unflatten(
            0, (B, NV))  # transform negative depth to positive
        normal = normal.permute(0, 3, 1, 2).unflatten(0, (B, NV))

        out = {
            'img': img,
            'mask': antilias_mask,
            'depth': depth,
            'normal': normal,
            'sdf': sdf,
            'mesh_v': mesh_v,
            'mesh_f': mesh_f,
            'sdf_reg_loss': sdf_reg_loss,
        }
        return out

    # ! TODO, requires fusion with the current forward
    # def forward(self, images, cameras, render_cameras, render_size: int):
    #     # images: [B, V, C_img, H_img, W_img]
    #     # cameras: [B, V, 16]
    #     # render_cameras: [B, M, D_cam_render]
    #     # render_size: int
    #     B, M = render_cameras.shape[:2]

    #     planes = self.forward_planes(images, cameras)
    #     out = self.forward_geometry(planes, render_cameras, render_size=render_size)
    #     st()

    #     return {
    #         'planes': planes,
    #         **out
    #     }

    def extract_mesh(
        self,
        planes: torch.Tensor,
        use_texture_map: bool = False,
        texture_resolution: int = 1024,
        **kwargs,
    ):
        '''
        Extract a 3D mesh from FlexiCubes. Only support batch_size 1.
        :param planes: triplane features
        :param use_texture_map: use texture map or vertex color
        :param texture_resolution: the resolution of texure map
        '''
        assert planes.shape[0] == 1
        device = planes.device

        # predict geometry first
        mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(
            planes)
        vertices, faces = mesh_v[0], mesh_f[0]

        if not use_texture_map:
            # query vertex colors
            vertices_tensor = vertices.unsqueeze(0)
            vertices_colors = self.synthesizer.get_texture_prediction(
                planes, vertices_tensor).clamp(0, 1).squeeze(0).cpu().numpy()
            vertices_colors = (vertices_colors * 255).astype(np.uint8)

            return vertices.cpu().numpy(), faces.cpu().numpy(), vertices_colors

        # use x-atlas to get uv mapping for the mesh
        ctx = dr.RasterizeCudaContext(device=device)
        uvs, mesh_tex_idx, gb_pos, tex_hard_mask = xatlas_uvmap(
            self.geometry.renderer.ctx,
            vertices,
            faces,
            resolution=texture_resolution)
        tex_hard_mask = tex_hard_mask.float()

        # query the texture field to get the RGB color for texture map
        tex_feat = self.get_texture_prediction(planes, [gb_pos], tex_hard_mask)
        background_feature = torch.zeros_like(tex_feat)
        img_feat = torch.lerp(background_feature, tex_feat, tex_hard_mask)
        texture_map = img_feat.permute(0, 3, 1, 2).squeeze(0)

        return vertices, faces, uvs, mesh_tex_idx, texture_map

    def triplane_decode(self,
                        vit_decode_out,
                        c,
                        return_raw_only=False,
                        **kwargs):
        # ! call rendering here

        out = self.forward_geometry(vit_decode_out['latent_after_vit'],
                                    c,
                                    render_size=256)
        st()
# 
        return {**vit_decode_out, **out}

    # def triplane_decode(self, vit_decode_out, c, return_raw_only=False):

    #     assert isinstance(vit_decode_out, dict)
    #     # latent_after_vit, bg_plane = (vit_decode_out.get(k, None)
    #     #                                 for k in ('latent_after_vit',
    #     #                                             'bg_plane'))

    #     # * triplane rendering
    #     ret_dict = self.triplane_decoder(
    #         vit_decode_out['latent_after_vit'],
    #         vit_decode_out['bg_plane'],
    #         c,
    #         return_raw_only=return_raw_only)  # triplane latent -> imgs
    #     # st()
    #     ret_dict.update({
    #         # 'latent_after_vit': vit_decode_out['latent_after_vit'],
    #         **vit_decode_out
    #     })

    #     return ret_dict


# ! flexicube port done


# srt tokenizer
class srt_tokenizer(
        RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder_S
):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)
        self.D_roll_out_input = False

        self.superresolution.update(
            dict(quant_conv=nn.Conv2d(
                2 * self.ldm_z_channels,
                2 * self.ldm_embed_dim,
                kernel_size=1,  # just MLP
                groups=1)))

    def vae_reparameterization(self, latent, sample_posterior):
        return super().vae_reparameterization(latent, sample_posterior)

    def vae_encode(self, h):
        # * smooth convolution before triplane
        B, C, H, W = h.shape  # C=24 here
        moments = self.superresolution['quant_conv'](
            h)  # Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), groups=3)
        moments = rearrange(moments,
                            'B C (N H) W -> B C N H W',
                            N=self.plane_n,
                            B=B,
                            H=W,
                            W=W)

        moments = moments.flatten(-2)  # B C 3 L

        posterior = DiagonalGaussianDistribution(
            moments, soft_clamp=self.reparameterization_soft_clamp)
        return posterior


# ============ 3dgs render ============
class splatting_v0(
        RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder_S
):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        # ! modify decoder to output 3dgs parameters
        self.superresolution.update(
            dict(
                conv_sr=Decoder(  # serve as Deconv
                    resolution=128,
                    in_channels=3,
                    ch=32,
                    ch_mult=[1, 2, 2, 4],
                    num_res_blocks=1,
                    dropout=0.0,
                    attn_resolutions=[],
                    out_ch=14,
                    z_channels=vit_decoder.embed_dim,
                ), ))
        self.D_roll_out_input = False

        self.superresolution.update({
            'before_gs_conv':
            nn.Conv2d(14, 14, kernel_size=1),
        })

        # ! add actiations
        self.pos_act = lambda x: x.clamp(-0.45, 0.45)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        # self.scale_act = lambda x: (0.01 * F.softplus(x)).clamp(0, 0.45*0.1) # avoid scaling larger than 0.1 * scene_extent
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = F.normalize
        # lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: 0.5 * torch.tanh(
            x) + 0.5  # NOTE: may use sigmoid if train again

        # ! renderer
        # self.gs = GaussianRenderer(opt)
        self.gs = triplane_decoder  # compat

    def forward_gaussians(self, ret_after_decoder, c=None):
        # ! modified from https://github.com/3DTopia/LGM/blob/main/core/models.py
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        # st()
        x = ret_after_decoder['latent_after_vit']
        B, VC, H, W = x.shape  # B (V C) H W
        V = VC // 14
        x = rearrange(x, 'b (v c) h w-> (b v) c h w', v=V)

        # B, V, C, H, W = images.shape
        # images = images.view(B*V, C, H, W)

        # x = self.unet(images) # [B*V, 14, h, w]
        x = self.superresolution['before_gs_conv'](x)  # [B*v, 14, h, w]

        x = x.reshape(B, V, 14, H, W)  # TODO

        ## visualize multi-view gaussian features for plotting figure
        # tmp_alpha = self.opacity_act(x[0, :, 3:4])
        # tmp_img_rgb = self.rgb_act(x[0, :, 11:]) * tmp_alpha + (1 - tmp_alpha)
        # tmp_img_pos = self.pos_act(x[0, :, 0:3]) * 0.5 + 0.5
        # kiui.vis.plot_image(tmp_img_rgb, save=True)
        # kiui.vis.plot_image(tmp_img_pos, save=True)

        # save points for visualization
        # if True:
        if False:
            unmerged_pos = self.pos_act(x[:, :, 0:3,
                                          ...]).permute(0, 1, 3, 4,
                                                        2)  # B V H W 3
            unmerged_pos = unmerged_pos.reshape(B, V, -1, 3)

            unmerged_rgb = self.rgb_act(x[:, :, 11:,
                                          ...]).permute(0, 1, 3, 4,
                                                        2)  # B V H W 3
            unmerged_rgb = unmerged_rgb.reshape(B, V, -1, 3)

            for b in range(B):
                for v in range(V):
                    # pcu.save_mesh_vc(f'tmp/dust3r/lambda100/add3dsupp-{b}-{v}.ply',
                    pcu.save_mesh_vc(f'tmp/lambda50/add3dsupp-{b}-{v}.ply',
                                     unmerged_pos[b][v].detach().cpu().numpy(),
                                     unmerged_rgb[b][v].detach().cpu().numpy())

        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)  # B

        pos = self.pos_act(x[..., 0:3])  # [B, N, 3] # ! better act?
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs],
                              dim=-1)  # [B, N, 14]

        # ! get per-view pos prediction
        per_view_pos = rearrange(pos, 'B (V N) C -> (B V) N C', B=B, V=V)

        ret_after_decoder.update({
            'gaussians': gaussians,
            'per_view_pos': per_view_pos,
            'pos': pos
        })
        # ret_after_decoder.update({'gaussians': gaussians})

        return ret_after_decoder

    def triplane_decode(self,
                        ret_after_gaussian_forward,
                        c,
                        bg_color=None,
                        **kwargs):
        # ! for compat, should be gaussian_render

        data = c
        # always use white bg
        # bg_color = torch.ones(3, dtype=torch.float32, device=dist_util.dev())

        # data = {} # TODO, align the camera format

        # use the other views for rendering and supervision

        # with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU,], record_shapes=True) as prof:
        #     # ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     with record_function("self.gs.render"):

        # st()
        results = self.gs.render(
            ret_after_gaussian_forward['gaussians'],  #  type: ignore
            data['cam_view'],
            data['cam_view_proj'],
            data['cam_pos'],
            tanfov=data['tanfov'],
            bg_color=bg_color,
        )

        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        #  bg_color=bg_color)
        # pred_images = results['image'] # [B, V, C, output_size, output_size]
        # pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]

        results['image_raw'] = results[
            'image'] * 2 - 1  # [0,1] -> [-1,1], match tradition
        results['image_depth'] = results['depth']
        results['image_mask'] = results['alpha']

        # ! vis
        # B, V = results['image_raw'].shape[:2]
        # for b in range(B):
        #     torchvision.utils.save_image(results['image_raw'][b],
        #                                 #  f'tmp/vis-{b}.jpg',
        #                                 #  f'tmp/dust3r/add3dsupp-{b}.jpg',
        #                                  f'tmp/lambda50/add3dsupp-{b}.jpg',
        #                                  normalize=True,
        #                                  value_range=(-1, 1))
        # st()

        return results

    def vit_decode(self, latent, img_size, c=None, sample_posterior=True):
        ret_after_decoder = super().vit_decode(latent, img_size,
                                               sample_posterior)
        # st()
        return self.forward_gaussians(ret_after_decoder, c=c)

    # def forward(self, latent, c, img_size):
    #     ret_after_decoder = self.vit_decode(latent, img_size)
    #     # ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))
    #     return self.gs_render(ret_after_decoder, {'c': c})

    def transform_rotations(self, rotations, source_cv2wT_quat):
        """
        Applies a transform that rotates the predicted rotations from 
        camera space to world space.
        Args:
            rotations: predicted in-camera rotation quaternions (B x N x 4)
            source_cameras_to_world: transformation quaternions from 
                camera-to-world matrices transposed(B x 4)
        Retures:
            rotations with appropriately applied transform to world space
        """

        Mq = source_cv2wT_quat.unsqueeze(1).expand(*rotations.shape)

        rotations = quaternion_raw_multiply(Mq, rotations)

        return rotations


class splatting_v1(splatting_v0):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            plane_n=4,
            # vae_dit_token_size=16,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        self.plane_n = plane_n

        self.vit_decoder.pos_embed = nn.Parameter(
            torch.zeros(1,
                        self.plane_n * (self.token_size**2 + self.cls_token),
                        vit_decoder.embed_dim))

        self.init_weights()

        # ! use group=4
        # TODO, ablate group conv
        self.superresolution.update(
            dict(
                quant_conv=nn.Conv2d(
                    2 * self.plane_n * self.ldm_z_channels,
                    2 * self.ldm_embed_dim * self.plane_n,
                    kernel_size=1,
                    groups=self.plane_n,
                ),
                # ldm_upsample=PatchEmbed(
                #     self.vae_p * self.token_size,
                #     self.vae_p,
                #     self.plane_n * self.ldm_embed_dim,  # B 3 L C
                #     vit_decoder.embed_dim,
                #     bias=True)))  # group=1
                ldm_upsample=PatchEmbedTriplane(
                    self.vae_p * self.token_size,
                    self.vae_p,
                    self.plane_n * self.ldm_embed_dim,  # B 3 L C
                    vit_decoder.embed_dim,
                    bias=True,
                    plane_n=self.plane_n,
                )))  # group=1

    def forward_vit_decoder(self, x, img_size=None):
        return super().forward_vit_decoder(x, img_size)

    def init_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:

        self.vit_decoder.pos_embed = nn.Parameter(
            torch.zeros(1,
                        self.plane_n * (self.token_size**2 + self.cls_token),
                        self.vit_decoder.embed_dim))

        p = self.token_size
        D = self.vit_decoder.pos_embed.shape[-1]
        grid_size = (self.plane_n * p, p)
        pos_embed = get_2d_sincos_pos_embed(D, grid_size).reshape(
            self.plane_n * p * p, D)  # H*W, D
        self.vit_decoder.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))
        logger.log('init pos_embed with sincos')

    # ''' # for SD Decoder
    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        def unflatten_token(x, p=None):
            B, L, C = x.shape
            x = x.reshape(B, self.plane_n, L // self.plane_n, C)

            if self.cls_token:  # TODO, how to better use cls token
                x = x[:, :, 1:]  # B 3 256 C

            h = w = int((x.shape[2])**.5)
            assert h * w == x.shape[2]

            if p is None:
                x = x.reshape(shape=(B, self.plane_n, h, w, -1))
                if not self.D_roll_out_input:
                    x = rearrange(
                        x, 'b n h w c->(b n) c h w'
                    )  # merge plane into Batch and prepare for rendering
                else:
                    x = rearrange(
                        x, 'b n h w c->b c h (n w)'
                    )  # merge plane into Batch and prepare for rendering
            else:
                x = x.reshape(shape=(B, self.plane_n, h, w, p, p, -1))
                if self.D_roll_out_input:
                    x = rearrange(
                        x, 'b n h w p1 p2 c->b c (h p1) (n w p2)'
                    )  # merge plane into Batch and prepare for rendering
                else:
                    x = rearrange(
                        x, 'b n h w p1 p2 c->(b n) c (h p1) (w p2)'
                    )  # merge plane into Batch and prepare for rendering

            return x

        latent = unflatten_token(
            latent_from_vit)  # B 3 h w vit_decoder.embed_dim

        # ! x2 upsapmle, 16 -32 before sending into SD Decoder
        # latent = self.superresolution['after_vit_upsampler'](latent) # B*3 192 32 32

        # latent = unflatten_token(latent_from_vit, p=2)

        # ! SD SR
        latent = self.superresolution['conv_sr'](latent)  # still B 3C H W
        if not self.D_roll_out_input:
            latent = rearrange(latent,
                               '(b n) c h w->b (n c) h w',
                               n=self.plane_n)
        else:
            latent = rearrange(latent,
                               'b c h (n w)->b (n c) h w',
                               n=self.plane_n)

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        # sr_w_code = self.w_avg
        # assert sr_w_code is not None
        # ret_dict.update(
        #     dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
        #         latent_from_vit.shape[0], 0), ))  # type: ignore

        return ret_dict


class splatting_v2(splatting_v1):
    # pred depth, pixel-aligned gaussian; no canonical alignment, view transform independently
    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        # https://github.com/szymanowiczs/splatter-image/blob/98b465731c3273bf8f42a747d1b6ce1a93faf3d6/scene/gaussian_predictor.py#L532
        self.depth_act = nn.Sigmoid()  # follow splatter-image

        # ! modify decoder to output 3dgs parameters
        self.superresolution.update(
            dict(
                before_gs_conv=nn.Identity(),
                conv_sr=Decoder(  # serve as Deconv
                    resolution=128,
                    in_channels=3,
                    ch=32,
                    ch_mult=[1, 2, 2, 4],
                    num_res_blocks=1,
                    dropout=0.0,
                    attn_resolutions=[],
                    out_ch=12,  # predict depth, 14-2
                    z_channels=vit_decoder.embed_dim,
                ),
            ))

        self.ray_sampler = RaySampler()

    def forward_gaussians(self, ret_after_decoder, c):
        # ! modified from https://github.com/3DTopia/LGM/blob/main/core/models.py
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        # st()
        # w2c = c['orig_w2c']
        x = ret_after_decoder['latent_after_vit']
        B, VC, H, W = x.shape  # B (V C) H W
        # st()
        V = VC // 12
        x = rearrange(x, 'b (v c) h w-> (b v) c h w', v=V)

        # B, V, C, H, W = images.shape
        # images = images.view(B*V, C, H, W)

        # x = self.unet(images) # [B*V, 14, h, w]
        x = self.superresolution['before_gs_conv'](x)  # [B*v, 14, h, w]

        x = x.reshape(B, V, 12, H, W)  # TODO

        # ! no merge V required
        # x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 12)  # B
        x = x.permute(0, 1, 3, 4, 2).reshape(B * V, -1,
                                             12)  # B V H W C -> BV HW C

        # pos = self.pos_act(x[..., 0:3])  # [B, N, 3] # ! better act?
        # depth = self.depth_act(x[..., 0:1])  # [B, N, 3] # ! better act?

        # ! normalize to near, far.
        # origin_distances=0 here, all normalized into [-0.45, 0.45] # https://github.com/szymanowiczs/splatter-image/blob/98b465731c3273bf8f42a747d1b6ce1a93faf3d6/scene/gaussian_predictor.py#L741
        depth = self.depth_act(x[..., 0:1])
        depth = self.rendering_kwargs['z_near'] + depth * (
            self.rendering_kwargs['z_far'] - self.rendering_kwargs['z_near']
        )  # [B, N, 3] # ! better act?
        opacity = self.opacity_act(x[..., 1:2])
        scale = self.scale_act(x[..., 2:5])
        rotation = self.rot_act(x[..., 5:9])
        rgbs = self.rgb_act(x[..., 9:])

        # ! convert depth to pos

        cam2world_matrix = c['orig_c2w'][:, :, :16].reshape(B * V, 4, 4)
        intrinsics = c['orig_pose'][:, :, 16:25].reshape(B * V, 3, 3)

        # rotation = torch.bmm(cam2world_matrix, rotation) # transform rotation into the world space. How?
        source_cv2wT_quat = c['source_cv2wT_quat']
        source_cv2wT_quat = source_cv2wT_quat.reshape(
            B * V, *source_cv2wT_quat.shape[2:])
        rotation = self.transform_rotations(
            rotation,  # ! transform rotations to the world space
            source_cv2wT_quat=source_cv2wT_quat)

        # ! already in the world space after ray_sampler()
        ray_origins, ray_directions = self.ray_sampler(  # shape: 
            cam2world_matrix, intrinsics, H)[:2]
        # self.gs.output_size,)[:2]
        pos = ray_origins + depth * ray_directions  # BV HW 3, already in the world space

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs],
                              dim=-1)  # [B, N, 14]

        gaussians = rearrange(gaussians, '(B V) N C -> B (V N) C', B=B,
                              V=V)  # merge V back to N

        if False:
            # if True:

            unmerged_pos = pos.reshape(B, V, -1, 3)
            unmerged_rgb = rgbs.reshape(B, V, -1, 3)

            for b in range(B):
                for v in range(V):
                    pcu.save_mesh_vc(f'tmp/splatting_v2-{b}-{v}.ply',
                                     unmerged_pos[b][v].detach().cpu().numpy(),
                                     unmerged_rgb[b][v].detach().cpu().numpy())
                    # pcu.save_mesh_v(f'tmp/splatting_v2-{b}-{v}.ply',

        # st()

        ret_after_decoder.update({'gaussians': gaussians})

        return ret_after_decoder

    # def vit_decode(self, latent, img_size, c, sample_posterior=True):
    #     ret_after_decoder = super().vit_decode(latent, img_size,
    #                                            sample_posterior)
    #     return self.forward_gaussians(ret_after_decoder, c)

    def forward(self, latent, c, img_size):

        latent_normalized = self.vit_decode(latent, img_size, c)
        return self.triplane_decode(latent_normalized, c)


class splatting_v3(splatting_v2):
    # ! DUST3R mode, using frame-1 as the canonical frame.
    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

    def forward_gaussians(self, ret_after_decoder, c):
        # ! modified from https://github.com/3DTopia/LGM/blob/main/core/models.py
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        # st()
        # w2c = c['orig_w2c']
        x = ret_after_decoder['latent_after_vit']
        B, VC, H, W = x.shape  # B (V C) H W
        V = VC // 12
        x = rearrange(x, 'b (v c) h w-> (b v) c h w', v=V)

        # B, V, C, H, W = images.shape
        # images = images.view(B*V, C, H, W)

        # x = self.unet(images) # [B*V, 14, h, w]
        x = self.superresolution['before_gs_conv'](x)  # [B*v, 14, h, w]

        x = x.reshape(B, V, 12, H, W)  # TODO

        # ! no merge V required
        # x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 12)  # B
        x = x.permute(0, 1, 3, 4, 2).reshape(B * V, -1,
                                             12)  # B V H W C -> BV HW C

        # pos = self.pos_act(x[..., 0:3])  # [B, N, 3] # ! better act?
        # depth = self.depth_act(x[..., 0:1])  # [B, N, 3] # ! better act?

        # ! normalize to near, far.
        # origin_distances=0 here, all normalized into [-0.45, 0.45] # https://github.com/szymanowiczs/splatter-image/blob/98b465731c3273bf8f42a747d1b6ce1a93faf3d6/scene/gaussian_predictor.py#L741
        depth = self.depth_act(x[..., 0:1])
        depth = self.rendering_kwargs['z_near'] + depth * (
            self.rendering_kwargs['z_far'] - self.rendering_kwargs['z_near']
        )  # [B, N, 3] # ! better act?
        opacity = self.opacity_act(x[..., 1:2])
        scale = self.scale_act(x[..., 2:5])
        rotation = self.rot_act(x[..., 5:9])
        rgbs = self.rgb_act(x[..., 9:])

        # ! convert depth to pos
        # ! all aligned in frame 1canonical space
        cam2world_matrix = c['orig_c2w'][:, 0:1, :16].repeat_interleave(
            V, dim=1).reshape(B * V, 4, 4)
        # intrinsics = c['orig_pose'][:, 0:1, 16:25].repeat_interleave(B*V, 3, 3) # intrinsics are the same
        intrinsics = c['orig_pose'][:, 0:1,
                                    16:25].repeat_interleave(V, dim=1).reshape(
                                        B * V, 3, 3)

        # rotation = torch.bmm(cam2world_matrix, rotation) # transform rotation into the world space. How?
        source_cv2wT_quat = c['source_cv2wT_quat']
        source_cv2wT_quat = source_cv2wT_quat.reshape(
            B * V, *source_cv2wT_quat.shape[2:])
        rotation = self.transform_rotations(
            rotation,  # ! transform rotations to the world space
            source_cv2wT_quat=source_cv2wT_quat)

        # ! already in the world space after ray_sampler()
        ray_origins, ray_directions = self.ray_sampler(  # shape: 
            cam2world_matrix, intrinsics, H)[:2]
        # self.gs.output_size,)[:2]
        pos = ray_origins + depth * ray_directions  # BV HW 3, already in the world space

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs],
                              dim=-1)  # [B, N, 14]

        gaussians = rearrange(gaussians, '(B V) N C -> B (V N) C', B=B,
                              V=V)  # merge V back to N

        if False:
            # if True:

            unmerged_pos = pos.reshape(B, V, -1, 3)
            unmerged_rgb = rgbs.reshape(B, V, -1, 3)

            for b in range(B):
                for v in range(V):
                    pcu.save_mesh_vc(f'tmp/splatting_v2-{b}-{v}.ply',
                                     unmerged_pos[b][v].detach().cpu().numpy(),
                                     unmerged_rgb[b][v].detach().cpu().numpy())
                    # pcu.save_mesh_v(f'tmp/splatting_v2-{b}-{v}.ply',

        # st()

        ret_after_decoder.update({'gaussians': gaussians})

        return ret_after_decoder


class splatting_v4_dpt(splatting_v2):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        # ! modify decoder to output 3dgs parameters
        self.superresolution.update(
            dict(
                before_gs_conv=nn.Identity(),
                conv_sr=create_dpt_head_ln3diff(  # serve as Deconv
                    out_nchan=12,
                    # feature_dim=256,  # ? projection dim
                    feature_dim=128,  # ? projection dim
                    l2=len(vit_decoder.blocks),  # type: ignore
                    dec_embed_dim=vit_decoder.embed_dim,
                    patch_size=vit_decoder.patch_size,
                ),
            ))

    # ''' # for SD Decoder
    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        def unflatten_token(x, p=None):
            B, L, C = x.shape
            x = x.reshape(B, 3, L // 3, C)

            if self.cls_token:  # TODO, how to better use cls token
                x = x[:, :, 1:]  # B 3 256 C

            h = w = int((x.shape[2])**.5)
            assert h * w == x.shape[2]

            if p is None:
                x = x.reshape(shape=(B, 3, h, w, -1))
                if not self.D_roll_out_input:
                    x = rearrange(
                        x, 'b n h w c->(b n) c h w'
                    )  # merge plane into Batch and prepare for rendering
                else:
                    x = rearrange(
                        x, 'b n h w c->b c h (n w)'
                    )  # merge plane into Batch and prepare for rendering
            else:
                x = x.reshape(shape=(B, 3, h, w, p, p, -1))
                if self.D_roll_out_input:
                    x = rearrange(
                        x, 'b n h w p1 p2 c->b c (h p1) (n w p2)'
                    )  # merge plane into Batch and prepare for rendering
                else:
                    x = rearrange(
                        x, 'b n h w p1 p2 c->(b n) c (h p1) (w p2)'
                    )  # merge plane into Batch and prepare for rendering

            return x

        # latent = unflatten_token(
        #     latent_from_vit)  # B 3 h w vit_decoder.embed_dim

        # ! x2 upsapmle, 16 -32 before sending into SD Decoder
        # latent = self.superresolution['after_vit_upsampler'](latent) # B*3 192 32 32

        # latent = unflatten_token(latent_from_vit, p=2)

        # ! SD SR
        # ! no flatten required, since DPT will do so.

        h = w = int((latent_from_vit[0].shape[2])**.5)
        latent = self.superresolution['conv_sr'](
            latent_from_vit, img_info=(h, w))  # still B 3C H W

        if not self.D_roll_out_input:
            latent = rearrange(latent,
                               '(b n) c h w->b (n c) h w',
                               n=self.plane_n)
        else:
            latent = rearrange(latent,
                               'b c h (n w)->b (n c) h w',
                               n=self.plane_n)
        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        # sr_w_code = self.w_avg
        # assert sr_w_code is not None
        # ret_dict.update(
        #     dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
        #         latent_from_vit.shape[0], 0), ))  # type: ignore

        return ret_dict


# ! dust3R setting, predict xyz but add dense supervision
class splatting_dust3r_v1(splatting_v1):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        self.decoder_output_dim = 14

    def forward_gaussians(self, ret_after_decoder, c):
        # ! modified from https://github.com/3DTopia/LGM/blob/main/core/models.py
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        # st()
        # w2c = c['orig_w2c']
        x = ret_after_decoder['latent_after_vit']
        B, VC, H, W = x.shape  # B (V C) H W
        V = VC // 14
        x = rearrange(x, 'b (v c) h w-> (b v) c h w', v=V)

        # B, V, C, H, W = images.shape
        # images = images.view(B*V, C, H, W)

        # x = self.unet(images) # [B*V, 14, h, w]
        x = self.superresolution['before_gs_conv'](x)  # [B*v, 14, h, w]

        x = x.reshape(B, V, self.decoder_output_dim, H, W)  # TODO

        # ! no merge V required
        # x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 12)  # B
        x = x.permute(0, 1, 3, 4, 2).reshape(
            B * V, -1, self.decoder_output_dim)  # B V H W C -> BV HW C

        # pos = self.pos_act(x[..., 0:3])  # [B, N, 3] # ! better act?
        # depth = self.depth_act(x[..., 0:1])  # [B, N, 3] # ! better act?

        # ! normalize to near, far.
        # origin_distances=0 here, all normalized into [-0.45, 0.45] # https://github.com/szymanowiczs/splatter-image/blob/98b465731c3273bf8f42a747d1b6ce1a93faf3d6/scene/gaussian_predictor.py#L741

        pos = self.pos_act(x[..., 0:3])  # [B, N, 3] # ! better act?
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        # st()

        # ! convert depth to pos
        # ! all aligned in frame 1canonical space
        cam2world_matrix = c['orig_c2w'][:, 0:1, :16].repeat_interleave(
            V, dim=1).reshape(B * V, 4, 4)
        # # intrinsics = c['orig_pose'][:, 0:1, 16:25].repeat_interleave(B*V, 3, 3) # intrinsics are the same
        # intrinsics = c['orig_pose'][:, 0:1,
        #                             16:25].repeat_interleave(V, dim=1).reshape(
        #                                 B * V, 3, 3)

        # rotation = torch.bmm(cam2world_matrix, rotation) # transform rotation into the world space. How?
        # ! merge to frame 1 as the canonical
        source_cv2wT_quat = c['source_cv2wT_quat'][:, 0:1,
                                                   ...].repeat_interleave(
                                                       V, dim=1)
        # st()
        source_cv2wT_quat = source_cv2wT_quat.reshape(
            B * V, *source_cv2wT_quat.shape[2:])
        rotation = self.transform_rotations(
            rotation,  # ! transform rotations to the world space
            source_cv2wT_quat=source_cv2wT_quat)

        # ! already in the world space after ray_sampler()
        # ray_origins, ray_directions = self.ray_sampler(  # shape:
        #     cam2world_matrix, intrinsics, H)[:2]
        # self.gs.output_size,)[:2]
        # pos = ray_origins + depth * ray_directions  # BV HW 3, already in the world space

        # ! transform frame1 xyz to world space
        pos_het = torch.cat([pos, torch.ones_like(pos[..., 0:1])], dim=-1)
        pos = torch.bmm(cam2world_matrix,
                        pos_het.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]
        # rotation = torch.bmm(cam2world_matrix, rotation) # transform rotation into the world space. How?

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs],
                              dim=-1)  # [B, N, 14]

        gaussians = rearrange(gaussians, '(B V) N C -> B (V N) C', B=B,
                              V=V)  # merge V back to N

        if False:
            # if True:

            unmerged_pos = pos.reshape(B, V, -1, 3)
            unmerged_rgb = rgbs.reshape(B, V, -1, 3)

            for b in range(B):
                for v in range(V):
                    pcu.save_mesh_vc(f'tmp/splatting_v2-{b}-{v}.ply',
                                     unmerged_pos[b][v].detach().cpu().numpy(),
                                     unmerged_rgb[b][v].detach().cpu().numpy())
                    # pcu.save_mesh_v(f'tmp/splatting_v2-{b}-{v}.ply',

        # ! get per-view pos prediction
        per_view_pos = rearrange(pos, '(B V) N C -> B V N C', B=B, V=V)

        ret_after_decoder.update({
            'gaussians': gaussians,
            'per_view_pos': per_view_pos
        })

        return ret_after_decoder


# ! remove conv decoder, direct DiT prediction
class splatting_dit_v1(splatting_v4_dpt):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        # ! modify decoder to output 3dgs parameters
        self.superresolution.update(
            dict(
                before_gs_conv=nn.Identity(),
                conv_sr=create_dpt_head_ln3diff(  # serve as Deconv
                    out_nchan=14,  # predict xyz
                    # feature_dim=256,  # ? projection dim
                    feature_dim=128,  # ? projection dim
                    l2=len(vit_decoder.blocks),  # type: ignore
                    dec_embed_dim=vit_decoder.embed_dim,
                    patch_size=vit_decoder.patch_size,
                    head_type='regression_gs',
                ),
            ))

    # ! SD Decoder not required anymore
    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        latent = self.superresolution['conv_sr'](
            latent_from_vit,  # list of B 1024 C
            img_info=(32 * 2, 32 * 2))  # B 14 H W

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        return ret_dict

    def forward_gaussians(self, ret_after_decoder, c):
        # ! modified from https://github.com/3DTopia/LGM/blob/main/core/models.py
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        # st()
        # w2c = c['orig_w2c']
        x = ret_after_decoder['latent_after_vit']
        B, C, H, W = x.shape  # B (V C) H W
        # st()
        # V = VC // 12
        # x = rearrange(x, 'b (v c) h w-> (b v) c h w', v=V)

        # B, V, C, H, W = images.shape
        # images = images.view(B*V, C, H, W)

        # x = self.unet(images) # [B*V, 14, h, w]
        x = self.superresolution['before_gs_conv'](x)  # [B*v, 14, h, w]

        # x = x.reshape(B, V, 12, H, W)  # TODO

        # ! no merge V required
        # x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 12)  # B
        # x = x.permute(0, 1, 3, 4, 2).reshape(B * V, -1,
        #                                      12)  # B V H W C -> BV HW C

        x = x.permute(0, 2, 3, 1).reshape(B, -1, C)  # B C H W -> B HW C

        # pos = self.pos_act(x[..., 0:3])  # [B, N, 3] # ! better act?
        # depth = self.depth_act(x[..., 0:1])  # [B, N, 3] # ! better act?

        # ! normalize to near, far.
        # origin_distances=0 here, all normalized into [-0.45, 0.45] # https://github.com/szymanowiczs/splatter-image/blob/98b465731c3273bf8f42a747d1b6ce1a93faf3d6/scene/gaussian_predictor.py#L741
        # depth = self.depth_act(x[..., 0:1])
        # depth = self.rendering_kwargs['z_near'] + depth * (
        #     self.rendering_kwargs['z_far'] - self.rendering_kwargs['z_near']
        # )  # [B, N, 3] # ! better act?
        # opacity = self.opacity_act(x[..., 1:2])
        # scale = self.scale_act(x[..., 2:5])
        # rotation = self.rot_act(x[..., 5:9])
        # rgbs = self.rgb_act(x[..., 9:])

        pos = self.pos_act(x[..., 0:3])  # [B, N, 3] # ! better act?
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        # ! convert depth to pos

        # cam2world_matrix = c['orig_c2w'][:, :, :16].reshape(B * V, 4, 4)
        # intrinsics = c['orig_pose'][:, :, 16:25].reshape(B * V, 3, 3)

        # # rotation = torch.bmm(cam2world_matrix, rotation) # transform rotation into the world space. How?
        # source_cv2wT_quat = c['source_cv2wT_quat']
        # source_cv2wT_quat = source_cv2wT_quat.reshape(
        #     B * V, *source_cv2wT_quat.shape[2:])
        # rotation = self.transform_rotations(
        #     rotation,  # ! transform rotations to the world space
        #     source_cv2wT_quat=source_cv2wT_quat)

        # # ! already in the world space after ray_sampler()
        # ray_origins, ray_directions = self.ray_sampler(  # shape:
        #     cam2world_matrix, intrinsics, H)[:2]
        # # self.gs.output_size,)[:2]
        # pos = ray_origins + depth * ray_directions  # BV HW 3, already in the world space

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs],
                              dim=-1)  # [B, N, 14]

        # gaussians = rearrange(gaussians, '(B V) N C -> B (V N) C', B=B,
        #                       V=V)  # merge V back to N

        if False:
            # if True:

            unmerged_pos = pos.reshape(B, V, -1, 3)
            unmerged_rgb = rgbs.reshape(B, V, -1, 3)

            for b in range(B):
                for v in range(V):
                    pcu.save_mesh_vc(f'tmp/splatting_v2-{b}-{v}.ply',
                                     unmerged_pos[b][v].detach().cpu().numpy(),
                                     unmerged_rgb[b][v].detach().cpu().numpy())
                    # pcu.save_mesh_v(f'tmp/splatting_v2-{b}-{v}.ply',

        # per_view_pos = rearrange(pos, '(B V) N C -> B V N C', B=B, V=V)
        ret_after_decoder.update({
            'gaussians': gaussians,
            'pos': gaussians[..., :3]
        })

        # st()

        return ret_after_decoder


class splatting_dit_v2_voxel(splatting_dit_v1):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)
        # predict voxel (fixed size) offsets

        self.deform_from_gt = False
        self.superresolution.update(
            dict(
                before_gs_conv=nn.Identity(),
                conv_sr=create_dpt_head_ln3diff(  # serve as Deconv
                    out_nchan=14,  # predict xyz
                    # feature_dim=256,  # ? projection dim
                    feature_dim=128,  # ? projection dim
                    l2=len(vit_decoder.blocks),  # type: ignore
                    dec_embed_dim=vit_decoder.embed_dim,
                    patch_size=vit_decoder.patch_size,
                    head_type='regression_voxel',
                ),
            ))
        # self.decoder.rendering_kwargs = self.rendering_kwargs
        self.rendering_kwargs = self.gs.rendering_kwargs
        self.scene_range = [
            self.rendering_kwargs['sampler_bbox_min'],
            self.rendering_kwargs['sampler_bbox_max']
        ]
        # self.g_cube = self.create_voxel_grid()  # just a voxel
        self.g_cube = self.create_voxel_grid() / 2  # just a voxel
        # self.g_cube = self.create_voxel_grid() / 3  # just a voxel
        # self.g_cube = self.create_voxel_grid() * 2 / 3  # just a voxel
        # self.g_cube = self.create_voxel_grid() * 1 / 2  # just a voxel
        # ! or in a sphere, also fine?
        # self.offset_act = lambda x: torch.tanh(x) * (self.rendering_kwargs['sampler_bbox_max'] / 32) # only small offsets
        # self.offset_act = lambda x: torch.tanh(x) * (self.rendering_kwargs['sampler_bbox_max']) / 2 # only small offsets
        # self.offset_act = lambda x: torch.tanh(x) * (self.rendering_kwargs['sampler_bbox_max']) # only small offsets
        # self.offset_act = lambda x: torch.tanh(x) * (self.rendering_kwargs['sampler_bbox_max']) * 2/3 # only small offsets
        # self.offset_act = lambda x: torch.tanh(x) * (self.rendering_kwargs['sampler_bbox_max']) / 3 # only small offsets

        # self.gt_shape = pcu.load_mesh_v('/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/Reconstruction/pcd-V=6/fps-pcd/Animals/0/10017/fps-10000.ply')
        # self.gt_shape = pcu.load_mesh_v(
        #     '/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/Reconstruction/pcd-V=6_from256/fps-pcd/Animals/0/10017/fps-10000.ply'
        # )
        # self.gt_shape = torch.from_numpy(self.gt_shape).to(
        #     dist_util.dev()).unsqueeze(0)
        # self.gt_shape = self.gt_shape / 2 # half the K

        # self.offset_act = lambda x: torch.tanh(x) * (self.rendering_kwargs['sampler_bbox_max']) * 2/3 # only small offsets
        # self.offset_act = lambda x: torch.tanh(x) * (self.rendering_kwargs['sampler_bbox_max']) * 1/5 # only small offsets
        self.offset_act = lambda x: torch.tanh(x) * (self.rendering_kwargs[
            'sampler_bbox_max']) * 0.05  # only small offsets
        self.scale_act = lambda x: (0.05 * F.softplus(x)).clamp(
            0, 0.45 * 0.1)  # avoid scaling larger than 0.1 * scene_extent
        # todo, update initialization following splatter image.

    def create_voxel_grid(self, grid_size=32, aabb=None, **kwargs):
        # planes: (N, 3, D', H', W')
        # grid_size: int

        # assert isinstance(self.rendering_kwargs, dict)

        # aabb: (N, 2, 3)
        if aabb is None:
            if 'sampler_bbox_min' in self.rendering_kwargs:
                aabb = torch.tensor([
                    [self.rendering_kwargs['sampler_bbox_min']] * 3,
                    [self.rendering_kwargs['sampler_bbox_max']] * 3,
                ],
                                    device=dist_util.dev(),
                                    dtype=torch.float32).unsqueeze(0)
                # .repeat( planes.shape[0], 1, 1)
            else:  # shapenet dataset, follow eg3d
                aabb = torch.tensor(
                    [  # https://github.com/NVlabs/eg3d/blob/7cf1fd1e99e1061e8b6ba850f91c94fe56e7afe4/eg3d/gen_samples.py#L188
                        [-self.rendering_kwargs['box_warp'] / 2] * 3,
                        [self.rendering_kwargs['box_warp'] / 2] * 3,
                    ],
                    device=dist_util.dev(),
                    dtype=torch.float32).unsqueeze(0)
                # .repeat( planes.shape[0], 1, 1)

        # create grid points for triplane query
        grid_points = []
        # for i in range(N):
        i = 0
        grid_points.append(
            torch.stack(torch.meshgrid(
                torch.linspace(aabb[i, 0, 0],
                               aabb[i, 1, 0],
                               grid_size,
                               device=dist_util.dev()),
                torch.linspace(aabb[i, 0, 1],
                               aabb[i, 1, 1],
                               grid_size,
                               device=dist_util.dev()),
                torch.linspace(aabb[i, 0, 2],
                               aabb[i, 1, 2],
                               grid_size,
                               device=dist_util.dev()),
                indexing='ij',
            ),
                        dim=-1).reshape(-1, 3))

        cube_grid = torch.stack(grid_points,
                                dim=0).to(dist_util.dev())  # 1 N 3
        return cube_grid

    def forward_gaussians(self, ret_after_decoder, c=None):
        # ! create grid and deform Gaussians accordingly
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        # st()
        x = ret_after_decoder['latent_after_vit']
        B, C, D, H, W = x.shape  # B C D H W, 14-dim voxel features
        assert C == 14
        assert (D == H and H == W)  # voxel here

        # x = rearrange(x, 'b (v c) h w-> (b v) c h w', v=V)

        # B, V, C, H, W = images.shape
        # images = images.view(B*V, C, H, W)

        # x = self.unet(images) # [B*V, 14, h, w]
        # x = self.superresolution['before_gs_conv'](x)  # [B*v, 14, h, w]

        # x = x.reshape(B, 14, D, H, W)  # TODO

        ## visualize multi-view gaussian features for plotting figure
        # tmp_alpha = self.opacity_act(x[0, :, 3:4])
        # tmp_img_rgb = self.rgb_act(x[0, :, 11:]) * tmp_alpha + (1 - tmp_alpha)
        # tmp_img_pos = self.pos_act(x[0, :, 0:3]) * 0.5 + 0.5
        # kiui.vis.plot_image(tmp_img_rgb, save=True)
        # kiui.vis.plot_image(tmp_img_pos, save=True)

        # save points for visualization

        # x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)  # B
        x = x.reshape(B, C, -1).permute(0, 2, 1)  # merge D H W -> B N C

        # create grid here
        # ! learn offests
        # offsets = self.offset_act(x[..., 0:3])  # [B, N, 3] # ! learned offsets
        # opacity = self.opacity_act(x[..., 3:4])
        # scale = self.scale_act(x[..., 4:7])
        # rotation = self.rot_act(x[..., 7:11])
        # rgbs = self.rgb_act(x[..., 11:])
        # pos = offsets + self.g_cube.repeat_interleave(B, dim=0)

        # ! ablation: directly use gt shape here
        # pos = self.gt_shape.repeat_interleave(B, dim=0) # type: ignore
        N = self.gt_shape.shape[1]

        # offsets = self.offset_act(x[..., 0:3])  # [B, N, 3] # ! learned offsets
        offsets = self.offset_act(x[:, :N,
                                    0:3])  # [B, N, 3] # ! learned offsets
        if self.deform_from_gt:
            pos = offsets + self.gt_shape.repeat_interleave(B, dim=0)
        else:
            st()
            pos = offsets + self.g_cube.repeat_interleave(B, dim=0)

        # pos = pos[:, :N, 0:3]

        opacity = self.opacity_act(x[:, :N, 3:4])

        scale = self.scale_act(x[:, :N, 4:7])

        rotation = self.rot_act(x[:, :N, 7:11])
        rgbs = self.rgb_act(x[:, :N, 11:])

        # st() # softplus(-2) = 0.0127

        # pos = offsets

        # if True:
        #     for b in range(1):
        #         pcu.save_mesh_v(f'tmp/voxel/fixinit-{b}.ply',
        #                             pos[b].detach().cpu().numpy(),)
        #                         #  unmerged_rgb[b].detach().cpu().numpy())

        # # pcu.save_mesh_v(f'tmp/voxel/gcube.ply', self.g_cube[0].detach().cpu().numpy(),)
        # # pcu.save_mesh_v(f'tmp/voxel/1_32-offsets.ply', offsets[0].detach().cpu().numpy(),)
        # # pcu.save_mesh_v(f'tmp/voxel/full-gtinit-2_3deform_full-offsets.ply', offsets[0].detach().cpu().numpy(),)
        # pcu.save_mesh_v(f'tmp/voxel/fixinit-offsets.ply', offsets[0].detach().cpu().numpy(),)

        # st()

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs],
                              dim=-1)  # [B, N, 14]

        # ! get per-view pos prediction
        # per_view_pos = rearrange(pos, 'B (V N) C -> (B V) N C', B=B, V=V)

        ret_after_decoder.update({
            'gaussians': gaussians,
            # 'per_view_pos': per_view_pos,
            'pos': pos
        })
        # ret_after_decoder.update({'gaussians': gaussians})

        return ret_after_decoder


class splatting_dit_v2_voxel_tuneInit(splatting_dit_v2_voxel):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)
        # self.scale_act = torch.exp  # from splatter image

        self.scale_act = lambda x: F.softplus(x) * 0.1

        # TODO, after GT shape fixed, larger deformation (1/3 e.g. available.)
        self.offset_act = lambda x: torch.tanh(x) * (self.rendering_kwargs[
            'sampler_bbox_max']) * 0.1  # regularize small offsets

        # self.superresolution.update({ # for normalization
        #     'before_gs_conv':
        #     nn.Conv2d(14, 14, kernel_size=1),
        # })

        #  init 14 dim predictions
        self.init_gaussian_prediction()
        self.deform_from_gt = True

    def init_gaussian_prediction(self):

        # https://github.com/szymanowiczs/splatter-image/blob/98b465731c3273bf8f42a747d1b6ce1a93faf3d6/configs/dataset/chairs.yaml#L15

        out_channels = [3, 1, 3, 4, 3]  # xyz, opacity, scale, rotation, rgb
        scale_inits = [  # ! avoid affecting final value (offset) 
            0,  #xyz_scale
            0.0,  #cfg.model.opacity_scale, 
            # 0.001,  #cfg.model.scale_scale,
            0,  #cfg.model.scale_scale,
            1,  # rotation
            0
        ]  # rgb

        bias_inits = [
            0.0,  # cfg.model.xyz_bias, no deformation here
            0,  # cfg.model.opacity_bias, sigmoid(0)=0.5 at init
            -2.5,  # scale_bias
            0.0,  # rotation
            0.5
        ]  # rgb

        start_channels = 0

        # for out_channel, b, s in zip(out_channels, bias, scale):
        for out_channel, b, s in zip(out_channels, bias_inits, scale_inits):
            # nn.init.xavier_uniform_(
            #     self.superresolution['conv_sr'].dpt.head[-1].weight[
            #         start_channels:start_channels + out_channel, ...], s)
            nn.init.constant_(
                self.superresolution['conv_sr'].dpt.head[-1].weight[
                    start_channels:start_channels + out_channel, ...], s)
            nn.init.constant_(
                self.superresolution['conv_sr'].dpt.head[-1].
                bias[start_channels:start_channels + out_channel], b)
            start_channels += out_channel

        # st()
        pass


class splatting_dit_v2_voxel_tuneInit_voxelemd(splatting_dit_v2_voxel_tuneInit
                                               ):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        # ! deform from a half-scene voxel

        self.offset_act = lambda x: torch.tanh(x) * (self.scene_range[
            1]) * 0.5  # regularize small offsets
        self.g_cube = self.create_voxel_grid() / 2  # just a voxel
        self.deform_from_gt = False  # deform from a voxel cube

    def forward_gaussians(self, ret_after_decoder, c=None):
        # ! create grid and deform Gaussians accordingly
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        # st()
        x = ret_after_decoder['latent_after_vit']
        B, C, D, H, W = x.shape  # B C D H W, 14-dim voxel features
        assert C == 14
        assert (D == H and H == W)  # voxel here

        # x = rearrange(x, 'b (v c) h w-> (b v) c h w', v=V)

        # B, V, C, H, W = images.shape
        # images = images.view(B*V, C, H, W)

        # x = self.unet(images) # [B*V, 14, h, w]
        # x = self.superresolution['before_gs_conv'](x)  # [B*v, 14, h, w]

        # x = x.reshape(B, 14, D, H, W)  # TODO

        ## visualize multi-view gaussian features for plotting figure
        # tmp_alpha = self.opacity_act(x[0, :, 3:4])
        # tmp_img_rgb = self.rgb_act(x[0, :, 11:]) * tmp_alpha + (1 - tmp_alpha)
        # tmp_img_pos = self.pos_act(x[0, :, 0:3]) * 0.5 + 0.5
        # kiui.vis.plot_image(tmp_img_rgb, save=True)
        # kiui.vis.plot_image(tmp_img_pos, save=True)

        # save points for visualization

        # x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)  # B
        x = x.reshape(B, C, -1).permute(0, 2, 1)  # merge D H W -> B N C

        # create grid here
        # ! learn offests
        # offsets = self.offset_act(x[..., 0:3])  # [B, N, 3] # ! learned offsets
        # opacity = self.opacity_act(x[..., 3:4])
        # scale = self.scale_act(x[..., 4:7])
        # rotation = self.rot_act(x[..., 7:11])
        # rgbs = self.rgb_act(x[..., 11:])
        # pos = offsets + self.g_cube.repeat_interleave(B, dim=0)

        # ! ablation: directly use gt shape here
        # pos = self.gt_shape.repeat_interleave(B, dim=0) # type: ignore
        # N = self.gt_shape.shape[1]

        offsets = self.offset_act(x[..., 0:3])  # [B, N, 3] # ! learned offsets
        # offsets = self.offset_act(x[:, :N,
        #                             0:3])  # [B, N, 3] # ! learned offsets
        if self.deform_from_gt:
            pos = offsets + self.gt_shape.repeat_interleave(B, dim=0)
        else:
            pos = offsets + self.g_cube.repeat_interleave(B, dim=0)

        opacity = self.opacity_act(x[..., 3:4])

        scale = self.scale_act(x[..., 4:7])

        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        # if True:
        #     for b in range(1):
        #         # pcu.save_mesh_v(f'tmp/voxel/emd/v5-fps-{b}.ply',
        #         pcu.save_mesh_v(f'tmp/voxel/emd/-fps-{b}.ply',
        #                             pos[b].detach().cpu().numpy(),)
        #                         #  unmerged_rgb[b].detach().cpu().numpy())

        # pcu.save_mesh_v(f'tmp/voxel/emd/v5-offsets.ply', offsets[0].detach().cpu().numpy(),)

        # st()

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs],
                              dim=-1)  # [B, N, 14]

        # ! get per-view pos prediction
        # per_view_pos = rearrange(pos, 'B (V N) C -> (B V) N C', B=B, V=V)

        ret_after_decoder.update({
            'gaussians': gaussians,
            # 'per_view_pos': per_view_pos,
            'pos': pos
        })
        # ret_after_decoder.update({'gaussians': gaussians})

        return ret_after_decoder


class splatting_dit_v3_voxel_tuneInit_voxelemd_dynamicLength(
        splatting_dit_v2_voxel_tuneInit_voxelemd):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            # vae_dit_token_size=16,
            plane_n=3,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        # st()
        # ! deprecated
        self.superresolution.update(
            dict(
                quant_conv=nn.Conv2d(  # no plane concept here, already fused.
                    2 * self.ldm_z_channels,
                    2 * self.ldm_embed_dim,
                    kernel_size=1,
                    # groups=self.plane_n,
                    groups=1,
                ),
                ldm_upsample=PatchEmbedTriplane(
                    self.vae_p * self.token_size,
                    self.vae_p,
                    self.ldm_embed_dim,  # B 3 L C
                    vit_decoder.embed_dim,
                    bias=True,
                    plane_n=plane_n,
                ),
                # ldm_upsample=PatchEmbed(
                #     256//8, # f=8, hard-coded.
                #     # self.vae_p,
                #     1, # no token downsample anymore
                #     self.ldm_embed_dim,  # B 3 L C
                #     vit_decoder.embed_dim,
                #     bias=True)
            ))  # group=1, already fused.

    # def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):
    #     return super().vit_decode_postprocess(latent_from_vit, ret_dict)

    def vit_decode_backbone(self, latent, img_size):
        return super().vit_decode_backbone(latent, img_size)


class splatting_dit_v3_voxel_tuneInit_voxelemd_dynamicLength_triLatent(
        splatting_dit_v3_voxel_tuneInit_voxelemd_dynamicLength):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)
        self.plane_n = 3


# https://github.com/facebookresearch/MCC/blob/main/mcc_model.py#L81
class XYZPosEmbed(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, embed_dim, multires=10):
        super().__init__()
        self.embed_dim = embed_dim
        # no [cls] token here.

        # ! use fixed PE here
        self.embed_fn, self.embed_input_ch = get_embedder(multires)
        # st()

        # self.two_d_pos_embed = nn.Parameter(
        #     # torch.zeros(1, 64 + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        #     torch.zeros(1, 64, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.win_size = 8

        self.xyz_projection = nn.Linear(self.embed_input_ch, embed_dim)

        # self.blocks = nn.ModuleList([
        #     Block(embed_dim, num_heads=12, mlp_ratio=2.0, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        #     for _ in range(1)
        # ])

        # self.invalid_xyz_token = nn.Parameter(torch.zeros(embed_dim,))

        # self.initialize_weights()

    # def initialize_weights(self):
    #     # torch.nn.init.normal_(self.cls_token, std=.02)

    #     two_d_pos_embed = get_2d_sincos_pos_embed(self.two_d_pos_embed.shape[-1], 8, cls_token=False)
    #     self.two_d_pos_embed.data.copy_(torch.from_numpy(two_d_pos_embed).float().unsqueeze(0))

    #     torch.nn.init.normal_(self.invalid_xyz_token, std=.02)

    def forward(self, xyz):
        xyz = self.embed_fn(xyz)  # PE encoding
        xyz = self.xyz_projection(xyz)  # linear projection
        return xyz


class gaussian_prediction(nn.Module):

    def __init__(
        self,
        query_dim,
    ) -> None:
        super().__init__()
        self.gaussian_pred = nn.Sequential(
            nn.SiLU(), nn.Linear(query_dim, 14,
                                 bias=True))  # TODO, init require

        self.init_gaussian_prediction()

    def init_gaussian_prediction(self):

        # https://github.com/szymanowiczs/splatter-image/blob/98b465731c3273bf8f42a747d1b6ce1a93faf3d6/configs/dataset/chairs.yaml#L15

        out_channels = [3, 1, 3, 4, 3]  # xyz, opacity, scale, rotation, rgb
        scale_inits = [  # ! avoid affecting final value (offset) 
            0,  #xyz_scale
            0.0,  #cfg.model.opacity_scale, 
            # 0.001,  #cfg.model.scale_scale,
            0,  #cfg.model.scale_scale,
            1.0,  # rotation
            0
        ]  # rgb

        bias_inits = [
            0.0,  # cfg.model.xyz_bias, no deformation here
            0,  # cfg.model.opacity_bias, sigmoid(0)=0.5 at init
            -2.5,  # scale_bias
            0.0,  # rotation
            0.5
        ]  # rgb

        start_channels = 0

        # for out_channel, b, s in zip(out_channels, bias, scale):
        for out_channel, b, s in zip(out_channels, bias_inits, scale_inits):
            # nn.init.xavier_uniform_(
            #     self.superresolution['conv_sr'].dpt.head[-1].weight[
            #         start_channels:start_channels + out_channel, ...], s)
            nn.init.constant_(
                self.gaussian_pred[1].weight[start_channels:start_channels +
                                             out_channel, ...], s)
            nn.init.constant_(
                self.gaussian_pred[1].bias[start_channels:start_channels +
                                           out_channel], b)
            start_channels += out_channel

    def forward(self, x):

        return self.gaussian_pred(x)


class surfel_prediction(nn.Module):
    # for 2dgs

    def __init__(
        self,
        query_dim,
    ) -> None:
        super().__init__()
        self.gaussian_pred = nn.Sequential(
            nn.SiLU(), nn.Linear(query_dim, 13,
                                 bias=True))  # TODO, init require

        self.init_gaussian_prediction()

    def init_gaussian_prediction(self):

        # https://github.com/szymanowiczs/splatter-image/blob/98b465731c3273bf8f42a747d1b6ce1a93faf3d6/configs/dataset/chairs.yaml#L15

        out_channels = [3, 1, 2, 4, 3]  # xyz, opacity, scale, rotation, rgb
        scale_inits = [  # ! avoid affecting final value (offset) 
            0,  #xyz_scale
            0.0,  #cfg.model.opacity_scale, 
            # 0.001,  #cfg.model.scale_scale,
            0,  #cfg.model.scale_scale,
            1.0,  # rotation
            0
        ]  # rgb

        bias_inits = [
            0.0,  # cfg.model.xyz_bias, no deformation here
            0,  # cfg.model.opacity_bias, sigmoid(0)=0.5 at init
            -2.5,  # scale_bias
            0,  # scale bias, also 0
            0.0,  # rotation
            0.5
        ]  # rgb

        start_channels = 0

        # for out_channel, b, s in zip(out_channels, bias, scale):
        for out_channel, b, s in zip(out_channels, bias_inits, scale_inits):
            # nn.init.xavier_uniform_(
            #     self.superresolution['conv_sr'].dpt.head[-1].weight[
            #         start_channels:start_channels + out_channel, ...], s)
            nn.init.constant_(
                self.gaussian_pred[1].weight[start_channels:start_channels +
                                             out_channel, ...], s)
            nn.init.constant_(
                self.gaussian_pred[1].bias[start_channels:start_channels +
                                           out_channel], b)
            start_channels += out_channel

    def forward(self, x):

        return self.gaussian_pred(x)


class pointInfinityWriteCA(gaussian_prediction):

    def __init__(self,
                 query_dim,
                 context_dim,
                 heads=8,
                 dim_head=64,
                 dropout=0.0) -> None:
        super().__init__(query_dim=query_dim)
        self.write_ca = MemoryEfficientCrossAttention(query_dim, context_dim,
                                                      heads, dim_head, dropout)

    def forward(self, x, z, return_x=False):
        # x: point to write
        # z: extracted latent
        x = self.write_ca(x, z)  # write from z to x
        if return_x:
            return self.gaussian_pred(x), x  # ! integrate it into dit?
        else:
            return self.gaussian_pred(x)  # ! integrate it into dit?


class pointInfinityWriteCA_cascade(pointInfinityWriteCA):
    # gradually (in 6 times) add deformation offsets to the initialized canonical pts, follow PI
    def __init__(self,
                 vit_depth,
                 query_dim,
                 context_dim,
                 heads=8,
                 dim_head=64,
                 dropout=0) -> None:
        super().__init__(query_dim, context_dim, heads, dim_head, dropout)

        del self.write_ca
        # query_dim = 384 # to speed up CA compute
        write_ca_interval = 12 // 4
        # self.deform_pred = nn.Sequential( # to-gaussian layer
        #     nn.SiLU(), nn.Linear(query_dim, 3, bias=True)) # TODO, init require

        # query_dim = 384 here
        self.write_ca_blocks = nn.ModuleList([
            MemoryEfficientCrossAttention(query_dim, context_dim,
                                          heads=heads)  # make it lite
            for _ in range(write_ca_interval)
            # for _ in range(write_ca_interval)
        ])
        self.hooks = [3, 7, 11]  # hard coded for now
        # [(vit_depth * 1 // 3) - 1, (vit_depth * 2 // 4) - 1, (vit_depth * 3 // 4) - 1,
        #             vit_depth - 1]

    def forward(self, x: torch.Tensor, z: list):
        # x is the canonical point
        # z: extracted latent (for different layers), all layers in dit
        # TODO, optimize memory, no need to return all layers?
        # st()

        z = [z[hook] for hook in self.hooks]
        # st()

        for idx, ca_blk in enumerate(self.write_ca_blocks):
            x = x + ca_blk(x, z[idx])  # learn residual feature

        return self.gaussian_pred(x)


class splatting_dit_v4_PI_V1(
        splatting_dit_v3_voxel_tuneInit_voxelemd_dynamicLength):
    # last layer add a write CA
    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        self.embed_dim = vit_decoder.embed_dim

        self.superresolution.update(
            dict(
                before_gs_conv=nn.Identity(),
                xyz_pos_embed=XYZPosEmbed(self.embed_dim),
                conv_sr=pointInfinityWriteCA(self.embed_dim,
                                             self.embed_dim),  # write
            ))

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        B = latent_from_vit.shape[0]
        # st()
        # fixed xyz for now
        # xyz = self.g_cube
        xyz_emb = self.superresolution['xyz_pos_embed'](
            self.g_cube).repeat_interleave(B, 0)  # B 14 H W
        latent = self.superresolution['conv_sr'](xyz_emb,
                                                 latent_from_vit)  # B 14 H W

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        return ret_dict

    def forward_gaussians(self, ret_after_decoder, c=None):
        # ! create grid and deform Gaussians accordingly
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        x = ret_after_decoder['latent_after_vit']
        B, N, C = x.shape  # B C D H W, 14-dim voxel features
        # st()
        assert C == 14

        # pos = self.pos_act(x[..., 0:3])  # [B, N, 3] # ! learned offsets
        offsets = self.offset_act(x[..., 0:3])  # [B, N, 3] # ! learned offsets
        # # offsets = self.offset_act(x[:, :N,
        # #                             0:3])  # [B, N, 3] # ! learned offsets
        # if self.deform_from_gt:
        #     pos = offsets + self.gt_shape.repeat_interleave(B, dim=0)
        # else:
        pos = offsets + self.g_cube.repeat_interleave(B, dim=0)

        opacity = self.opacity_act(x[..., 3:4])

        scale = self.scale_act(x[..., 4:7])

        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        # if True:
        #     for b in range(1):
        #         # pcu.save_mesh_v(f'tmp/voxel/emd/pi-fullfps-8k-fps-{b}.ply',
        #         # pcu.save_mesh_v(f'tmp/voxel/emd/pi-fullfps-4k-fps-{b}.ply',
        #         pcu.save_mesh_v(f'tmp/voxel/emd/pi-fullfps-1k-fps-{b}.ply',
        #                             pos[b].detach().cpu().numpy(),)
        #                         #  unmerged_rgb[b].detach().cpu().numpy())

        # # pcu.save_mesh_v(f'tmp/voxel/gcube.ply', self.g_cube[0].detach().cpu().numpy(),)
        # # pcu.save_mesh_v(f'tmp/voxel/1_32-offsets.ply', offsets[0].detach().cpu().numpy(),)
        # # pcu.save_mesh_v(f'tmp/voxel/full-gtinit-2_3deform_full-offsets.ply', offsets[0].detach().cpu().numpy(),)
        # # pcu.save_mesh_v(f'tmp/voxel/emd/pi-v2-17k.ply', offsets[0].detach().cpu().numpy(),)
        # pcu.save_mesh_v(f'tmp/voxel/emd/pi-fullfps-1k.ply', offsets[0].detach().cpu().numpy(),)

        # st()

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs],
                              dim=-1)  # [B, N, 14]

        # ! get per-view pos prediction
        # per_view_pos = rearrange(pos, 'B (V N) C -> (B V) N C', B=B, V=V)

        ret_after_decoder.update({
            'gaussians': gaussians,
            # 'per_view_pos': per_view_pos,
            'pos': pos
        })
        # ret_after_decoder.update({'gaussians': gaussians})

        return ret_after_decoder


class splatting_dit_v4_PI_V1_lite(splatting_dit_v4_PI_V1):
    # half CA dim. worse performance.
    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)
        ctx_dim = vit_decoder.embed_dim
        query_dim = vit_decoder.embed_dim // 2

        self.superresolution.update(
            dict(
                before_gs_conv=nn.Identity(),
                xyz_pos_embed=XYZPosEmbed(query_dim),
                conv_sr=pointInfinityWriteCA(query_dim, ctx_dim,
                                             heads=6),  # write
            ))


class splatting_dit_v4_PI_V2(splatting_dit_v4_PI_V1):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)
        self.xyz_embed_dim = vit_decoder.embed_dim // 2  # 384

        self.superresolution.update(
            dict(
                # before_gs_conv=nn.Identity(),
                xyz_pos_embed=XYZPosEmbed(self.xyz_embed_dim),
                conv_sr=pointInfinityWriteCA_cascade(vit_decoder.depth,
                                                     self.xyz_embed_dim,
                                                     vit_decoder.embed_dim,
                                                     heads=6),  # write
            ))

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        B = latent_from_vit[0].shape[0]
        # st()
        # fixed xyz for now
        # xyz = self.g_cube
        xyz_emb = self.superresolution['xyz_pos_embed'](
            self.g_cube).repeat_interleave(B, 0)  # B 14 H W
        latent = self.superresolution['conv_sr'](xyz_emb,
                                                 latent_from_vit)  # B 14 H W

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        return ret_dict


# import torch
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


def create_sphere(radius, num_points):
    # Generate spherical coordinates
    phi = torch.linspace(0, 2 * torch.pi, num_points)
    theta = torch.linspace(0, torch.pi, num_points)
    phi, theta = torch.meshgrid(phi, theta, indexing='xy')

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * torch.sin(theta) * torch.cos(phi)
    y = radius * torch.sin(theta) * torch.sin(phi)
    z = radius * torch.cos(theta)

    # Stack x, y, z coordinates
    points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)

    return points


# # Parameters
# radius = 1.0
# num_points = 100

# # Create the sphere
# x, y, z = create_sphere(radius, num_points)

# # Plot the sphere
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, y, z, color='b', alpha=0.5)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Sphere')
# plt.show()


class splatting_dit_v4_PI_V1_trilatent(splatting_dit_v4_PI_V1):
    # tri-plane latent 768 token
    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            fusion_blk,
            channel_multiplier,
            #  plane_n=3,
            **kwargs)
        # self.plane_n = 3

        # self.superresolution.update(
        #     dict(
        #         ldm_upsample=PatchEmbedTriplane(
        #             self.vae_p * self.token_size,
        #             self.vae_p,
        #             self.ldm_embed_dim,  # B 3 L C
        #             vit_decoder.embed_dim,
        #             bias=True,
        #             plane_n=self.plane_n,
        #         ),
        #     ))

    def vit_decode_backbone(self, latent, img_size):
        return super().vit_decode_backbone(latent, img_size)

    def vit_decode(self, latent, img_size, c=None, sample_posterior=True):
        return super().vit_decode(latent, img_size, c, sample_posterior)

    def forward_vit_decoder(self, x, img_size=None):
        return super().forward_vit_decoder(x, img_size)

    def forward_gaussians(self, ret_after_decoder, c=None):
        # ! create grid and deform Gaussians accordingly
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        x = ret_after_decoder['latent_after_vit']
        B, N, C = x.shape  # B C D H W, 14-dim voxel features
        # st()
        assert C == 14

        # pos = self.pos_act(x[..., 0:3])  # [B, N, 3] # ! learned offsets
        offsets = self.offset_act(x[..., 0:3])  # [B, N, 3] # ! learned offsets
        # # offsets = self.offset_act(x[:, :N,
        # #                             0:3])  # [B, N, 3] # ! learned offsets
        # if self.deform_from_gt:
        #     pos = offsets + self.gt_shape.repeat_interleave(B, dim=0)
        # else:
        pos = offsets + self.g_cube.repeat_interleave(B, dim=0)

        opacity = self.opacity_act(x[..., 3:4])

        scale = self.scale_act(x[..., 4:7])

        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        # if True:
        #     for b in range(B):
        #         # pcu.save_mesh_v(f'tmp/voxel/emd/pi-fullfps-1k-fps-{b}.ply',
        #         pcu.save_mesh_v(f'tmp/voxel/emd/pi-fullfps-1k-fps-{b}.ply',
        #                             pos[b].detach().cpu().numpy(),)
        #                         #  unmerged_rgb[b].detach().cpu().numpy())

        # # pcu.save_mesh_v(f'tmp/voxel/emd/pi-fullfps-1k.ply', offsets[0].detach().cpu().numpy(),)

        # st()

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs],
                              dim=-1)  # [B, N, 14]

        # ! get per-view pos prediction
        # per_view_pos = rearrange(pos, 'B (V N) C -> (B V) N C', B=B, V=V)

        ret_after_decoder.update({
            'gaussians': gaussians,
            # 'per_view_pos': per_view_pos,
            'pos': pos
        })
        # ret_after_decoder.update({'gaussians': gaussians})

        return ret_after_decoder


class splatting_dit_v4_PI_V1_trilatent_sphere(splatting_dit_v4_PI_V1_trilatent
                                              ):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=181,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        # self.g_cube = create_sphere(self.scene_range[1]*2/3, 192).to(dist_util.dev()).unsqueeze(0)  # just a voxel
        # self.g_cube = create_sphere(self.scene_range[1]*2/3, 181).to(dist_util.dev()).unsqueeze(0)  # just a voxel
        self.g_cube = create_sphere(self.scene_range[1] * 2 / 3, pcd_reso).to(
            dist_util.dev()).unsqueeze(0)  # just a voxel
        # self.g_cube = create_sphere(radius=self.scene_range[1]*2/3, num_spherical_samples=32**3)[0]
        # pcu.save_mesh_v(f'tmp/sphere.ply',self.g_cube[0].cpu().numpy())

        # ! TODO, requires lint
        self.vit_decoder.pos_embed = nn.Parameter(
            torch.zeros(1,
                        self.plane_n * (self.token_size**2 + self.cls_token),
                        vit_decoder.embed_dim))
        self.init_weights()  # re-init weights after re-writing token_size
        # pass

    def vis_gaussian(self, gaussians, file_name_base):
        # gaussians = ret_after_decoder['gaussians']
        # gaussians = ret_after_decoder['latent_after_vit']['gaussians_base']
        B = gaussians.shape[0]
        pos, opacity, scale, rotation, rgbs = gaussians[..., 0:3], gaussians[
            ..., 3:4], gaussians[..., 4:7], gaussians[...,
                                                      7:11], gaussians[...,
                                                                       11:14]
        file_path = Path(logger.get_dir())

        for b in range(B):
            # file_name = f'pcd48-f4-opacity1{b}'
            # file_name = f'pcd48-f4-opacity1{b}'
            # file_name = f'pcd48-f4-aug0.2-{b}'
            file_name = f'{file_name_base}-{b}'

            np.save(file_path / f'{file_name}_opacity.npy',
                    opacity[b].float().detach().cpu().numpy())
            np.save(file_path / f'{file_name}_scale.npy',
                    scale[b].float().detach().cpu().numpy())
            np.save(file_path / f'{file_name}_rotation.npy',
                    rotation[b].float().detach().cpu().numpy())

            pcu.save_mesh_vc(str(file_path / f'{file_name}.ply'),
                             pos[b].float().detach().cpu().numpy(),
                             rgbs[b].float().detach().cpu().numpy())

    def forward_gaussians(self, ret_after_decoder, c=None):
        # ! create grid and deform Gaussians accordingly
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        x = ret_after_decoder['latent_after_vit']
        B, N, C = x.shape  # B C D H W, 14-dim voxel features
        # st()
        assert C == 14

        # pos = self.pos_act(x[..., 0:3])  # [B, N, 3] # ! learned offsets
        offsets = self.offset_act(x[..., 0:3])  # [B, N, 3] # ! learned offsets
        # # offsets = self.offset_act(x[:, :N,
        # #                             0:3])  # [B, N, 3] # ! learned offsets
        # if self.deform_from_gt:
        #     pos = offsets + self.gt_shape.repeat_interleave(B, dim=0)
        # else:
        pos = offsets + self.g_cube.repeat_interleave(B, dim=0)

        opacity = self.opacity_act(x[..., 3:4])

        scale = self.scale_act(x[..., 4:7])

        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        # st()

        # if True:
        #     for b in range(B):
        #         file_path = Path('tmp/sphere/residual')
        #         # file_name = f'pcd48-{b}'
        #         # file_name = f'pcd48-again-{b}'
        #         # file_name = f'pcd48-f4-{b}'
        #         file_name = f'pcd48-f4-{b}'

        #         np.save(file_path / f'{file_name}_opacity.npy', opacity[b].float().detach().cpu().numpy())
        #         np.save(file_path / f'{file_name}_scale.npy', scale[b].float().detach().cpu().numpy())
        #         np.save(file_path / f'{file_name}_rotation.npy', rotation[b].float().detach().cpu().numpy())

        #         pcu.save_mesh_vc(str(file_path / f'{file_name}.ply'),
        #                             pos[b].float().detach().cpu().numpy(),
        #                             rgbs[b].float().detach().cpu().numpy())

        # st()

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs],
                              dim=-1)  # [B, N, 14]

        # st()
        # self.vis_gaussian(gaussians, 'base')
        # # self.vis_gaussian(
        # #     ret_after_decoder['latent_after_vit']['gaussians_upsampled'],
        # #     'augbg-f4')
        # st()

        # ! get per-view pos prediction
        # per_view_pos = rearrange(pos, 'B (V N) C -> (B V) N C', B=B, V=V)

        ret_after_decoder.update({
            'gaussians': gaussians,
            # 'per_view_pos': per_view_pos,
            'pos': pos
        })
        # ret_after_decoder.update({'gaussians': gaussians})

        return ret_after_decoder


class splatting_dit_v4_PI_V1_trilatent_sphere_L(
        splatting_dit_v4_PI_V1_trilatent_sphere):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=181,
            **kwargs) -> None:
        super().__init__(vit_decoder,
                         triplane_decoder,
                         cls_token,
                         normalize_feat,
                         sr_ratio,
                         use_fusion_blk,
                         fusion_blk_depth,
                         fusion_blk,
                         channel_multiplier,
                         pcd_reso=256,
                         **kwargs)


class splatting_dit_v4_PI_V1_trilatent_sphere_S(
        splatting_dit_v4_PI_V1_trilatent_sphere):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=181,
            **kwargs) -> None:
        super().__init__(vit_decoder,
                         triplane_decoder,
                         cls_token,
                         normalize_feat,
                         sr_ratio,
                         use_fusion_blk,
                         fusion_blk_depth,
                         fusion_blk,
                         channel_multiplier,
                         pcd_reso=64,
                         **kwargs)

        # self.token_size = vae_dit_token_size  # use dino-v2 dim tradition here


class splatting_dit_v4_PI_V1_trilatent_sphere_S_48(
        splatting_dit_v4_PI_V1_trilatent_sphere):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=181,
            # vae_dit_token_size=16,
            **kwargs) -> None:
        # st()

        # assert pcd_reo == 48
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            fusion_blk,
            channel_multiplier,
            pcd_reso=48,
            #  pcd_reso=pcd_reso,
            #  vae_dit_token_size=vae_dit_token_size,
            **kwargs)
        # st()
        pass


class splatting_dit_v4_PI_V1_trilatent_sphere_S_60(
        splatting_dit_v4_PI_V1_trilatent_sphere):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=181,
            # vae_dit_token_size=16,
            **kwargs) -> None:
        # st()

        # assert pcd_reo == 48
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            fusion_blk,
            channel_multiplier,
            pcd_reso=60,
            #  pcd_reso=pcd_reso,
            #  vae_dit_token_size=vae_dit_token_size,
            **kwargs)
        # st()
        pass


# ==== direct XYZ prediction


# class splatting_dit_v5_directXYZpred(splatting_dit_v4_PI_V1_trilatent_sphere_S_60):
class splatting_dit_v5_directXYZpred(
        splatting_dit_v4_PI_V1_trilatent_sphere_S_48):
    # avoid deforming a sphere here
    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=181,
            # vae_dit_token_size=16,
            **kwargs) -> None:
        # st()

        # assert pcd_reo == 48
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            fusion_blk,
            channel_multiplier,
            pcd_reso=60,
            #  pcd_reso=pcd_reso,
            #  vae_dit_token_size=vae_dit_token_size,
            **kwargs)
        # st()

        self.xyz_pred = lambda x: torch.tanh(x) * (self.scene_range[
            1])  # directly pred scene scale coordinate

    def forward_gaussians(self, ret_after_decoder, c=None):
        # ! create grid and deform Gaussians accordingly
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        x = ret_after_decoder['latent_after_vit']
        B, N, C = x.shape  # B C D H W, 14-dim voxel features
        # st()
        assert C == 14

        # pos = self.pos_act(x[..., 0:3])  # [B, N, 3] # ! learned offsets
        pos = self.xyz_pred(x[..., 0:3])  # [B, N, 3] # ! learned offsets
        # # offsets = self.offset_act(x[:, :N,
        # #                             0:3])  # [B, N, 3] # ! learned offsets
        # if self.deform_from_gt:
        #     pos = offsets + self.gt_shape.repeat_interleave(B, dim=0)
        # else:
        # pos = offsets + self.g_cube.repeat_interleave(B, dim=0)

        opacity = self.opacity_act(x[..., 3:4])

        scale = self.scale_act(x[..., 4:7])

        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        # st()

        # if True:
        #     for b in range(B):
        #         file_path = Path('tmp/sphere/residual')
        #         # file_name = f'pcd48-{b}'
        #         # file_name = f'pcd48-again-{b}'
        #         # file_name = f'pcd48-f4-{b}'
        #         file_name = f'pcd48-f4-{b}'

        #         np.save(file_path / f'{file_name}_opacity.npy', opacity[b].float().detach().cpu().numpy())
        #         np.save(file_path / f'{file_name}_scale.npy', scale[b].float().detach().cpu().numpy())
        #         np.save(file_path / f'{file_name}_rotation.npy', rotation[b].float().detach().cpu().numpy())

        #         pcu.save_mesh_vc(str(file_path / f'{file_name}.ply'),
        #                             pos[b].float().detach().cpu().numpy(),
        #                             rgbs[b].float().detach().cpu().numpy())

        # st()

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs],
                              dim=-1)  # [B, N, 14]

        # st()
        # self.vis_gaussian(gaussians, 'base')
        # # self.vis_gaussian(
        # #     ret_after_decoder['latent_after_vit']['gaussians_upsampled'],
        # #     'augbg-f4')
        # st()

        # ! get per-view pos prediction
        # per_view_pos = rearrange(pos, 'B (V N) C -> (B V) N C', B=B, V=V)

        ret_after_decoder.update({
            'gaussians': gaussians,
            # 'per_view_pos': per_view_pos,
            'pos': pos
        })
        # ret_after_decoder.update({'gaussians': gaussians})

        return ret_after_decoder


# ===== direct XYZ prediction done


class splatting_dit_v4_PI_V1_trilatent_sphere_S_100(
        splatting_dit_v4_PI_V1_trilatent_sphere):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=181,
            **kwargs) -> None:
        super().__init__(vit_decoder,
                         triplane_decoder,
                         cls_token,
                         normalize_feat,
                         sr_ratio,
                         use_fusion_blk,
                         fusion_blk_depth,
                         fusion_blk,
                         channel_multiplier,
                         pcd_reso=100,
                         **kwargs)


class splatting_dit_v4_PI_V1_trilatent_sphere_S_256(
        splatting_dit_v4_PI_V1_trilatent_sphere):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=181,
            **kwargs) -> None:
        super().__init__(vit_decoder,
                         triplane_decoder,
                         cls_token,
                         normalize_feat,
                         sr_ratio,
                         use_fusion_blk,
                         fusion_blk_depth,
                         fusion_blk,
                         channel_multiplier,
                         pcd_reso=256,
                         **kwargs)


class splatting_dit_v4_PI_V1_trilatent_sphere_L_opacityRes(
        splatting_dit_v4_PI_V1_trilatent_sphere_L):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=181,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         pcd_reso, **kwargs)
        self.opacity_act = lambda x: torch.sigmoid(x) / 2

    def forward_gaussians(self, ret_after_decoder, c=None):
        # ! create grid and deform Gaussians accordingly
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        x = ret_after_decoder['latent_after_vit']
        B, N, C = x.shape  # B C D H W, 14-dim voxel features
        # st()
        assert C == 14

        # pos = self.pos_act(x[..., 0:3])  # [B, N, 3] # ! learned offsets
        offsets = self.offset_act(x[..., 0:3])  # [B, N, 3] # ! learned offsets
        # # offsets = self.offset_act(x[:, :N,
        # #                             0:3])  # [B, N, 3] # ! learned offsets
        # if self.deform_from_gt:
        #     pos = offsets + self.gt_shape.repeat_interleave(B, dim=0)
        # else:
        pos = offsets + self.g_cube.repeat_interleave(B, dim=0)

        opacity = self.opacity_act(
            x[..., 3:4]) + 0.5  # ! residual, avoid empty points

        scale = self.scale_act(x[..., 4:7])

        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        # if True:
        #     for b in range(1):
        #         # pcu.save_mesh_v(f'tmp/voxel/emd/pi-fullfps-8k-fps-{b}.ply',
        #         # pcu.save_mesh_v(f'tmp/voxel/emd/pi-fullfps-4k-fps-{b}.ply',
        #         pcu.save_mesh_v(f'tmp/voxel/emd/pi-fullfps-1k-fps-{b}.ply',
        #                             pos[b].detach().cpu().numpy(),)
        #                         #  unmerged_rgb[b].detach().cpu().numpy())

        # # pcu.save_mesh_v(f'tmp/voxel/gcube.ply', self.g_cube[0].detach().cpu().numpy(),)
        # # pcu.save_mesh_v(f'tmp/voxel/1_32-offsets.ply', offsets[0].detach().cpu().numpy(),)
        # # pcu.save_mesh_v(f'tmp/voxel/full-gtinit-2_3deform_full-offsets.ply', offsets[0].detach().cpu().numpy(),)
        # # pcu.save_mesh_v(f'tmp/voxel/emd/pi-v2-17k.ply', offsets[0].detach().cpu().numpy(),)
        # pcu.save_mesh_v(f'tmp/voxel/emd/pi-fullfps-1k.ply', offsets[0].detach().cpu().numpy(),)

        # st()

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs],
                              dim=-1)  # [B, N, 14]

        # ! get per-view pos prediction
        # per_view_pos = rearrange(pos, 'B (V N) C -> (B V) N C', B=B, V=V)

        ret_after_decoder.update({
            'gaussians': gaussians,
            # 'per_view_pos': per_view_pos,
            'pos': pos
        })
        # ret_after_decoder.update({'gaussians': gaussians})

        return ret_after_decoder


class GS_Adaptive_Write_CA(nn.Module):

    def __init__(
            self,
            query_dim,
            context_dim,
            f=4,  # upsampling ratio
            heads=8,
            dim_head=64,
            dropout=0.0) -> None:
        super().__init__()

        self.f = f
        self.write_ca = MemoryEfficientCrossAttention(query_dim, context_dim,
                                                      heads, dim_head, dropout)
        self.gaussian_residual_pred = nn.Sequential(
            nn.SiLU(),
            nn.Linear(query_dim, 14,
                      bias=True))  # predict residual, before activations

        # ! hard coded
        self.scene_extent = 0.9  # g-buffer, [-0.45, 0.45]
        self.percent_dense = 0.01  # 3dgs official value
        self.residual_offset_act = lambda x: torch.tanh(
            x) * self.scene_extent * 0.015  # avoid large deformation

        init_gaussian_prediction(self.gaussian_residual_pred[1])

    # def densify_and_split(self, gaussians_base, base_gaussian_xyz_embed):

    def forward(self,
                gaussians_base,
                gaussian_base_pre_activate,
                gaussian_base_feat,
                xyz_embed_fn,
                shrink_scale=True):
        # gaussians_base: xyz_base after activations and deform offset
        # xyz_base: original features (before activations)

        # ! use point embedder, or other features?
        # base_gaussian_xyz_embed = xyz_embed_fn(gaussians_base[..., :3])

        # x = self.densify_and_split(gaussians_base, base_gaussian_xyz_embed)

        # ! densify
        B, N = gaussians_base.shape[:2]  # gaussians upsample factor
        # n_init_points = self.get_xyz.shape[0]

        pos, opacity, scaling, rotation = gaussians_base[
            ..., 0:3], gaussians_base[..., 3:4], gaussians_base[
                ..., 4:7], gaussians_base[..., 7:11]

        # ! filter clone/densify based on scaling range
        split_mask = scaling.max(
            dim=-1
        )[0] > self.scene_extent * self.percent_dense  # shape: B 4096
        # clone_mask = ~split_mask

        stds = scaling.repeat_interleave(self.f, dim=1)  #  0 0 1 1 2 2...
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)  # B f*N 3

        # rots = build_rotation(rotation).repeat(N, 1, 1)
        # rots = rearrange(build_rotation(rearrange(rotation, 'B N ... -> (B N) ...')), '(B N) ... -> B N ...', B=B, N=N)
        # rots = rots.repeat_interleave(self.f, dim=1) # B f*N 3 3

        # torch.bmm only supports ndim=3 Tensor
        # new_xyz = torch.matmul(rots, samples.unsqueeze(-1)).squeeze(-1) + pos.repeat_interleave(self.f, dim=1)
        new_xyz = samples + pos.repeat_interleave(
            self.f, dim=1)  # ! no rotation for now
        # new_xyz: B f*N 3

        # ! new points to features
        new_xyz_embed = xyz_embed_fn(new_xyz)
        new_gaussian_embed = self.write_ca(
            new_xyz_embed, gaussian_base_feat)  # write from z to x

        # ! predict gaussians residuals
        gaussian_residual_pre_activate = self.gaussian_residual_pred(
            new_gaussian_embed)

        # ! add back. how to deal with new rotations? check the range first.
        # scaling and rotation.
        if shrink_scale:
            gaussian_base_pre_activate[split_mask][
                4:7] -= 1  # reduce scale for those points

        gaussian_base_pre_activate_repeat = gaussian_base_pre_activate.repeat_interleave(
            self.f, dim=1)

        # new scaling
        # ! pre-activate scaling value, shall be negative? since more values are 0.1 before softplus.
        # TODO wrong here, shall get new scaling before repeat
        gaussians = gaussian_residual_pre_activate + gaussian_base_pre_activate_repeat  # learn the residual

        new_gaussians_pos = new_xyz + self.residual_offset_act(
            gaussians[..., :3])

        return gaussians, new_gaussians_pos  # return positions independently


class GS_Adaptive_Read_Write_CA(nn.Module):

    def __init__(
            self,
            query_dim,
            context_dim,
            mlp_ratio,
            vit_heads,
            f=4,  # upsampling ratio
            heads=8,
            dim_head=64,
            dropout=0.0,
            depth=2,
            vit_blk=DiTBlock2) -> None:
        super().__init__()

        self.f = f
        self.read_ca = MemoryEfficientCrossAttention(query_dim, context_dim,
                                                     heads, dim_head, dropout)

        # more dit blocks
        self.point_infinity_blocks = nn.ModuleList([
            vit_blk(context_dim, num_heads=vit_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)  # since dit-b here
        ])

        self.write_ca = MemoryEfficientCrossAttention(query_dim, context_dim,
                                                      heads, dim_head, dropout)

        self.gaussian_residual_pred = nn.Sequential(
            nn.SiLU(),
            nn.Linear(query_dim, 14,
                      bias=True))  # predict residual, before activations

        # ! hard coded
        self.scene_extent = 0.9  # g-buffer, [-0.45, 0.45]
        self.percent_dense = 0.01  # 3dgs official value
        self.residual_offset_act = lambda x: torch.tanh(
            x) * self.scene_extent * 0.015  # avoid large deformation

        self.initialize_weights()

    def initialize_weights(self):
        init_gaussian_prediction(self.gaussian_residual_pred[1])

        for block in self.point_infinity_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    # def densify_and_split(self, gaussians_base, base_gaussian_xyz_embed):

    def forward(self, gaussians_base, gaussian_base_pre_activate,
                gaussian_base_feat, latent_from_vit, vae_latent, xyz_embed_fn):
        # gaussians_base: xyz_base after activations and deform offset
        # xyz_base: original features (before activations)

        # ========= START read CA ========
        latent_from_vit = self.read_ca(latent_from_vit,
                                       gaussian_base_feat)  # z_i -> z_(i+1)

        for blk_idx, block in enumerate(self.point_infinity_blocks):
            latent_from_vit = block(latent_from_vit,
                                    vae_latent)  # vae_latent: c

        # ========= END read CA ========

        # ! use point embedder, or other features?
        # base_gaussian_xyz_embed = xyz_embed_fn(gaussians_base[..., :3])

        # x = self.densify_and_split(gaussians_base, base_gaussian_xyz_embed)

        # ! densify
        B, N = gaussians_base.shape[:2]  # gaussians upsample factor
        # n_init_points = self.get_xyz.shape[0]

        pos, opacity, scaling, rotation = gaussians_base[
            ..., 0:3], gaussians_base[..., 3:4], gaussians_base[
                ..., 4:7], gaussians_base[..., 7:11]

        # ! filter clone/densify based on scaling range
        split_mask = scaling.max(
            dim=-1
        )[0] > self.scene_extent * self.percent_dense  # shape: B 4096
        # clone_mask = ~split_mask

        stds = scaling.repeat_interleave(self.f, dim=1)  #  0 0 1 1 2 2...
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)  # B f*N 3

        rots = build_rotation(rotation).repeat(N, 1, 1)
        rots = rearrange(build_rotation(
            rearrange(rotation, 'B N ... -> (B N) ...')),
                         '(B N) ... -> B N ...',
                         B=B,
                         N=N)
        rots = rots.repeat_interleave(self.f, dim=1)  # B f*N 3 3

        # torch.bmm only supports ndim=3 Tensor
        new_xyz = torch.matmul(
            rots, samples.unsqueeze(-1)).squeeze(-1) + pos.repeat_interleave(
                self.f, dim=1)
        # new_xyz = samples + pos.repeat_interleave(
        #     self.f, dim=1)  # ! no rotation for now
        # new_xyz: B f*N 3

        # ! new points to features
        new_xyz_embed = xyz_embed_fn(new_xyz)
        new_gaussian_embed = self.write_ca(
            new_xyz_embed, latent_from_vit
        )  # ! use z_(i+1), rather than gaussian_base_feat here

        # ! predict gaussians residuals
        gaussian_residual_pre_activate = self.gaussian_residual_pred(
            new_gaussian_embed)

        # ! add back. how to deal with new rotations? check the range first.
        # scaling and rotation.
        gaussian_base_pre_activate[split_mask][
            4:7] -= 1  # reduce scale for those points
        gaussian_base_pre_activate_repeat = gaussian_base_pre_activate.repeat_interleave(
            self.f, dim=1)

        # new scaling
        # ! pre-activate scaling value, shall be negative? since more values are 0.1 before softplus.
        # TODO wrong here, shall get new scaling before repeat
        gaussians = gaussian_residual_pre_activate + gaussian_base_pre_activate_repeat  # learn the residual

        new_gaussians_pos = new_xyz + self.residual_offset_act(
            gaussians[..., :3])

        return gaussians, new_gaussians_pos, latent_from_vit, new_gaussian_embed  # return positions independently


class GS_Adaptive_Read_Write_CA_adaptive(GS_Adaptive_Read_Write_CA):

    def __init__(self,
                 query_dim,
                 context_dim,
                 mlp_ratio,
                 vit_heads,
                 f=4,
                 heads=8,
                 dim_head=64,
                 dropout=0,
                 depth=2,
                 vit_blk=DiTBlock2) -> None:
        super().__init__(query_dim, context_dim, mlp_ratio, vit_heads, f,
                         heads, dim_head, dropout, depth, vit_blk)

        # assert self.f == 6

    def forward(self, gaussians_base, gaussian_base_pre_activate,
                gaussian_base_feat, latent_from_vit, vae_latent, xyz_embed_fn):
        # gaussians_base: xyz_base after activations and deform offset
        # xyz_base: original features (before activations)

        # ========= START read CA ========
        latent_from_vit = self.read_ca(latent_from_vit,
                                       gaussian_base_feat)  # z_i -> z_(i+1)

        for blk_idx, block in enumerate(self.point_infinity_blocks):
            latent_from_vit = block(latent_from_vit,
                                    vae_latent)  # vae_latent: c

        # ========= END read CA ========

        # ! use point embedder, or other features?
        # base_gaussian_xyz_embed = xyz_embed_fn(gaussians_base[..., :3])

        # x = self.densify_and_split(gaussians_base, base_gaussian_xyz_embed)

        # ! densify
        B, N = gaussians_base.shape[:2]  # gaussians upsample factor
        # n_init_points = self.get_xyz.shape[0]

        pos, opacity, scaling, rotation = gaussians_base[
            ..., 0:3], gaussians_base[..., 3:4], gaussians_base[
                ..., 4:7], gaussians_base[..., 7:11]

        # ! filter clone/densify based on scaling range

        split_mask = scaling.max(
            dim=-1
        )[0] > self.scene_extent * self.percent_dense  # shape: B 4096

        # clone_mask = ~split_mask

        # stds = scaling.repeat_interleave(self.f, dim=1)  #  B 13824 3
        # stds = scaling.unsqueeze(1).repeat_interleave(self.f, dim=1)  #  B 6 13824 3
        stds = scaling  #  B 13824 3

        # TODO, in mat form. axis aligned creation.
        samples = torch.zeros(B, N, 3, 3).to(stds.device)

        samples[..., 0, 0] = stds[..., 0]
        samples[..., 1, 1] = stds[..., 1]
        samples[..., 2, 2] = stds[..., 2]

        eye_mat = torch.cat([torch.eye(3), -torch.eye(3)],
                            0)  # 6 * 3, to put gaussians along the axis
        eye_mat = eye_mat.reshape(1, 1, 6, 3).repeat(B, N, 1,
                                                     1).to(stds.device)
        samples = (eye_mat @ samples).squeeze(-1)

        # st()
        # means = torch.zeros_like(stds)
        # samples = torch.normal(mean=means, std=stds)  # B f*N 3

        rots = rearrange(build_rotation(
            rearrange(rotation, 'B N ... -> (B N) ...')),
                         '(B N) ... -> B N ...',
                         B=B,
                         N=N)
        rots = rots.unsqueeze(2).repeat_interleave(self.f, dim=2)  # B f*N 3 3

        # torch.bmm only supports ndim=3 Tensor
        # new_xyz = torch.matmul(rots, samples.unsqueeze(-1)).squeeze(-1) + pos.repeat_interleave(self.f, dim=1)
        # st()

        # new_xyz = torch.matmul(rots, samples.unsqueeze(-1)).squeeze(-1) + pos.repeat_interleave(self.f, dim=1)
        new_xyz = (rots @ samples.unsqueeze(-1)).squeeze(-1) + pos.unsqueeze(
            2).repeat_interleave(self.f, dim=2)  # B N 6 3
        new_xyz = rearrange(new_xyz, 'b n f c -> b (n f) c')

        # ! not considering rotation here
        # new_xyz = samples + pos.repeat_interleave(
        #     self.f, dim=1)  # ! no rotation for now
        # new_xyz: B f*N 3

        # ! new points to features
        new_xyz_embed = xyz_embed_fn(new_xyz)
        new_gaussian_embed = self.write_ca(
            new_xyz_embed, latent_from_vit
        )  # ! use z_(i+1), rather than gaussian_base_feat here

        # ! predict gaussians residuals
        gaussian_residual_pre_activate = self.gaussian_residual_pred(
            new_gaussian_embed)

        # ! add back. how to deal with new rotations? check the range first.
        # scaling and rotation.
        # gaussian_base_pre_activate[split_mask][
        #     4:7] -= 1  # reduce scale for those points

        gaussian_base_pre_activate_repeat = gaussian_base_pre_activate.repeat_interleave(
            self.f, dim=1)

        # new scaling
        # ! pre-activate scaling value, shall be negative? since more values are 0.1 before softplus.
        # TODO wrong here, shall get new scaling before repeat
        gaussians = gaussian_residual_pre_activate + gaussian_base_pre_activate_repeat  # learn the residual

        # new_gaussians_pos = new_xyz + self.residual_offset_act(
        #     gaussians[..., :3])

        return gaussians, new_xyz, latent_from_vit, new_gaussian_embed  # return positions independently


class GS_Adaptive_Read_Write_CA_adaptive_f14(GS_Adaptive_Read_Write_CA_adaptive
                                             ):

    def __init__(self,
                 query_dim,
                 context_dim,
                 mlp_ratio,
                 vit_heads,
                 f=4,
                 heads=8,
                 dim_head=64,
                 dropout=0,
                 depth=2,
                 vit_blk=DiTBlock2) -> None:
        super().__init__(query_dim, context_dim, mlp_ratio, vit_heads, f,
                         heads, dim_head, dropout, depth, vit_blk)

        corner_mat = torch.empty(8, 3)
        counter = 0
        for i in range(-1, 3, 2):
            for j in range(-1, 3, 2):
                for k in range(-1, 3, 2):
                    corner_mat[counter] = torch.Tensor([i, j, k])
                    counter += 1

        self.corner_mat = corner_mat.contiguous().to(dist_util.dev()).reshape(
            1, 1, 8, 3)

    def forward(self, gaussians_base, gaussian_base_pre_activate,
                gaussian_base_feat, latent_from_vit, vae_latent, xyz_embed_fn):
        # gaussians_base: xyz_base after activations and deform offset
        # xyz_base: original features (before activations)

        # ========= START read CA ========
        latent_from_vit = self.read_ca(latent_from_vit,
                                       gaussian_base_feat)  # z_i -> z_(i+1)

        for blk_idx, block in enumerate(self.point_infinity_blocks):
            latent_from_vit = block(latent_from_vit,
                                    vae_latent)  # vae_latent: c

        # ========= END read CA ========

        # ! use point embedder, or other features?
        # base_gaussian_xyz_embed = xyz_embed_fn(gaussians_base[..., :3])

        # x = self.densify_and_split(gaussians_base, base_gaussian_xyz_embed)

        # ! densify
        B, N = gaussians_base.shape[:2]  # gaussians upsample factor
        # n_init_points = self.get_xyz.shape[0]

        pos, opacity, scaling, rotation = gaussians_base[
            ..., 0:3], gaussians_base[..., 3:4], gaussians_base[
                ..., 4:7], gaussians_base[..., 7:11]

        # ! filter clone/densify based on scaling range

        split_mask = scaling.max(
            dim=-1
        )[0] > self.scene_extent * self.percent_dense  # shape: B 4096

        # clone_mask = ~split_mask

        # stds = scaling.repeat_interleave(self.f, dim=1)  #  B 13824 3
        # stds = scaling.unsqueeze(1).repeat_interleave(self.f, dim=1)  #  B 6 13824 3
        stds = scaling  #  B 13824 3

        # TODO, in mat form. axis aligned creation.
        samples = torch.zeros(B, N, 3, 3).to(stds.device)

        samples[..., 0, 0] = stds[..., 0]
        samples[..., 1, 1] = stds[..., 1]
        samples[..., 2, 2] = stds[..., 2]

        eye_mat = torch.cat([torch.eye(3), -torch.eye(3)],
                            0)  # 6 * 3, to put gaussians along the axis
        eye_mat = eye_mat.reshape(1, 1, 6, 3).repeat(B, N, 1,
                                                     1).to(stds.device)
        samples = (eye_mat @ samples).squeeze(-1)  # B N 6 3

        # ! create corner
        samples_corner = stds.clone().unsqueeze(-2).repeat(1, 1, 8,
                                                           1)  # B N 8 3
        # samples_corner = torch.ones(B,N,8,3).to(stds.device) # B N 8 3
        # counter = 0
        # for i in range(-1,3,2):
        #     for j in range(-1,3,2):
        #         for k in range(-1,3,2):
        #             samples_corner[..., counter, :] *= torch.Tensor([i,j,k]).to(stds.device)
        #             counter += 1

        # ! optimize with matmul, register to self
        samples_corner = torch.mul(samples_corner, self.corner_mat)

        samples = torch.cat([samples, samples_corner], -2)
        # means = torch.zeros_like(stds)
        # samples = torch.normal(mean=means, std=stds)  # B f*N 3

        rots = rearrange(build_rotation(
            rearrange(rotation, 'B N ... -> (B N) ...')),
                         '(B N) ... -> B N ...',
                         B=B,
                         N=N)
        rots = rots.unsqueeze(2).repeat_interleave(self.f, dim=2)  # B f*N 3 3

        # torch.bmm only supports ndim=3 Tensor
        # new_xyz = torch.matmul(rots, samples.unsqueeze(-1)).squeeze(-1) + pos.repeat_interleave(self.f, dim=1)
        # st()

        # new_xyz = torch.matmul(rots, samples.unsqueeze(-1)).squeeze(-1) + pos.repeat_interleave(self.f, dim=1)
        new_xyz = (rots @ samples.unsqueeze(-1)).squeeze(-1) + pos.unsqueeze(
            2).repeat_interleave(self.f, dim=2)  # B N 6 3
        new_xyz = rearrange(new_xyz, 'b n f c -> b (n f) c')

        # ! not considering rotation here
        # new_xyz = samples + pos.repeat_interleave(
        #     self.f, dim=1)  # ! no rotation for now
        # new_xyz: B f*N 3

        # ! new points to features
        new_xyz_embed = xyz_embed_fn(new_xyz)
        new_gaussian_embed = self.write_ca(
            new_xyz_embed, latent_from_vit
        )  # ! use z_(i+1), rather than gaussian_base_feat here

        # ! predict gaussians residuals
        gaussian_residual_pre_activate = self.gaussian_residual_pred(
            new_gaussian_embed)

        # ! add back. how to deal with new rotations? check the range first.
        # scaling and rotation.
        # gaussian_base_pre_activate[split_mask][
        #     4:7] -= 1  # reduce scale for those points

        gaussian_base_pre_activate_repeat = gaussian_base_pre_activate.repeat_interleave(
            self.f, dim=1)

        # new scaling
        # ! pre-activate scaling value, shall be negative? since more values are 0.1 before softplus.
        # TODO wrong here, shall get new scaling before repeat
        gaussians = gaussian_residual_pre_activate + gaussian_base_pre_activate_repeat  # learn the residual

        # new_gaussians_pos = new_xyz + self.residual_offset_act(
        #     gaussians[..., :3])

        return gaussians, new_xyz, latent_from_vit, new_gaussian_embed  # return positions independently


# add local deformation
class splatting_dit_v4_PI_V1_trilatent_sphere_S_residual(
        splatting_dit_v4_PI_V1_trilatent_sphere_S_48):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=48,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         pcd_reso, **kwargs)

        # xyz_embed_dim = vit_decoder.embed_dim // 2  # 384
        self.superresolution.update(
            dict(
                ada_CA_f4_1=GS_Adaptive_Write_CA(self.embed_dim,
                                                 vit_decoder.embed_dim,
                                                 f=4,
                                                 heads=6),  # write
            ))
        self.aug_opacity_prob = 0.0

    def _get_base_pcd(self, x):

        # x = ret_after_decoder['latent_after_vit']
        B, N, C = x.shape  # B C D H W, 14-dim voxel features
        offsets = self.offset_act(x[..., 0:3])  # [B, N, 3] # ! learned offsets
        pos = offsets + self.g_cube.repeat_interleave(B, dim=0)

        return self._gaussian_pred_activations(pos, x)

        # opacity = self.opacity_act(x[..., 3:4])

        # scale = self.scale_act(x[..., 4:7])

        # rotation = self.rot_act(x[..., 7:11])
        # rgbs = self.rgb_act(x[..., 11:])

        # gaussians = torch.cat([pos, opacity, scale, rotation, rgbs],
        #                       dim=-1)  # [B, N, 14]

        # return gaussians

    # ! decode
    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        B = latent_from_vit.shape[0]

        # ! base, fixed
        xyz_emb = self.superresolution['xyz_pos_embed'](
            self.g_cube).repeat_interleave(B, 0)  # B 14 H W
        # st()
        gaussian_base_pre_activate, gaussian_base_feat = self.superresolution[
            'conv_sr'](xyz_emb, latent_from_vit, return_x=True)  # B 14 H W

        # (Pdb) gaussian_base_feat.shape
        # torch.Size([2, 4096, 768])
        # (Pdb) p gaussian_base_pre_activate.shape
        # torch.Size([2, 4096, 14])

        # ! learn residual, f=4
        # st()
        # latent = None
        gaussians_base = self._get_base_pcd(gaussian_base_pre_activate)
        gaussians_upsampled_preactivate, gaussians_upsampled_pos = self.superresolution[
            'ada_CA_f4_1'](gaussians_base, gaussian_base_pre_activate,
                           gaussian_base_feat,
                           self.superresolution['xyz_pos_embed'])
        gaussians_upsampled = self._gaussian_pred_activations(
            pos=gaussians_upsampled_pos, x=gaussians_upsampled_preactivate)

        latent_after_vit = {
            'gaussians_base': gaussians_base,
            'gaussians_upsampled': gaussians_upsampled,
        }

        ret_dict.update(
            dict(cls_token=cls_token, latent_after_vit=latent_after_vit))

        return ret_dict

    def _gaussian_pred_activations(self, pos, x):
        # if pos is None:
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs],
                              dim=-1)  # [B, N, 14]

        return gaussians

    def forward_gaussians(self, ret_after_decoder, c=None):
        # ! create grid and deform Gaussians accordingly
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        # ! placeholder for compat
        ret_after_decoder['gaussians'] = ret_after_decoder['latent_after_vit'][
            'gaussians_upsampled']

        if self.aug_opacity_prob > 0:
            if np.random.rand() < self.aug_opacity_prob:  # uniform sample
                ret_after_decoder['gaussians'][
                    ..., 3:4] = 1  # set opacity to 1, avoid floating points

        # ret_after_decoder['gaussians'][
        #     ..., 3:4] = 1  # set opacity to 1, avoid floating points

        # st()
        # self.vis_gaussian(
        #     ret_after_decoder['latent_after_vit']['gaussians_base'],
        #     'augbg-base')
        # self.vis_gaussian(
        #     ret_after_decoder['latent_after_vit']['gaussians_upsampled'],
        #     'augbg-f4')
        # st()

        ret_after_decoder['pos'] = ret_after_decoder['latent_after_vit'][
            'gaussians_upsampled'][..., 0:3]  #
        # x = ret_after_decoder['latent_after_vit']

        return ret_after_decoder

    def forward_vit_decoder(self, x, img_size=None):
        return super().forward_vit_decoder(x, img_size)


class splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_augopacity(
        splatting_dit_v4_PI_V1_trilatent_sphere_S_residual):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=181,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         pcd_reso, **kwargs)
        self.aug_opacity_prob = 0.2


class splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_commitLoss(
        splatting_dit_v4_PI_V1_trilatent_sphere_S_residual):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=181,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         pcd_reso, **kwargs)


class splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_addReadCA(
        splatting_dit_v4_PI_V1_trilatent_sphere_S_residual):
    # add joint render also
    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=181,
            f=4,
            joint_render=True,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         pcd_reso, **kwargs)
        # when doing cross attention in the second stage write, use z as context. before that, add a "read" cross attention.

        self.joint_render = joint_render  # same performance when doing it together

        self.superresolution.update(
            dict(
                ada_CA_f4_1=GS_Adaptive_Read_Write_CA(
                    self.embed_dim,
                    vit_decoder.embed_dim,
                    vit_heads=vit_decoder.num_heads,
                    mlp_ratio=vit_decoder.mlp_ratio,
                    depth=vit_decoder.depth // 6,
                    f=f,
                    heads=8),  # write
            ))

    def forward_gaussians(self, ret_after_decoder, c=None):
        # ! create grid and deform Gaussians accordingly
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        # ! merge gaussians

        if self.joint_render:
            ret_after_decoder['gaussians'] = torch.cat([
                ret_after_decoder['latent_after_vit']['gaussians_base'],
                ret_after_decoder['latent_after_vit']['gaussians_upsampled'],
            ],
                                                       dim=1)
        else:
            ret_after_decoder['gaussians'] = ret_after_decoder[
                'latent_after_vit']['gaussians_upsampled']

        if self.aug_opacity_prob > 0:
            if np.random.rand() < self.aug_opacity_prob:  # uniform sample
                ret_after_decoder['gaussians'][
                    ..., 3:4] = 1  # set opacity to 1, avoid floating points

        # st()
        # self.vis_gaussian(ret_after_decoder['latent_after_vit']['gaussians_base'], 'base')
        # self.vis_gaussian(ret_after_decoder['latent_after_vit']['gaussians_upsampled'], 'f25')
        # st()

        # ! but still supervising the base gaussians for EMD
        ret_after_decoder['pos'] = ret_after_decoder['latent_after_vit'][
            'gaussians_base'][..., 0:3]  #
        # x = ret_after_decoder['latent_after_vit']

        return ret_after_decoder

    # ! decode
    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        vae_latent, latent_from_vit = latent_from_vit[
            'latent'], latent_from_vit['latent_from_vit']
        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        B = latent_from_vit.shape[0]

        # ! base, fixed
        xyz_emb = self.superresolution['xyz_pos_embed'](
            self.g_cube).repeat_interleave(B, 0)  # B 14 H W
        # st()
        gaussian_base_pre_activate, gaussian_base_feat = self.superresolution[
            'conv_sr'](xyz_emb, latent_from_vit, return_x=True)  # B 14 H W

        # (Pdb) gaussian_base_feat.shape
        # torch.Size([2, 4096, 768])
        # (Pdb) p gaussian_base_pre_activate.shape
        # torch.Size([2, 4096, 14])

        # ! learn residual, f=4
        # latent = None
        gaussians_base = self._get_base_pcd(gaussian_base_pre_activate)

        gaussians_upsampled_preactivate, gaussians_upsampled_pos, _, _ = self.superresolution[
            'ada_CA_f4_1'](gaussians_base, gaussian_base_pre_activate,
                           gaussian_base_feat, latent_from_vit, vae_latent,
                           self.superresolution['xyz_pos_embed'])

        gaussians_upsampled = self._gaussian_pred_activations(
            pos=gaussians_upsampled_pos, x=gaussians_upsampled_preactivate)

        latent_after_vit = {
            'gaussians_base': gaussians_base,
            'gaussians_upsampled': gaussians_upsampled,
        }

        ret_dict.update(
            dict(cls_token=cls_token, latent_after_vit=latent_after_vit))

        return ret_dict

    def vit_decode_backbone(self, latent, img_size):
        # assert x.ndim == 3  # N L C
        if isinstance(latent, dict):
            latent = latent['latent_normalized_2Ddiffusion']  # B, C*3, H, W

        # assert latent.shape == (
        #     latent.shape[0], 3 * (self.token_size * self.vae_p)**2,
        #     self.ldm_z_channels), f'latent.shape: {latent.shape}'

        # st() # latent: B 12 32 32
        latent = self.superresolution['ldm_upsample'](  # ! B 768 (3*256) 768 
            latent)  # torch.Size([8, 12, 32, 32]) => torch.Size([8, 256, 768])
        # latent: torch.Size([8, 768, 768])
        # st()

        # ! directly feed to vit_decoder
        return {
            'latent': latent,
            'latent_from_vit': self.forward_vit_decoder(latent, img_size)
        }  # pred_vit_latent

    def triplane_decode(self, ret_after_gaussian_forward, c, **kwargs):
        return super().triplane_decode(ret_after_gaussian_forward, c, **kwargs)


class splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_addReadCA_f9(
        splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_addReadCA):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=181,
            f=4,
            **kwargs) -> None:
        super().__init__(vit_decoder,
                         triplane_decoder,
                         cls_token,
                         normalize_feat,
                         sr_ratio,
                         use_fusion_blk,
                         fusion_blk_depth,
                         fusion_blk,
                         channel_multiplier,
                         pcd_reso,
                         f=9,
                         **kwargs)


class splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_addReadCA_f8(
        splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_addReadCA):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=181,
            f=4,
            **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            fusion_blk,
            channel_multiplier,
            pcd_reso,
            f=8,
            joint_render=False,  # !
            **kwargs)


class splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_addReadCA_f8_adaptive(
        splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_addReadCA):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=181,
            f=4,
            **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            fusion_blk,
            channel_multiplier,
            pcd_reso,
            f=8,
            joint_render=False,  # !
            **kwargs)

        self.superresolution.update(
            dict(
                ada_CA_f4_1=GS_Adaptive_Read_Write_CA_adaptive(
                    self.embed_dim,
                    vit_decoder.embed_dim,
                    vit_heads=vit_decoder.num_heads,
                    mlp_ratio=vit_decoder.mlp_ratio,
                    depth=vit_decoder.depth // 6,
                    f=6,  # anchored on the scale
                    heads=8),  # write
            ))


class splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_addReadCA_f14_adaptive(
        splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_addReadCA):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=181,
            f=4,
            **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            fusion_blk,
            channel_multiplier,
            pcd_reso,
            f=8,
            joint_render=False,  # !
            **kwargs)

        self.superresolution.update(  # f=14 here
            dict(
                ada_CA_f4_1=GS_Adaptive_Read_Write_CA_adaptive_f14(
                    self.embed_dim,
                    vit_decoder.embed_dim,
                    vit_heads=vit_decoder.num_heads,
                    mlp_ratio=vit_decoder.mlp_ratio,
                    depth=vit_decoder.depth // 6,
                    f=14,  # anchored on the scale
                    heads=8),  # write
            ))


# ! check prepend transformer upsampling


class GS_Adaptive_Read_Write_CA_adaptive_f14_prepend(
        GS_Adaptive_Read_Write_CA_adaptive):

    def __init__(self,
                 query_dim,
                 context_dim,
                 mlp_ratio,
                 vit_heads,
                 f=4,
                 heads=8,
                 dim_head=64,
                 dropout=0,
                 depth=2,
                 vit_blk=DiTBlock2, 
                 no_flash_op=False,) -> None:
        super().__init__(query_dim, context_dim, mlp_ratio, vit_heads, f,
                         heads, dim_head, dropout, depth, vit_blk)

        # corner_mat = torch.empty(8,3)
        # counter = 0
        # for i in range(-1,3,2):
        #     for j in range(-1,3,2):
        #         for k in range(-1,3,2):
        #             corner_mat[counter] = torch.Tensor([i,j,k])
        #             counter += 1

        # self.corner_mat=corner_mat.contiguous().to(dist_util.dev()).reshape(1,1,8,3)

        del self.read_ca, self.write_ca
        del self.point_infinity_blocks

        # ? why not saved to checkpoint
        # self.latent_embedding = nn.Parameter(torch.randn(1, f, query_dim)).to(
        #     dist_util.dev())

        # ! not .cuda() here
        self.latent_embedding = nn.Parameter(torch.randn(1, f, query_dim),
                                             requires_grad=True)

        self.transformer = SRT_TX(
            context_dim,  # 12 * 64 = 768
            depth=depth,
            heads=context_dim // 64,  # vit-b default.
            mlp_dim=4 * context_dim,  # 1536 by default
            no_flash_op=no_flash_op,
        )

        # self.offset_act = lambda x: torch.tanh(x) * (self.scene_range[
        #     1]) * 0.5  # regularize small offsets

    def forward(self, gaussians_base, gaussian_base_pre_activate,
                gaussian_base_feat, latent_from_vit, vae_latent, xyz_embed_fn,
                offset_act):
        # gaussians_base: xyz_base after activations and deform offset
        # xyz_base: original features (before activations)

        # ========= START read CA ========
        # latent_from_vit = self.read_ca(latent_from_vit,
        #                                gaussian_base_feat)  # z_i -> z_(i+1)

        # for blk_idx, block in enumerate(self.point_infinity_blocks):
        #     latent_from_vit = block(latent_from_vit,
        #                             vae_latent)  # vae_latent: c

        # ========= END read CA ========

        # ! use point embedder, or other features?
        # base_gaussian_xyz_embed = xyz_embed_fn(gaussians_base[..., :3])

        # x = self.densify_and_split(gaussians_base, base_gaussian_xyz_embed)

        # ! densify
        B, N = gaussians_base.shape[:2]  # gaussians upsample factor
        # n_init_points = self.get_xyz.shape[0]

        pos, opacity, scaling, rotation = gaussians_base[
            ..., 0:3], gaussians_base[..., 3:4], gaussians_base[
                ..., 4:7], gaussians_base[..., 7:11]

        # ! filter clone/densify based on scaling range
        """

        # split_mask = scaling.max(
        #     dim=-1
        # )[0] > self.scene_extent * self.percent_dense  # shape: B 4096

        stds = scaling  #  B 13824 3

        # TODO, in mat form. axis aligned creation.
        samples = torch.zeros(B, N, 3, 3).to(stds.device) 

        samples[..., 0,0] = stds[..., 0]
        samples[..., 1,1] = stds[..., 1]
        samples[..., 2,2] = stds[..., 2]

        eye_mat = torch.cat([torch.eye(3), -torch.eye(3)], 0) # 6 * 3, to put gaussians along the axis
        eye_mat = eye_mat.reshape(1,1,6,3).repeat(B, N, 1, 1).to(stds.device)
        samples = (eye_mat @ samples).squeeze(-1) # B N 6 3

        # ! create corner
        samples_corner = stds.clone().unsqueeze(-2).repeat(1,1,8,1) # B N 8 3

        # ! optimize with matmul, register to self
        samples_corner = torch.mul(samples_corner,self.corner_mat)

        samples = torch.cat([samples, samples_corner], -2)

        rots = rearrange(build_rotation(rearrange(rotation, 'B N ... -> (B N) ...')), '(B N) ... -> B N ...', B=B, N=N)
        rots = rots.unsqueeze(2).repeat_interleave(self.f, dim=2) # B f*N 3 3

        new_xyz = (rots @ samples.unsqueeze(-1)).squeeze(-1) + pos.unsqueeze(2).repeat_interleave(self.f, dim=2) # B N 6 3
        new_xyz = rearrange(new_xyz, 'b n f c -> b (n f) c')
        
        # ! new points to features
        new_xyz_embed = xyz_embed_fn(new_xyz)
        new_gaussian_embed = self.write_ca(
            new_xyz_embed, latent_from_vit
        )  # ! use z_(i+1), rather than gaussian_base_feat here

        """

        # ! [global_emb, local_emb, learnable_query_emb] self attention -> fetch last K tokens as the learned query -> add to base

        # ! query from local point emb
        global_local_query_emb = torch.cat(
            [
                # rearrange(latent_from_vit.unsqueeze(1).expand(-1,N,-1,-1), 'B N L C -> (B N) L C'), # 8, 768, 1024. expand() returns a new view.
                rearrange(gaussian_base_feat,
                          'B N C -> (B N) 1 C'),  # 8, 2304, 1024 -> 8*2304 1 C
                self.latent_embedding.repeat(B * N, 1,
                                             1)  # 1, 14, 1024 -> B*N 14 1024
            ],
            dim=1)  # OOM if prepend global feat
        global_local_query_emb = self.transformer(
            global_local_query_emb)  # torch.Size([18432, 15, 1024])
        # st() # do self attention

        # ! query from global shape emb
        # new_gaussian_embed = self.write_ca(
        #     global_local_query_emb,
        #     rearrange(latent_from_vit.unsqueeze(1).expand(-1,N,-1,-1), 'B N L C -> (B N) L C'),
        # )  # ! use z_(i+1), rather than gaussian_base_feat here

        # ! predict gaussians residuals
        gaussian_residual_pre_activate = self.gaussian_residual_pred(
            global_local_query_emb[:, 1:, :])

        gaussian_residual_pre_activate = rearrange(
            gaussian_residual_pre_activate, '(B N) L C -> B N L C', B=B,
            N=N)  # B 2304 14 C
        # TODO here
        # ? new_xyz from where
        offsets = offset_act(gaussian_residual_pre_activate[..., 0:3])
        new_xyz = offsets + pos.unsqueeze(2).repeat_interleave(
            self.f, dim=2)  # B N F 3
        new_xyz = rearrange(new_xyz, 'b n f c -> b (n f) c')

        gaussian_base_pre_activate_repeat = gaussian_base_pre_activate.unsqueeze(
            -2).expand(-1, -1, self.f, -1)  # avoid new memory allocation
        gaussians = rearrange(gaussian_residual_pre_activate +
                              gaussian_base_pre_activate_repeat,
                              'B N F C -> B (N F) C',
                              B=B,
                              N=N)  # learn the residual in the feature space

        # return gaussians, new_xyz, latent_from_vit, new_gaussian_embed  # return positions independently
        # return gaussians, latent_from_vit, new_gaussian_embed  # return positions independently
        return gaussians, new_xyz


class GS_Adaptive_Read_Write_CA_adaptive_2dgs(
        GS_Adaptive_Read_Write_CA_adaptive_f14_prepend):

    def __init__(self,
                 query_dim,
                 context_dim,
                 mlp_ratio,
                 vit_heads,
                 f=16,
                 heads=8,
                 dim_head=64,
                 dropout=0,
                 depth=2,
                 vit_blk=DiTBlock2, 
                 no_flash_op=False,
                 cross_attention=False,) -> None:
        super().__init__(query_dim, context_dim, mlp_ratio, vit_heads, f,
                         heads, dim_head, dropout, depth, vit_blk, no_flash_op)

        # del self.gaussian_residual_pred # will use base one

        self.cross_attention = cross_attention
        if cross_attention: # since much efficient than self attention, linear complexity
            # del self.transformer
            self.sr_ca = CrossAttention(query_dim, context_dim, # xformers fails large batch size: https://github.com/facebookresearch/xformers/issues/845
                                                      heads, dim_head, dropout, 
                                                      no_flash_op=no_flash_op)

        # predict residual over base (features)
        self.gaussian_residual_pred = PreNorm(  # add prenorm since using pre-norm TX as the sr module
            query_dim, nn.Linear(query_dim, 13, bias=True))

        # init as full zero, since predicting residual here
        nn.init.constant_(self.gaussian_residual_pred.fn.weight, 0)
        nn.init.constant_(self.gaussian_residual_pred.fn.bias, 0)

    def forward(self,
                latent_from_vit,
                base_gaussians,
                skip_weight,
                offset_act,
                gs_pred_fn,
                gs_act_fn,
                gaussian_base_pre_activate=None):
        B, N, C = latent_from_vit.shape  # e.g., B 768 768

        if not self.cross_attention:
            # ! query from local point emb
            global_local_query_emb = torch.cat(
                [
                    rearrange(latent_from_vit,
                            'B N C -> (B N) 1 C'),  # 8, 2304, 1024 -> 8*2304 1 C
                    self.latent_embedding.repeat(B * N, 1, 1).to(
                        latent_from_vit)  # 1, 14, 1024 -> B*N 14 1024
                ],
                dim=1)  # OOM if prepend global feat

            global_local_query_emb = self.transformer(
                global_local_query_emb)  # torch.Size([18432, 15, 1024])

            # ! add residuals to the base features
            global_local_query_emb = rearrange(global_local_query_emb[:, 1:],
                                            '(B N) L C -> B N L C',
                                            B=B,
                                            N=N)  # B N C f
        else:

            # st()
            # for xformers debug
            # global_local_query_emb = self.sr_ca( self.latent_embedding.repeat(B, 1, 1).to( latent_from_vit).contiguous(), latent_from_vit[:, 0:1, :],)
            # st()

            # self.sr_ca( self.latent_embedding.repeat(B * N, 1, 1).to(latent_from_vit)[:8000], rearrange(latent_from_vit, 'B N C -> (B N) 1 C')[:8000],).shape
            global_local_query_emb = self.sr_ca( self.latent_embedding.repeat(B * N, 1, 1).to(latent_from_vit), rearrange(latent_from_vit, 'B N C -> (B N) 1 C'),)

            global_local_query_emb = self.transformer(
                global_local_query_emb)  # torch.Size([18432, 15, 1024])

            # ! add residuals to the base features
            global_local_query_emb = rearrange(global_local_query_emb,
                                            '(B N) L C -> B N L C',
                                            B=B,
                                            N=N)  # B N C f

        # * predict residual features
        gaussian_residual_pre_activate = self.gaussian_residual_pred(
            global_local_query_emb)

        # ! directly add xyz offsets 
        offsets = offset_act(gaussian_residual_pre_activate[..., :3])

        gaussians_upsampled_pos = offsets + einops.repeat(
            base_gaussians[..., :3], 'B N C -> B N F C',
            F=self.f)  # ! reasonable init

        # ! add residual features
        gaussian_residual_pre_activate = gaussian_residual_pre_activate + einops.repeat(
            gaussian_base_pre_activate, 'B N C -> B N F C', F=self.f)

        gaussians_upsampled = gs_act_fn(pos=gaussians_upsampled_pos,
                                        x=gaussian_residual_pre_activate)

        gaussians_upsampled = rearrange(gaussians_upsampled,
                                        'B N F C -> B (N F) C')

        return gaussians_upsampled, (rearrange(
            gaussian_residual_pre_activate, 'B N F C -> B (N F) C'
        ), rearrange(
            global_local_query_emb, 'B N F C -> B (N F) C'
        ))


class splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_addReadCA_f14_adaptive_prepend(
        splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_addReadCA):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=181,
            f=4,
            **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            fusion_blk,
            channel_multiplier,
            pcd_reso,
            f=8,
            joint_render=True,  # !
            **kwargs)

        self.superresolution.update(  # f=14 here
            dict(
                ada_CA_f4_1=GS_Adaptive_Read_Write_CA_adaptive_f14_prepend(
                    self.embed_dim,
                    vit_decoder.embed_dim,
                    vit_heads=vit_decoder.num_heads,
                    mlp_ratio=vit_decoder.mlp_ratio,
                    depth=vit_decoder.depth // 6,
                    f=14,  # anchored on the scale
                    heads=8),  # write
            ))

    # ! decode
    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        vae_latent, latent_from_vit = latent_from_vit[
            'latent'], latent_from_vit['latent_from_vit']
        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        B = latent_from_vit.shape[0]

        # ! base, fixed
        xyz_emb = self.superresolution['xyz_pos_embed'](
            self.g_cube).repeat_interleave(B, 0)  # B 14 H W
        # st()
        gaussian_base_pre_activate, gaussian_base_feat = self.superresolution[
            'conv_sr'](xyz_emb, latent_from_vit, return_x=True)  # B 14 H W

        # (Pdb) gaussian_base_feat.shape
        # torch.Size([2, 4096, 768])
        # (Pdb) p gaussian_base_pre_activate.shape
        # torch.Size([2, 4096, 14])

        # ! learn residual, f=4
        # latent = None
        gaussians_base = self._get_base_pcd(gaussian_base_pre_activate)

        gaussians_upsampled_preactivate, gaussians_upsampled_pos = self.superresolution[
            'ada_CA_f4_1'](gaussians_base, gaussian_base_pre_activate,
                           gaussian_base_feat, latent_from_vit, vae_latent,
                           self.superresolution['xyz_pos_embed'],
                           self.offset_act)

        gaussians_upsampled = self._gaussian_pred_activations(
            pos=gaussians_upsampled_pos, x=gaussians_upsampled_preactivate)

        latent_after_vit = {
            'gaussians_base': gaussians_base,
            'gaussians_upsampled': gaussians_upsampled,
        }

        ret_dict.update(
            dict(cls_token=cls_token, latent_after_vit=latent_after_vit))

        return ret_dict


class splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_addReadCA_f25_adaptive_prepend(
        splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_addReadCA_f14_adaptive_prepend
):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=181,
            f=4,
            **kwargs) -> None:
        super().__init__(vit_decoder,
                         triplane_decoder,
                         cls_token,
                         normalize_feat,
                         sr_ratio,
                         use_fusion_blk,
                         fusion_blk_depth,
                         fusion_blk,
                         channel_multiplier,
                         pcd_reso,
                         f=8,
                         **kwargs)

        self.superresolution.update(  # f=14 here
            dict(
                ada_CA_f4_1=GS_Adaptive_Read_Write_CA_adaptive_f14_prepend(
                    self.embed_dim,
                    vit_decoder.embed_dim,
                    vit_heads=vit_decoder.num_heads,
                    mlp_ratio=vit_decoder.mlp_ratio,
                    depth=vit_decoder.depth // 6,
                    f=25,  # anchored on the scale
                    heads=8),  # write
            ))


# ! end SR exp


class splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_addReadCA_f8_adaptive_jointrender(
        splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_addReadCA):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=181,
            f=4,
            **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            normalize_feat,
            sr_ratio,
            use_fusion_blk,
            fusion_blk_depth,
            fusion_blk,
            channel_multiplier,
            pcd_reso,
            f=8,
            joint_render=True,  # !
            **kwargs)

        self.superresolution.update(
            dict(
                ada_CA_f4_1=GS_Adaptive_Read_Write_CA_adaptive(
                    self.embed_dim,
                    vit_decoder.embed_dim,
                    vit_heads=vit_decoder.num_heads,
                    mlp_ratio=vit_decoder.mlp_ratio,
                    depth=vit_decoder.depth // 6,
                    f=6,  # anchored on the scale
                    heads=8),  # write
            ))


class splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_addReadCA_f2(
        splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_addReadCA_f8_adaptive
):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=181,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         pcd_reso, **kwargs)

        self.superresolution.update(
            dict(
                ada_CA_f4_2=GS_Adaptive_Read_Write_CA_adaptive(
                    self.embed_dim,
                    vit_decoder.embed_dim,
                    vit_heads=vit_decoder.num_heads,
                    mlp_ratio=vit_decoder.mlp_ratio,
                    depth=vit_decoder.depth // 6,
                    f=6,  # anchored on the scale
                    heads=8),  # write
            ))

    def forward_gaussians(self, ret_after_decoder, c=None):
        # ! create grid and deform Gaussians accordingly
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        # ! merge gaussians

        # ! for joint rendering
        ret_after_decoder['gaussians'] = torch.cat(
            [
                # ret_after_decoder['latent_after_vit']['gaussians_base'],
                ret_after_decoder['latent_after_vit']['gaussians_upsampled'],
                ret_after_decoder['latent_after_vit']['gaussians_upsampled_2'],
            ],
            dim=1)

        if self.aug_opacity_prob > 0:
            if np.random.rand() < self.aug_opacity_prob:  # uniform sample
                ret_after_decoder['gaussians'][
                    ..., 3:4] = 1  # set opacity to 1, avoid floating points

        # ret_after_decoder['gaussians'][
        #     ..., 3:4] = 1  # set opacity to 1, avoid floating points

        # st()
        # self.vis_gaussian(ret_after_decoder['latent_after_vit']['gaussians_upsampled'], 'augbg-base')
        # self.vis_gaussian(ret_after_decoder['latent_after_vit']['gaussians_upsampled'], 'augbg-f4')
        # st()

        ret_after_decoder['pos'] = ret_after_decoder['latent_after_vit'][
            'gaussians_base'][..., 0:3]  #
        # x = ret_after_decoder['latent_after_vit']

        return ret_after_decoder

    # ! decode
    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        vae_latent, latent_from_vit = latent_from_vit[
            'latent'], latent_from_vit['latent_from_vit']
        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        B = latent_from_vit.shape[0]

        # ! base, fixed
        xyz_emb = self.superresolution['xyz_pos_embed'](
            self.g_cube).repeat_interleave(B, 0)  # B 14 H W
        # st()
        gaussian_base_pre_activate, gaussian_base_feat = self.superresolution[
            'conv_sr'](xyz_emb, latent_from_vit, return_x=True)  # B 14 H W

        # (Pdb) gaussian_base_feat.shape
        # torch.Size([2, 4096, 768])
        # (Pdb) p gaussian_base_pre_activate.shape
        # torch.Size([2, 4096, 14])

        # ! learn residual, f=4
        # latent = None
        gaussians_base = self._get_base_pcd(gaussian_base_pre_activate)

        gaussians_upsampled_preactivate, gaussians_upsampled_pos, latent_from_vit, gaussian_upsampled_embed = self.superresolution[
            'ada_CA_f4_1'](gaussians_base, gaussian_base_pre_activate,
                           gaussian_base_feat, latent_from_vit, vae_latent,
                           self.superresolution['xyz_pos_embed'])

        gaussians_upsampled = self._gaussian_pred_activations(
            pos=gaussians_upsampled_pos, x=gaussians_upsampled_preactivate)

        # ! another upsampling

        gaussians_upsampled_preactivate_2, gaussians_upsampled_pos_2, _, _ = self.superresolution[
            'ada_CA_f4_2'](gaussians_upsampled,
                           gaussians_upsampled_preactivate,
                           gaussian_upsampled_embed, latent_from_vit,
                           vae_latent, self.superresolution['xyz_pos_embed'])

        gaussians_upsampled_2 = self._gaussian_pred_activations(
            pos=gaussians_upsampled_pos_2, x=gaussians_upsampled_preactivate_2)

        latent_after_vit = {
            'gaussians_base': gaussians_base,
            'gaussians_upsampled': gaussians_upsampled,
            'gaussians_upsampled_2': gaussians_upsampled_2,
        }

        ret_dict.update(
            dict(cls_token=cls_token, latent_after_vit=latent_after_vit))

        return ret_dict


class splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_addReadCA_f8_adaptive_jointrender_tristage(
        splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_addReadCA_f2):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=181,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         pcd_reso, **kwargs)

        self.superresolution.update(
            dict(
                ada_CA_f4_1=GS_Adaptive_Read_Write_CA_adaptive(
                    self.embed_dim,
                    vit_decoder.embed_dim,
                    vit_heads=vit_decoder.num_heads,
                    mlp_ratio=vit_decoder.mlp_ratio,
                    depth=vit_decoder.depth // 6,
                    f=6,  # anchored on the scale
                    heads=8),  # write
                ada_CA_f4_2=GS_Adaptive_Read_Write_CA_adaptive(
                    self.embed_dim,
                    vit_decoder.embed_dim,
                    vit_heads=vit_decoder.num_heads,
                    mlp_ratio=vit_decoder.mlp_ratio,
                    depth=vit_decoder.depth // 6,
                    f=6,
                    heads=8),  # write
            ))


class splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_superviseBaseEMD_jointRender(
        splatting_dit_v4_PI_V1_trilatent_sphere_S_residual):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=181,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         pcd_reso, **kwargs)
        # 1. avoid shrinking scale -> no more tiny points
        # 2. joint gaussians rendering -> better performance
        # 3. emd on base scale only (faster) -> faster, and base manifold good.

    def forward_gaussians(self, ret_after_decoder, c=None):
        # ! create grid and deform Gaussians accordingly
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        # ! merge gaussians

        ret_after_decoder['gaussians'] = torch.cat([
            ret_after_decoder['latent_after_vit']['gaussians_base'],
            ret_after_decoder['latent_after_vit']['gaussians_upsampled'],
        ],
                                                   dim=1)

        if self.aug_opacity_prob > 0:
            if np.random.rand() < self.aug_opacity_prob:  # uniform sample
                ret_after_decoder['gaussians'][
                    ..., 3:4] = 1  # set opacity to 1, avoid floating points

        # ret_after_decoder['gaussians'][
        #     ..., 3:4] = 1  # set opacity to 1, avoid floating points

        # st()
        # self.vis_gaussian(ret_after_decoder['latent_after_vit']['gaussians_upsampled'], 'augbg-base')
        # self.vis_gaussian(ret_after_decoder['latent_after_vit']['gaussians_upsampled'], 'augbg-f4')
        # st()

        ret_after_decoder['pos'] = ret_after_decoder['latent_after_vit'][
            'gaussians_base'][..., 0:3]  #
        # x = ret_after_decoder['latent_after_vit']

        return ret_after_decoder

    # ! decode
    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        B = latent_from_vit.shape[0]

        # ! base, fixed
        xyz_emb = self.superresolution['xyz_pos_embed'](
            self.g_cube).repeat_interleave(B, 0)  # B 14 H W
        # st()
        gaussian_base_pre_activate, gaussian_base_feat = self.superresolution[
            'conv_sr'](xyz_emb, latent_from_vit, return_x=True)  # B 14 H W

        gaussians_base = self._get_base_pcd(gaussian_base_pre_activate)
        gaussians_upsampled_preactivate, gaussians_upsampled_pos = self.superresolution[
            'ada_CA_f4_1'](gaussians_base,
                           gaussian_base_pre_activate,
                           gaussian_base_feat,
                           self.superresolution['xyz_pos_embed'],
                           shrink_scale=False)  # ! avoid manual scale tuning.

        gaussians_upsampled = self._gaussian_pred_activations(
            pos=gaussians_upsampled_pos, x=gaussians_upsampled_preactivate)

        latent_after_vit = {
            'gaussians_base': gaussians_base,
            'gaussians_upsampled': gaussians_upsampled,
        }

        ret_dict.update(
            dict(cls_token=cls_token, latent_after_vit=latent_after_vit))

        return ret_dict


# mip support


class splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_addReadCA_mip(
        splatting_dit_v4_PI_V1_trilatent_sphere_S_residual_addReadCA):
    # add joint render also
    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            pcd_reso=181,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         pcd_reso, **kwargs)

    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        print("Computing 3D filter")
        #TODO consider focal length and image width
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]),
                                   device=xyz.device,
                                   dtype=torch.bool)

        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
            # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]

            xyz_to_cam = torch.norm(xyz_cam, dim=1)

            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.2

            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)

            x = x / z * camera.focal_x + camera.image_width / 2.0
            y = y / z * camera.focal_y + camera.image_height / 2.0

            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))

            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(
                torch.logical_and(x >= -0.15 * camera.image_width, x
                                  <= camera.image_width * 1.15),
                torch.logical_and(y >= -0.15 * camera.image_height, y
                                  <= 1.15 * camera.image_height))

            valid = torch.logical_and(valid_depth, in_screen)

            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.focal_x:
                focal_length = camera.focal_x

        distance[~valid_points] = distance[valid_points].max()

        #TODO remove hard coded value
        #TODO box to gaussian transform
        filter_3D = distance / focal_length * (0.2**0.5)
        self.filter_3D = filter_3D[..., None]

    # @property
    def get_scaling_with_3D_filter(self, scales, filter_3D):
        # scales = self.get_scaling

        scales = torch.square(scales) + torch.square(filter_3D)
        scales = torch.sqrt(scales)
        return scales

    # @property
    def get_opacity_with_3D_filter(self, opacity, scales, filter_3D):
        # opacity = self.opacity_activation(self._opacity)
        # # apply 3D filter
        # scales = self.get_scaling

        scales_square = torch.square(scales)

        det1 = scales_square.prod(dim=-1)

        scales_after_square = scales_square + torch.square(filter_3D)
        det2 = scales_after_square.prod(dim=-1)
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]

    def forward_gaussians(self, ret_after_decoder, c=None):
        # ! create grid and deform Gaussians accordingly
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        # ! merge gaussians

        ret_after_decoder['gaussians'] = torch.cat([
            ret_after_decoder['latent_after_vit']['gaussians_base'],
            ret_after_decoder['latent_after_vit']['gaussians_upsampled'],
        ],
                                                   dim=1)

        if self.aug_opacity_prob > 0:
            if np.random.rand() < self.aug_opacity_prob:  # uniform sample
                ret_after_decoder['gaussians'][
                    ..., 3:4] = 1  # set opacity to 1, avoid floating points

        ret_after_decoder['pos'] = ret_after_decoder['latent_after_vit'][
            'gaussians_base'][..., 0:3]  #
        # x = ret_after_decoder['latent_after_vit']

        return ret_after_decoder

    # ! add smooth filter before rendering
    def triplane_decode(self,
                        ret_after_gaussian_forward,
                        c,
                        bg_color=None,
                        **kwargs):
        # ! for compat, should be gaussian_render

        data = c

        gaussians = ret_after_gaussian_forward['gaussians']
        # !
        # st() # get depth from w2c
        with torch.no_grad():
            pos = gaussians[..., :3]
            pos_homo = torch.cat([pos, torch.ones_like(pos[..., 0:1])],
                                 -1)  # B N 4
            # B 10 N 4 1
            gaussian_w2c = data['cam_view'].unsqueeze(2).repeat_interleave(
                pos.shape[1], 2) @ pos_homo.unsqueeze(1).repeat_interleave(
                    data['cam_view'].shape[1], 1).unsqueeze(-1)
            # gaussian_w2c = gaussian_w2c.squeeze(-1) # B chunk_size N 4

            # TODO, add in_screen valid check
            distance = gaussian_w2c[..., 2, 0].clamp(min=0.2)

            distance = distance.min(1)[0]  # shape: B N

            focal_length = data['orig_pose'][
                0, 0,
                16] * 256  # pixels=1 here. hard-coded 256 pixels in rendering.
            filter_3D = (distance / focal_length * (0.2**0.5)).unsqueeze(-1)

        scale = gaussians[..., 4:7]
        opacity = gaussians[..., 3:4]

        opacity_with_3D_filter = self.get_opacity_with_3D_filter(
            opacity, scale, filter_3D)
        scale_with_3D_filter = self.get_scaling_with_3D_filter(
            scale, filter_3D)
        smoothed_gaussians = torch.cat([
            pos, opacity_with_3D_filter, scale_with_3D_filter, gaussians[...,
                                                                         7:]
        ], -1)

        results = self.gs.render(
            # ret_after_gaussian_forward['gaussians'],  #  type: ignore
            smoothed_gaussians,
            data['cam_view'],
            data['cam_view_proj'],
            data['cam_pos'],
            tanfov=data['tanfov'],
            bg_color=bg_color,
        )
        #  bg_color=bg_color)
        # pred_images = results['image'] # [B, V, C, output_size, output_size]
        # pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]

        results['image_raw'] = results[
            'image'] * 2 - 1  # [0,1] -> [-1,1], match tradition
        results['image_depth'] = results['depth']
        results['image_mask'] = results['alpha']

        # ! vis
        # B, V = results['image_raw'].shape[:2]
        # for b in range(B):
        #     torchvision.utils.save_image(results['image_raw'][b],
        #                                 #  f'tmp/vis-{b}.jpg',
        #                                 #  f'tmp/dust3r/add3dsupp-{b}.jpg',
        #                                  f'tmp/lambda50/add3dsupp-{b}.jpg',
        #                                  normalize=True,
        #                                  value_range=(-1, 1))
        # st()

        return results


# srt tokenizer on gs
class srt_tokenizer_gs(
        # RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder_S
        splatting_dit_v4_PI_V1_trilatent_sphere_S_60):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)
        self.D_roll_out_input = False

        self.superresolution.update(
            dict(
                quant_conv=nn.Conv2d(
                    2 * self.ldm_z_channels,
                    2 * self.ldm_embed_dim,
                    kernel_size=1,  # just MLP
                    groups=1),
                ldm_upsample=PatchEmbedTriplane(
                    self.vae_p * self.token_size,
                    self.vae_p,
                    3 * self.ldm_embed_dim,  # B 3 L C
                    vit_decoder.embed_dim,
                    bias=True,
                    plane_n=self.plane_n,
                )))

    def vae_encode(self, h):
        # * smooth convolution before triplane
        B, C, H, W = h.shape  # C=24 here
        moments = self.superresolution['quant_conv'](
            h)  # Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), groups=3)
        moments = rearrange(moments,
                            'B C (N H) W -> B C N H W',
                            N=self.plane_n,
                            B=B,
                            H=W,
                            W=W)

        moments = moments.flatten(-2)  # B C 3 L

        posterior = DiagonalGaussianDistribution(
            moments, soft_clamp=self.reparameterization_soft_clamp)
        return posterior


class pcd_structured_latent_space(
        # RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder_S
        splatting_dit_v4_PI_V1_trilatent_sphere_S_60):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)
        self.D_roll_out_input = False
        # del self.g_cube  # directly predict xyz

        self.g_cube = create_sphere(self.scene_range[1] * 2 / 3, 28).to(
            dist_util.dev())[:768].unsqueeze(0)  # just a voxel

        self.superresolution.update(
            dict(
                conv_sr=gaussian_prediction(query_dim=vit_decoder.embed_dim),
                quant_conv=Mlp(in_features=2 * self.ldm_z_channels,
                               out_features=2 * self.ldm_embed_dim,
                               act_layer=approx_gelu,
                               drop=0),
                post_quant_conv=Mlp(in_features=self.ldm_z_channels,
                                    out_features=vit_decoder.embed_dim,
                                    act_layer=approx_gelu,
                                    drop=0),
                ldm_upsample=nn.Identity(),
                xyz_pos_embed=nn.Identity(),
            ))

        assert not self.cls_token

    def init_weights(self):
        # ! init (learnable) PE for DiT
        self.vit_decoder.pos_embed = nn.Parameter(
            torch.zeros(1, self.vit_decoder.embed_dim,
                        self.vit_decoder.embed_dim),
            requires_grad=True)  # token_size = embed_size by default.
        trunc_normal_(self.vit_decoder.pos_embed, std=.02)

    def vae_encode(self, h):
        # * smooth convolution before triplane
        # B, L, C = h.shape #
        moments = self.superresolution['quant_conv'](
            h)  # Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), groups=3)
        moments = rearrange(moments, 'B L C -> B C L')  # for vae code compat

        posterior = DiagonalGaussianDistribution(
            moments, soft_clamp=self.reparameterization_soft_clamp)

        return posterior

    def vae_reparameterization(self, latent, sample_posterior):
        # latent: B 24 32 32

        # assert self.vae_p > 1

        # ! do VAE here
        posterior = self.vae_encode(latent['h'])  # B self.ldm_z_channels 3 L

        if sample_posterior:
            kl_latent = posterior.sample()
        else:
            kl_latent = posterior.mode()  # B C 3 L

        # ! reshape for ViT decoder

        # st()
        ret_dict = dict(
            latent_normalized=rearrange(kl_latent, 'B C L -> B L C'),
            posterior=posterior,
            query_pcd_xyz=latent['query_pcd_xyz'],
        )

        return ret_dict

    def vit_decode_backbone(self, latent, img_size):
        # assert x.ndim == 3  # N L C
        if isinstance(latent, dict):
            latent = latent['latent_normalized']  # B, C*3, H, W

        latent = self.superresolution['post_quant_conv'](
            latent)  # to later dit embed dim

        # ! directly feed to vit_decoder
        return {
            'latent': latent,
            'latent_from_vit': self.forward_vit_decoder(latent, img_size)
        }  # pred_vit_latent

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):
        # from ViT_decode_backbone()

        vae_latent, latent_from_vit = latent_from_vit[
            'latent'], latent_from_vit['latent_from_vit']

        gaussian_base_pre_activate = self.superresolution['conv_sr'](
            latent_from_vit)  # B 14 H W
        # gaussians_base = self._get_base_pcd(gaussian_base_pre_activate)

        # latent_after_vit = {
        #     'gaussian_base_pre_activate': gaussian_base_pre_activate,
        #     # 'gaussians_upsampled': gaussians_upsampled,
        # }

        # ret_dict.update(dict(latent_after_vit=latent_after_vit)) #
        ret_dict.update(
            dict(gaussian_base_pre_activate=gaussian_base_pre_activate))  #

        return ret_dict

    def forward_gaussians(self, ret_after_decoder, c=None):
        # ! create grid and deform Gaussians accordingly

        x = ret_after_decoder['gaussian_base_pre_activate']
        B, N, C = x.shape  # B C D H W, 14-dim voxel features
        assert C == 14

        offsets = self.offset_act(x[..., 0:3])  # [B, N, 3] # ! learned offsets
        # pos = offsets + self.g_cube.repeat_interleave(B, dim=0) # much better performance
        pos = offsets  # ! init bug.

        opacity = self.opacity_act(x[..., 3:4])

        scale = self.scale_act(x[..., 4:7])

        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs],
                              dim=-1)  # [B, N, 14]

        # ret_after_decoder['gaussians'] = torch.cat([
        #     ret_after_decoder['latent_after_vit']['gaussians_base'],
        #     # ret_after_decoder['latent_after_vit']['gaussians_upsampled'],
        # ],
        #                                            dim=1)

        # ret_after_decoder['pos'] = ret_after_decoder['latent_after_vit'][
        #     'gaussians_base'][..., 0:3]  #

        ret_after_decoder.update({'gaussians': gaussians, 'pos': pos})
        # ret_after_decoder.update({'gaussians': gaussians})

        # ! render at L:8414
        return ret_after_decoder


class pcd_structured_latent_space_lion(
        # RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder_S
        pcd_structured_latent_space):
    # 1. add skip_weight for vae prediction
    # 2. force vae pred sigma = 0 during init.

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        # https://github.com/nv-tlabs/LION/blob/155a22f5c9f5ff4b2d15aed4e86fbdd8b4bf7ba1/script/train_vae.sh#L20C1-L20C12
        self.skip_weight = torch.tensor(0.01).to(dist_util.dev())
        self.sigma_offset = torch.tensor(-6).to(dist_util.dev())

    def vae_reparameterization(self, latent, sample_posterior):
        # latent: B 24 32 32

        # assert self.vae_p > 1

        # ! do VAE here
        posterior = self.vae_encode(latent)  # B self.ldm_z_channels 3 L

        assert sample_posterior
        if sample_posterior:
            # torch.manual_seed(0)
            # np.random.seed(0)
            kl_latent = posterior.sample()
        else:
            kl_latent = posterior.mode()  # B C 3 L

        # ! reshape for ViT decoder

        # st()
        ret_dict = dict(
            latent_normalized=rearrange(kl_latent, 'B C L -> B L C'),
            posterior=posterior,
            query_pcd_xyz=latent['query_pcd_xyz'],
        )

        return ret_dict

    def vae_encode(self, h):
        # * smooth convolution before triplane
        # B, L, C = h.shape #
        h, query_pcd_xyz = h['h'], h['query_pcd_xyz']
        moments = self.superresolution['quant_conv'](
            h)  # Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), groups=3)

        mu, logvar = torch.chunk(moments, 2, dim=2)  # B L C

        logvar = logvar + self.sigma_offset  # force std=0 when init
        mu = torch.cat(
            [mu[..., :3] * self.skip_weight + query_pcd_xyz, mu[..., 3:]],
            dim=2)

        moments = torch.cat([mu, logvar], 2)
        moments = rearrange(moments,
                            'B L C -> B C L')  # for sd vae code compat

        posterior = DiagonalGaussianDistribution(
            moments, soft_clamp=self.reparameterization_soft_clamp)

        return posterior

    def forward_gaussians(self, ret_after_decoder, c=None):
        # ! create grid and deform Gaussians accordingly

        x = ret_after_decoder['gaussian_base_pre_activate']
        B, N, C = x.shape  # B C D H W, 14-dim voxel features
        assert C == 14

        offsets = self.offset_act(x[..., 0:3])  # ! model prediction
        vae_sampled_xyz = ret_after_decoder['latent_normalized'][..., :
                                                                 3]  # B L C
        pos = offsets * self.skip_weight + vae_sampled_xyz  # ! reasonable init

        opacity = self.opacity_act(x[..., 3:4])

        scale = self.scale_act(x[..., 4:7])

        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs],
                              dim=-1)  # [B, N, 14]

        # self.vis_gaussian(gaussians, 'anchor') # check why limited performance.
        # st()

        ret_after_decoder.update({'gaussians': gaussians, 'pos': pos})
        # ret_after_decoder.update({'gaussians': gaussians})

        # ! render at L:8414
        return ret_after_decoder


class pcd_structured_latent_space_lion_learnoffset(
        pcd_structured_latent_space_lion):
    # 1. weaker skip_weight for vae prediction
    # 2. only force point vae pred sigma = 0 during init.
    # 3. smaller KL loss for first 3 dim

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        self.skip_weight = torch.tensor(0.05).to(dist_util.dev())  # more DoF
        self.sigma_offset = torch.tensor(-6).to(dist_util.dev())

    def vae_encode(self, h):
        # * smooth convolution before triplane
        # B, L, C = h.shape #
        h, query_pcd_xyz = h['h'], h['query_pcd_xyz']
        moments = self.superresolution['quant_conv'](
            h)  # Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), groups=3)

        mu, logvar = torch.chunk(moments, 2, dim=2)  # B L C

        mu = torch.cat(
            [mu[..., :3] * self.skip_weight + query_pcd_xyz, mu[..., 3:]],
            dim=2)
        logvar = torch.cat(
            [logvar[..., :3] + self.sigma_offset, logvar[..., 3:]], dim=2)

        moments = torch.cat([mu, logvar], 2)
        moments = rearrange(moments,
                            'B L C -> B C L')  # for sd vae code compat

        posterior = DiagonalGaussianDistribution(
            moments, soft_clamp=self.reparameterization_soft_clamp)

        return posterior


class pcd_structured_latent_space_lion_learnoffset_surfel(
        pcd_structured_latent_space_lion_learnoffset):
    # for 2dgs

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        self.superresolution.update(
            dict(conv_sr=surfel_prediction(query_dim=vit_decoder.embed_dim), ))

        # self.rot_act = lambda x: F.normalize(x, dim=-1) # as fixed in lgm

    def forward_gaussians(self, ret_after_decoder, c=None):
        # ! create grid and deform Gaussians accordingly

        x = ret_after_decoder['gaussian_base_pre_activate']
        B, N, C = x.shape  # B C D H W, 14-dim voxel features
        assert C == 13  # 2dgs

        offsets = self.offset_act(x[..., 0:3])  # ! model prediction
        vae_sampled_xyz = ret_after_decoder['latent_normalized'][..., :
                                                                 3]  # B L C
        pos = offsets * self.skip_weight + vae_sampled_xyz  # ! reasonable init

        opacity = self.opacity_act(x[..., 3:4])

        scale = self.scale_act(x[..., 4:6])

        rotation = self.rot_act(x[..., 6:10])
        rgbs = self.rgb_act(x[..., 10:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs],
                              dim=-1)  # [B, N, 14]

        # self.vis_gaussian(gaussians, 'anchor') # check why limited performance.
        # st()

        ret_after_decoder.update({'gaussians': gaussians, 'pos': pos})
        # ret_after_decoder.update({'gaussians': gaussians})

        # ! render at L:8414
        return ret_after_decoder


class pcd_structured_latent_space_lion_learnoffset_surfel_sr(
        pcd_structured_latent_space_lion_learnoffset_surfel):
    # for 2dgs

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        self.rot_act = lambda x: F.normalize(x, dim=-1)  # as fixed in lgm

        self.superresolution.update(  # f=14 here
            dict(
                ada_CA_f4_1=GS_Adaptive_Read_Write_CA_adaptive_2dgs(
                    self.embed_dim,
                    vit_decoder.embed_dim,
                    vit_heads=vit_decoder.num_heads,
                    mlp_ratio=vit_decoder.mlp_ratio,
                    depth=vit_decoder.depth // 6,
                    # f=16,  # 4 x 4? indeed shall be 64 directly.
                    # f=24,  # 4 x 4? indeed shall be 64 directly.
                    f=32,  # 4 x 4? indeed shall be 64 directly.
                    heads=8),  # write
            ))

        # improve SR utiliaztion ratio. currently clutered around anchor points. spread out is better?
        # self.skip_weight = torch.tensor(1.0).to(dist_util.dev())
        self.skip_weight = torch.tensor(0.1).to(dist_util.dev())
        # self.skip_weight = torch.tensor(0.075).to(dist_util.dev())

    def _gaussian_pred_activations(self, pos, x):
        # if pos is None:
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:6])
        rotation = self.rot_act(x[..., 6:10])
        rgbs = self.rgb_act(x[..., 10:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs],
                              dim=-1)  # [B, N, 14]

        return gaussians.float()

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):
        # from ViT_decode_backbone()

        latent_from_vit = latent_from_vit['latent_from_vit']
        vae_sampled_xyz = ret_dict['latent_normalized'][..., :3]  # B L C

        gaussians_upsampled = self.superresolution['ada_CA_f4_1'](
            latent_from_vit,
            vae_sampled_xyz.to(latent_from_vit.dtype),
            skip_weight=self.skip_weight,
            gs_pred_fn=self.superresolution['conv_sr'],
            gs_act_fn=self._gaussian_pred_activations,
            offset_act=self.offset_act)

        ret_dict.update(dict(gaussians_upsampled=gaussians_upsampled))  #

        return ret_dict

    def forward_gaussians(self, ret_after_decoder, c=None):
        # return: Gaussians: [B, dim_t]

        # ! no need to merge gaussians
        ret_after_decoder['gaussians'] = ret_after_decoder[
            'gaussians_upsampled']
        ret_after_decoder['pos'] = ret_after_decoder['gaussians'][..., 0:3]  #

        # self.vis_gaussian(ret_after_decoder['gaussians'], 'f8-upsampled')
        # pcu.save_mesh_v(f'{Path(logger.get_dir())}/anchor.ply',ret_after_decoder['query_pcd_xyz'][0].float().detach().cpu().numpy())
        # st()

        return ret_after_decoder

    def vis_gaussian(self, gaussians, file_name_base):
        # gaussians = ret_after_decoder['gaussians']
        # gaussians = ret_after_decoder['latent_after_vit']['gaussians_base']
        B = gaussians.shape[0]
        pos, opacity, scale, rotation, rgbs = gaussians[..., 0:3], gaussians[
            ..., 3:4], gaussians[..., 4:6], gaussians[...,
                                                      6:10], gaussians[...,
                                                                       10:13]
        file_path = Path(logger.get_dir())

        for b in range(B):
            file_name = f'{file_name_base}-{b}'

            np.save(file_path / f'{file_name}_opacity.npy',
                    opacity[b].float().detach().cpu().numpy())
            np.save(file_path / f'{file_name}_scale.npy',
                    scale[b].float().detach().cpu().numpy())
            np.save(file_path / f'{file_name}_rotation.npy',
                    rotation[b].float().detach().cpu().numpy())

            pcu.save_mesh_vc(str(file_path / f'{file_name}.ply'),
                             pos[b].float().detach().cpu().numpy(),
                             rgbs[b].float().detach().cpu().numpy())

    # def triplane_d


class pcd_structured_latent_space_lion_learnoffset_surfel_sr_noptVAE(
        pcd_structured_latent_space_lion_learnoffset_surfel_sr):
    # for 2dgs

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)
        # self.skip_weight = torch.tensor(0.1).to(dist_util.dev()) # still large deformation allowed
        # self.skip_weight = torch.tensor(0.075).to(dist_util.dev()) # still large deformation allowed

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):
        # from ViT_decode_backbone()

        latent_from_vit = latent_from_vit['latent_from_vit']
        vae_sampled_xyz = ret_dict['query_pcd_xyz'].to(
            latent_from_vit.dtype)  # ! directly use fps pcd as "anchor points"

        gaussians_upsampled = self.superresolution['ada_CA_f4_1'](
            latent_from_vit,
            vae_sampled_xyz,
            skip_weight=self.skip_weight,
            gs_pred_fn=self.superresolution['conv_sr'],
            gs_act_fn=self._gaussian_pred_activations,
            offset_act=self.offset_act)

        ret_dict.update(dict(gaussians_upsampled=gaussians_upsampled))  #

        return ret_dict

    def vae_encode(self, h):
        # * smooth convolution before triplane
        # B, L, C = h.shape #
        h, query_pcd_xyz = h['h'], h['query_pcd_xyz']
        moments = self.superresolution['quant_conv'](
            h)  # Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), groups=3)

        moments = rearrange(moments,
                            'B L C -> B C L')  # for sd vae code compat

        posterior = DiagonalGaussianDistribution(
            moments, soft_clamp=self.reparameterization_soft_clamp)

        return posterior


class pcd_structured_latent_space_lion_learnoffset_surfel_sr_noptVAE_debugscale(
        pcd_structured_latent_space_lion_learnoffset_surfel_sr_noptVAE):
    # for 2dgs

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        # self.scale_act = lambda x: F.softplus(x) * 0.1
        # self.scene_extent = self.rendering_kwargs['sampler_bbox_max'] * 0.02
        self.scene_extent = self.rendering_kwargs['sampler_bbox_max'] * 0.01
        # self.scale_act = lambda x: F.tanh(x) * self.scene_extent + self.scene_extent # bounded
        # self.scale_act = lambda x: F.softplus(x).clamp(2) * self.scene_extent
        # self.scale_act = lambda x: (F.tanh(x)+1) / 2 * self.rendering_kwargs['sampler_bbox_max'] * 0.01 # bound the scaling withint 0.01 * scene_extent
        # self.scale_act = lambda x: F.softplus(x) * 0.01

        scaling_factor = (self.scene_extent /
                          F.softplus(torch.tensor(0.0))).to(dist_util.dev())
        self.scale_act = lambda x: F.softplus(
            x
        ) * scaling_factor  # make sure F.softplus(0) is the average scale size

        # self.scale_act = lambda x: F.softplus(x) * 0.5
        # self.scale_act = lambda x: torch.exp(x) * 0.1 # splatter image, # https://github.com/szymanowiczs/splatter-image/blob/78a6ad098e0cdc40c59c8ec98ca4fa439870fabd/scene/gaussian_predictor.py#L505

        self.superresolution.update(  # f=14 here
            dict(
                ada_CA_f4_1=GS_Adaptive_Read_Write_CA_adaptive_2dgs(
                    self.embed_dim,
                    vit_decoder.embed_dim,
                    vit_heads=vit_decoder.num_heads,
                    mlp_ratio=vit_decoder.mlp_ratio,
                    depth=vit_decoder.depth // 6,
                    f=8,  # 4 x 4? indeed shall be 64 directly.
                    heads=8),  # write
            ))


class pcd_structured_latent_space_lion_learnoffset_surfel_novaePT(
        pcd_structured_latent_space_lion_learnoffset_surfel_sr_noptVAE_debugscale
):
    # for 2dgs

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)
        del self.superresolution['ada_CA_f4_1']  # direct gs prediction here

        # N = 768 # hyp param, overfitting now

        # self.scale_expected_threshold = (1 / (N/2)) ** 0.5 * self.rendering_kwargs['sampler_bbox_max']

        # self.skip_weight = torch.tensor(1).to(dist_util.dev()) # still large deformation allowed

        # self.offset_act = lambda x: torch.tanh(x) * (self.scene_range[1]) * 0.5  #

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):
        # from ViT_decode_backbone()

        latent_from_vit = latent_from_vit['latent_from_vit']

        # gaussians_upsampled = self.superresolution[
        #     'ada_CA_f4_1'](latent_from_vit, vae_sampled_xyz, skip_weight=self.skip_weight, gs_pred_fn=self.superresolution['conv_sr'], gs_act_fn=self._gaussian_pred_activations, offset_act=self.offset_act)

        gaussian_base_pre_activate = self.superresolution['conv_sr'](
            latent_from_vit)  # B 14 H W
        ret_dict.update(
            dict(gaussian_base_pre_activate=gaussian_base_pre_activate))  #

        # ret_dict.update(dict(gaussians_upsampled=gaussians_upsampled)) #

        return ret_dict

    def _get_base_gaussians(self, ret_after_decoder, c=None):
        x = ret_after_decoder['gaussian_base_pre_activate']
        B, N, C = x.shape  # B C D H W, 14-dim voxel features
        assert C == 13  # 2dgs

        offsets = self.offset_act(x[..., 0:3])  # ! model prediction
        # st()
        # vae_sampled_xyz = ret_after_decoder['latent_normalized'][..., :3] # B L C

        vae_sampled_xyz = ret_after_decoder['query_pcd_xyz'].to(
            x.dtype)  # ! directly use fps pcd as "anchor points"

        pos = offsets * self.skip_weight + vae_sampled_xyz  # ! reasonable init

        opacity = self.opacity_act(x[..., 3:4])

        scale = self.scale_act(x[..., 4:6])

        rotation = self.rot_act(x[..., 6:10])
        rgbs = self.rgb_act(x[..., 10:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs],
                              dim=-1)  # [B, N, 14]

        return gaussians

    def forward_gaussians(self, ret_after_decoder, c=None):

        gaussians = self._get_base_gaussians(ret_after_decoder, c)

        ret_after_decoder.update({
            'gaussians': gaussians,
            'pos': gaussians[..., :3]
        })

        # ! render at L:8414 triplane_decode()
        return ret_after_decoder


# ! SR version of novae_pt


class pcd_structured_latent_space_lion_learnoffset_surfel_novaePT_sr(
        pcd_structured_latent_space_lion_learnoffset_surfel_novaePT):
    # for 2dgs

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        self.superresolution.update(  # f=14 here
            dict(
                ada_CA_f4_1=GS_Adaptive_Read_Write_CA_adaptive_2dgs(
                    self.embed_dim,
                    vit_decoder.embed_dim,
                    vit_heads=vit_decoder.num_heads,
                    mlp_ratio=vit_decoder.mlp_ratio,
                    # depth=vit_decoder.depth // 6,
                    depth=vit_decoder.depth // 6 if vit_decoder.depth==12 else 2,
                    # f=16,  # 
                    f=8,  # 
                    heads=8),  # write
            ))

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict, return_upsampled_residual=False):
        # from ViT_decode_backbone()

        # latent_from_vit = latent_from_vit['latent_from_vit']
        # vae_sampled_xyz = ret_dict['query_pcd_xyz'].to(latent_from_vit.dtype) # ! directly use fps pcd as "anchor points"
        gaussian_base_pre_activate = self.superresolution['conv_sr'](
            latent_from_vit['latent_from_vit'])  # B 14 H W

        gaussians_base = self._get_base_gaussians(
            {
                # 'latent_from_vit': latent_from_vit,  # latent (vae latent), latent_from_vit (dit)
                # 'ret_dict': ret_dict,
                **ret_dict,
                'gaussian_base_pre_activate':
                gaussian_base_pre_activate,
            }, )

        gaussians_upsampled, (gaussian_upsampled_residual_pre_activate, upsampled_global_local_query_emb) = self.superresolution['ada_CA_f4_1'](
            latent_from_vit['latent_from_vit'],
            gaussians_base,
            skip_weight=self.skip_weight,
            gs_pred_fn=self.superresolution['conv_sr'],
            gs_act_fn=self._gaussian_pred_activations,
            offset_act=self.offset_act,
            gaussian_base_pre_activate=gaussian_base_pre_activate)

        ret_dict.update({
            'gaussians_upsampled': gaussians_upsampled,
            'gaussians_base': gaussians_base
        })  #

        if return_upsampled_residual:
            return ret_dict, (gaussian_upsampled_residual_pre_activate, upsampled_global_local_query_emb)
        else:
            return ret_dict

    def forward_gaussians(self, ret_after_decoder, c=None):

        # ! currently, only using upsampled gaussians for training.

        # if True:
        if False:
            ret_after_decoder['gaussians'] = torch.cat([
                ret_after_decoder['gaussians_base'],
                ret_after_decoder['gaussians_upsampled'],
            ],
                                                       dim=1)
        else:  # only adopt SR
            # ! random drop out requires
            ret_after_decoder['gaussians'] = ret_after_decoder[
                'gaussians_upsampled']
            # ret_after_decoder['gaussians'] = ret_after_decoder['gaussians_base']
            pass  # directly use base. vis first.

        ret_after_decoder.update({
            'gaussians': ret_after_decoder['gaussians'],
            'pos': ret_after_decoder['gaussians'][..., :3],
            'gaussians_base_opa': ret_after_decoder['gaussians_base'][..., 3:4]
        })

        # st()
        # self.vis_gaussian(ret_after_decoder['gaussians'], 'sr-8')
        # self.vis_gaussian(ret_after_decoder['gaussians_base'], 'sr-8-base')
        # pcu.save_mesh_v(f'{Path(logger.get_dir())}/anchor-fps-8.ply',ret_after_decoder['query_pcd_xyz'][0].float().detach().cpu().numpy())
        # st()

        # ! render at L:8414 triplane_decode()
        return ret_after_decoder

# cascade rendering 

class pcd_structured_latent_space_lion_learnoffset_surfel_novaePT_sr_cascade(
        pcd_structured_latent_space_lion_learnoffset_surfel_novaePT_sr):
    # for 2dgs

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)
                        
        self.output_size = {
            'gaussians_base': 128,
            'gaussians_upsampled': 256,
            # 'gaussians_upsampled': 320,
        }
        self.rand_base_render = False


    def triplane_decode(self,
                        ret_after_gaussian_forward,
                        c,
                        bg_color=None, 
                        render_all_scale=False,
                        **kwargs):
        # ! render multi-res img with different gaussians

        def render_gs(gaussians, c_data, output_size):

            results = self.gs.render(
                gaussians,  #  type: ignore
                c_data['cam_view'],
                c_data['cam_view_proj'],
                c_data['cam_pos'],
                tanfov=c_data['tanfov'],
                bg_color=bg_color,
                output_size=output_size,
            )

            results['image_raw'] = results[
                'image'] * 2 - 1  # [0,1] -> [-1,1], match tradition
            results['image_depth'] = results['depth']
            results['image_mask'] = results['alpha']

            return results

        cascade_splatting_results = {}

        # for gaussians_key in ('gaussians_base', 'gaussians_upsampled'):
        all_keys_to_render = list(self.output_size.keys())

    
        if self.rand_base_render and not render_all_scale:
            keys_to_render = [random.choice(all_keys_to_render[:-1])] + [all_keys_to_render[-1]]
        else:
            keys_to_render = all_keys_to_render

        for gaussians_key in keys_to_render:
            cascade_splatting_results[gaussians_key] = render_gs(ret_after_gaussian_forward[gaussians_key], c, self.output_size[gaussians_key])

        return cascade_splatting_results


class pcd_structured_latent_space_lion_learnoffset_surfel_novaePT_sr_cascade_x8x4x4(
        pcd_structured_latent_space_lion_learnoffset_surfel_novaePT_sr_cascade):
    # for 2dgs

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)
                        
        # st()
        self.output_size = {
            'gaussians_base': 128, # 64x2
            'gaussians_upsampled': 256, # 64x4
            'gaussians_upsampled_2': 320,
            'gaussians_upsampled_3': 384,
        }
        self.rand_base_render = True

        # further x8 up-sampling.
        self.superresolution.update(  
            dict(
                ada_CA_f4_2=GS_Adaptive_Read_Write_CA_adaptive_2dgs(
                    self.embed_dim,
                    vit_decoder.embed_dim,
                    vit_heads=vit_decoder.num_heads,
                    mlp_ratio=vit_decoder.mlp_ratio,
                    # depth=vit_decoder.depth // 6,
                    depth=1,
                    f=4,  # 
                    heads=8, 
                    no_flash_op=True,  # fails when bs>1
                    cross_attention=False),  # write
                ada_CA_f4_3=GS_Adaptive_Read_Write_CA_adaptive_2dgs(
                    self.embed_dim,
                    vit_decoder.embed_dim,
                    vit_heads=vit_decoder.num_heads,
                    mlp_ratio=vit_decoder.mlp_ratio,
                    # depth=vit_decoder.depth // 6,
                    depth=1,
                    f=3,  # 
                    heads=8, 
                    no_flash_op=True, 
                    cross_attention=False),  # write
            ),
        )


    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        # further x8 using upper class
        # TODO, merge this into ln3diff open sourced code.
        ret_dict, (gaussian_upsampled_residual_pre_activate, upsampled_global_local_query_emb) = super().vit_decode_postprocess(latent_from_vit, ret_dict, return_upsampled_residual=True)

        gaussians_upsampled_2, (gaussian_upsampled_residual_pre_activate_2, upsampled_global_local_query_emb_2) = self.superresolution['ada_CA_f4_2'](
            upsampled_global_local_query_emb,
            ret_dict['gaussians_upsampled'],
            skip_weight=self.skip_weight,
            gs_pred_fn=self.superresolution['conv_sr'],
            gs_act_fn=self._gaussian_pred_activations,
            offset_act=self.offset_act,
            gaussian_base_pre_activate=gaussian_upsampled_residual_pre_activate)


        gaussians_upsampled_3, _ = self.superresolution['ada_CA_f4_3'](
            upsampled_global_local_query_emb_2,
            gaussians_upsampled_2,
            skip_weight=self.skip_weight,
            gs_pred_fn=self.superresolution['conv_sr'],
            gs_act_fn=self._gaussian_pred_activations,
            offset_act=self.offset_act,
            gaussian_base_pre_activate=gaussian_upsampled_residual_pre_activate_2)


        ret_dict.update({
            'gaussians_upsampled_2': gaussians_upsampled_2,
            'gaussians_upsampled_3': gaussians_upsampled_3,
        }) 

        return ret_dict

class pcd_structured_latent_space_lion_learnoffset_surfel_novaePT_sr_cascade_x8x4x4_512(
        pcd_structured_latent_space_lion_learnoffset_surfel_novaePT_sr_cascade_x8x4x4):
    # for 2dgs

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        self.output_size = {
            'gaussians_base': 128,
            'gaussians_upsampled': 256,
            'gaussians_upsampled_2': 384,
            'gaussians_upsampled_3': 512,
        }
        # all from ViTTriplaneDecomposed
        self.vit_decoder
        self.token_size
                        


class debug_cls(pcd_structured_latent_space_lion_learnoffset_surfel_novaePT_sr_cascade_x8x4x4_512):
    def __init__(self, vit_decoder: VisionTransformer, triplane_decoder: Triplane_fg_bg_plane, cls_token, normalize_feat=True, sr_ratio=2, use_fusion_blk=True, fusion_blk_depth=2, fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout, channel_multiplier=4, **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token, normalize_feat, sr_ratio, use_fusion_blk, fusion_blk_depth, fusion_blk, channel_multiplier, **kwargs)
        self.superresolution
        self.embed_dim
        self.decoder_pred_3d
        self.w_avg
        self.offset_act
        self.gs
        self.skip_weight
        self.rgb_act
        self.rot_act
        self.pos_act
        self.scale_act
        self.opacity_act


    def vit_decode_backbone(self, latent, img_size):
        return super().vit_decode_backbone(latent, img_size)

    def vit_decode(self, latent, img_size, c=None, sample_posterior=True):
        return super().vit_decode(latent, img_size, c, sample_posterior)

    def forward_vit_decoder(self, x, img_size=None):
        return super().forward_vit_decoder(x, img_size)

    # from pcd_structured_latent_space_lion
    def vae_reparameterization(self, latent, sample_posterior):
        return super().vae_reparameterization(latent, sample_posterior)

    # from pcd_structured_latent_space_lion_learnoffset_surfel_sr_noptVAE.vae_encode
    def vae_encode(self, h):
        return super().vae_encode(h)

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):
        return super().vit_decode_postprocess(latent_from_vit, ret_dict)

    def forward_gaussians(self, ret_after_decoder, c=None):
        return super().forward_gaussians(ret_after_decoder, c)
    
    def init_weights(self):
        return super().init_weights()



# merged above class into a single class

class vae_3d(nn.Module):
    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            ldm_z_channels,
            ldm_embed_dim,
            plane_n=1,
            vae_dit_token_size=16,
            **kwargs) -> None:
        super().__init__()

        self.reparameterization_soft_clamp = True  # some instability in training VAE

        # st()
        self.plane_n = plane_n
        self.cls_token = cls_token
        self.vit_decoder = vit_decoder
        self.triplane_decoder = triplane_decoder

        self.patch_size = 14  # TODO, hard coded here
        if isinstance(self.patch_size, tuple):  # dino-v2
            self.patch_size = self.patch_size[0]

        self.img_size = None  # TODO, hard coded

        self.ldm_z_channels = ldm_z_channels
        self.ldm_embed_dim = ldm_embed_dim
        self.vae_p = 4  # resolution = 4 * 16
        self.token_size = vae_dit_token_size  # use dino-v2 dim tradition here
        self.vae_res = self.vae_p * self.token_size

        self.superresolution = nn.ModuleDict({}) # put all the stuffs here
        self.embed_dim = vit_decoder.embed_dim

        # placeholder for compat issue
        self.decoder_pred = None
        self.decoder_pred_3d = None
        self.transformer_3D_blk = None
        self.logvar = None
        self.register_buffer('w_avg', torch.zeros([512]))



    def init_weights(self):
        # ! init (learnable) PE for DiT
        self.vit_decoder.pos_embed = nn.Parameter(
            torch.zeros(1, self.vit_decoder.embed_dim,
                        self.vit_decoder.embed_dim),
            requires_grad=True)  # token_size = embed_size by default.
        trunc_normal_(self.vit_decoder.pos_embed, std=.02)


# the base class
class pcd_structured_latent_space_vae_decoder(vae_3d):
    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token, **kwargs)
        # from splatting_dit_v4_PI_V1_trilatent_sphere
        self.D_roll_out_input = False

        # ! renderer
        self.gs = triplane_decoder  # compat

        self.rendering_kwargs = self.gs.rendering_kwargs
        self.scene_range = [
            self.rendering_kwargs['sampler_bbox_min'],
            self.rendering_kwargs['sampler_bbox_max']
        ]

        # hyper parameters
        self.skip_weight = torch.tensor(0.1).to(dist_util.dev())

        self.offset_act = lambda x: torch.tanh(x) * (self.scene_range[
            1]) * 0.5  # regularize small offsets

        self.vit_decoder.pos_embed = nn.Parameter(
            torch.zeros(1,
                        self.plane_n * (self.token_size**2 + self.cls_token),
                        vit_decoder.embed_dim))
        self.init_weights()  # re-init weights after re-writing token_size

        self.output_size = {
            'gaussians_base': 128,
        }

        # activations
        self.rot_act = lambda x: F.normalize(x, dim=-1)  # as fixed in lgm
        self.scene_extent = self.rendering_kwargs['sampler_bbox_max'] * 0.01
        scaling_factor = (self.scene_extent /
                          F.softplus(torch.tensor(0.0))).to(dist_util.dev())
        self.scale_act = lambda x: F.softplus(
            x
        ) * scaling_factor  # make sure F.softplus(0) is the average scale size
        self.rgb_act = lambda x: 0.5 * torch.tanh(
            x) + 0.5  # NOTE: may use sigmoid if train again
        self.pos_act = lambda x: x.clamp(-0.45, 0.45)
        self.opacity_act = lambda x: torch.sigmoid(x)


        self.superresolution.update(
            dict(
                conv_sr=surfel_prediction(query_dim=vit_decoder.embed_dim),
                quant_conv=Mlp(in_features=2 * self.ldm_z_channels,
                               out_features=2 * self.ldm_embed_dim,
                               act_layer=approx_gelu,
                               drop=0),
                post_quant_conv=Mlp(in_features=self.ldm_z_channels,
                                    out_features=vit_decoder.embed_dim,
                                    act_layer=approx_gelu,
                                    drop=0),
                ldm_upsample=nn.Identity(),
                xyz_pos_embed=nn.Identity(),
            ))

        # for gs prediction
        self.superresolution.update(  # f=14 here
            dict(
                ada_CA_f4_1=GS_Adaptive_Read_Write_CA_adaptive_2dgs(
                    self.embed_dim,
                    vit_decoder.embed_dim,
                    vit_heads=vit_decoder.num_heads,
                    mlp_ratio=vit_decoder.mlp_ratio,
                    # depth=vit_decoder.depth // 6,
                    depth=vit_decoder.depth // 6 if vit_decoder.depth==12 else 2,
                    # f=16,  # 
                    f=8,  # 
                    heads=8),  # write
            ))


    def vae_reparameterization(self, latent, sample_posterior):
        # latent: B 24 32 32

        # assert self.vae_p > 1

        # ! do VAE here
        posterior = self.vae_encode(latent)  # B self.ldm_z_channels 3 L

        assert sample_posterior
        if sample_posterior:
            # torch.manual_seed(0)
            # np.random.seed(0)
            kl_latent = posterior.sample()
        else:
            kl_latent = posterior.mode()  # B C 3 L

        ret_dict = dict(
            latent_normalized=rearrange(kl_latent, 'B C L -> B L C'),
            posterior=posterior,
            query_pcd_xyz=latent['query_pcd_xyz'],
        )

        return ret_dict

    # from pcd_structured_latent_space_lion_learnoffset_surfel_sr_noptVAE.vae_encode
    def vae_encode(self, h):
        # * smooth convolution before triplane
        # B, L, C = h.shape #
        h, query_pcd_xyz = h['h'], h['query_pcd_xyz']
        moments = self.superresolution['quant_conv'](
            h)  # Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), groups=3)

        moments = rearrange(moments,
                            'B L C -> B C L')  # for sd vae code compat

        posterior = DiagonalGaussianDistribution(
            moments, soft_clamp=self.reparameterization_soft_clamp)

        return posterior

    # from pcd_structured_latent_space_lion_learnoffset_surfel_novaePT._get_base_gaussians
    def _get_base_gaussians(self, ret_after_decoder, c=None):
        x = ret_after_decoder['gaussian_base_pre_activate']
        B, N, C = x.shape  # B C D H W, 14-dim voxel features
        assert C == 13  # 2dgs

        offsets = self.offset_act(x[..., 0:3])  # ! model prediction
        # st()
        # vae_sampled_xyz = ret_after_decoder['latent_normalized'][..., :3] # B L C

        vae_sampled_xyz = ret_after_decoder['query_pcd_xyz'].to(
            x.dtype)  # ! directly use fps pcd as "anchor points"

        pos = offsets * self.skip_weight + vae_sampled_xyz  # ! reasonable init

        opacity = self.opacity_act(x[..., 3:4])

        scale = self.scale_act(x[..., 4:6])

        rotation = self.rot_act(x[..., 6:10])
        rgbs = self.rgb_act(x[..., 10:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs],
                              dim=-1)  # [B, N, 14]

        return gaussians

    # from pcd_structured_latent_space
    def vit_decode_backbone(self, latent, img_size):
        # assert x.ndim == 3  # N L C
        if isinstance(latent, dict):
            latent = latent['latent_normalized']  # B, C*3, H, W

        latent = self.superresolution['post_quant_conv'](
            latent)  # to later dit embed dim

        # ! directly feed to vit_decoder
        return {
            'latent': latent,
            'latent_from_vit': self.forward_vit_decoder(latent, img_size)
        }  # pred_vit_latent

    # from pcd_structured_latent_space_lion_learnoffset_surfel_sr
    def _gaussian_pred_activations(self, pos, x):
        # if pos is None:
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:6])
        rotation = self.rot_act(x[..., 6:10])
        rgbs = self.rgb_act(x[..., 10:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs],
                              dim=-1)  # [B, N, 14]

        return gaussians.float()

    # from pcd_structured_latent_space_lion_learnoffset_surfel_sr
    def vis_gaussian(self, gaussians, file_name_base):
        # gaussians = ret_after_decoder['gaussians']
        # gaussians = ret_after_decoder['latent_after_vit']['gaussians_base']
        B = gaussians.shape[0]
        pos, opacity, scale, rotation, rgbs = gaussians[..., 0:3], gaussians[
            ..., 3:4], gaussians[..., 4:6], gaussians[...,
                                                      6:10], gaussians[...,
                                                                       10:13]
        file_path = Path(logger.get_dir())

        for b in range(B):
            file_name = f'{file_name_base}-{b}'

            np.save(file_path / f'{file_name}_opacity.npy',
                    opacity[b].float().detach().cpu().numpy())
            np.save(file_path / f'{file_name}_scale.npy',
                    scale[b].float().detach().cpu().numpy())
            np.save(file_path / f'{file_name}_rotation.npy',
                    rotation[b].float().detach().cpu().numpy())

            pcu.save_mesh_vc(str(file_path / f'{file_name}.ply'),
                             pos[b].float().detach().cpu().numpy(),
                             rgbs[b].float().detach().cpu().numpy())

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict, return_upsampled_residual=False):
        # from ViT_decode_backbone()

        # latent_from_vit = latent_from_vit['latent_from_vit']
        # vae_sampled_xyz = ret_dict['query_pcd_xyz'].to(latent_from_vit.dtype) # ! directly use fps pcd as "anchor points"
        gaussian_base_pre_activate = self.superresolution['conv_sr'](
            latent_from_vit['latent_from_vit'])  # B 14 H W

        gaussians_base = self._get_base_gaussians(
            {
                # 'latent_from_vit': latent_from_vit,  # latent (vae latent), latent_from_vit (dit)
                # 'ret_dict': ret_dict,
                **ret_dict,
                'gaussian_base_pre_activate':
                gaussian_base_pre_activate,
            }, )

        gaussians_upsampled, (gaussian_upsampled_residual_pre_activate, upsampled_global_local_query_emb) = self.superresolution['ada_CA_f4_1'](
            latent_from_vit['latent_from_vit'],
            gaussians_base,
            skip_weight=self.skip_weight,
            gs_pred_fn=self.superresolution['conv_sr'],
            gs_act_fn=self._gaussian_pred_activations,
            offset_act=self.offset_act,
            gaussian_base_pre_activate=gaussian_base_pre_activate)

        ret_dict.update({
            'gaussians_upsampled': gaussians_upsampled,
            'gaussians_base': gaussians_base
        })  #

        if return_upsampled_residual:
            return ret_dict, (gaussian_upsampled_residual_pre_activate, upsampled_global_local_query_emb)
        else:
            return ret_dict

    def vit_decode(self, latent, img_size, sample_posterior=True, c=None):

        ret_dict = self.vae_reparameterization(latent, sample_posterior)

        latent = self.vit_decode_backbone(ret_dict, img_size)
        ret_after_decoder = self.vit_decode_postprocess(latent, ret_dict)

        return self.forward_gaussians(ret_after_decoder, c=c)

    # from pcd_structured_latent_space_lion_learnoffset_surfel_novaePT_sr.forward_gaussians
    def forward_gaussians(self, ret_after_decoder, c=None):

        # ! currently, only using upsampled gaussians for training.

        # if True:
        if False:
            ret_after_decoder['gaussians'] = torch.cat([
                ret_after_decoder['gaussians_base'],
                ret_after_decoder['gaussians_upsampled'],
            ],
                                                       dim=1)
        else:  # only adopt SR
            # ! random drop out requires
            ret_after_decoder['gaussians'] = ret_after_decoder[
                'gaussians_upsampled']
            # ret_after_decoder['gaussians'] = ret_after_decoder['gaussians_base']
            pass  # directly use base. vis first.

        ret_after_decoder.update({
            'gaussians': ret_after_decoder['gaussians'],
            'pos': ret_after_decoder['gaussians'][..., :3],
            'gaussians_base_opa': ret_after_decoder['gaussians_base'][..., 3:4]
        })

        # st()
        # self.vis_gaussian(ret_after_decoder['gaussians'], 'sr-8')
        # self.vis_gaussian(ret_after_decoder['gaussians_base'], 'sr-8-base')
        # pcu.save_mesh_v(f'{Path(logger.get_dir())}/anchor-fps-8.ply',ret_after_decoder['query_pcd_xyz'][0].float().detach().cpu().numpy())
        # st()

        # ! render at L:8414 triplane_decode()
        return ret_after_decoder

    def forward_vit_decoder(self, x, img_size=None):
        return self.vit_decoder(x)

    # from pcd_structured_latent_space_lion_learnoffset_surfel_novaePT_sr_cascade.triplane_decode
    def triplane_decode(self,
                        ret_after_gaussian_forward,
                        c,
                        bg_color=None, 
                        render_all_scale=False,
                        **kwargs):
        # ! render multi-res img with different gaussians

        def render_gs(gaussians, c_data, output_size):

            results = self.gs.render(
                gaussians,  #  type: ignore
                c_data['cam_view'],
                c_data['cam_view_proj'],
                c_data['cam_pos'],
                tanfov=c_data['tanfov'],
                bg_color=bg_color,
                output_size=output_size,
            )

            results['image_raw'] = results[
                'image'] * 2 - 1  # [0,1] -> [-1,1], match tradition
            results['image_depth'] = results['depth']
            results['image_mask'] = results['alpha']

            return results

        cascade_splatting_results = {}

        # for gaussians_key in ('gaussians_base', 'gaussians_upsampled'):
        all_keys_to_render = list(self.output_size.keys())

    
        if self.rand_base_render and not render_all_scale:
            keys_to_render = [random.choice(all_keys_to_render[:-1])] + [all_keys_to_render[-1]]
        else:
            keys_to_render = all_keys_to_render

        for gaussians_key in keys_to_render:
            cascade_splatting_results[gaussians_key] = render_gs(ret_after_gaussian_forward[gaussians_key], c, self.output_size[gaussians_key])

        return cascade_splatting_results


class pcd_structured_latent_space_vae_decoder_cascaded(pcd_structured_latent_space_vae_decoder):
    # for 2dgs

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token, **kwargs)

        self.output_size.update(
            {
                'gaussians_upsampled': 256,
                'gaussians_upsampled_2': 384,
                'gaussians_upsampled_3': 512,
            }
        ) 
                        
        self.rand_base_render = True

        # further x8 up-sampling.
        self.superresolution.update(  
            dict(
                ada_CA_f4_2=GS_Adaptive_Read_Write_CA_adaptive_2dgs(
                    self.embed_dim,
                    vit_decoder.embed_dim,
                    vit_heads=vit_decoder.num_heads,
                    mlp_ratio=vit_decoder.mlp_ratio,
                    # depth=vit_decoder.depth // 6,
                    depth=1,
                    f=4,  # 
                    heads=8, 
                    no_flash_op=True,  # fails when bs>1
                    cross_attention=False),  # write
                ada_CA_f4_3=GS_Adaptive_Read_Write_CA_adaptive_2dgs(
                    self.embed_dim,
                    vit_decoder.embed_dim,
                    vit_heads=vit_decoder.num_heads,
                    mlp_ratio=vit_decoder.mlp_ratio,
                    # depth=vit_decoder.depth // 6,
                    depth=1,
                    f=3,  # 
                    heads=8, 
                    no_flash_op=True, 
                    cross_attention=False),  # write
            ),
        )



    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        # further x8 using upper class
        # TODO, merge this into ln3diff open sourced code.
        ret_dict, (gaussian_upsampled_residual_pre_activate, upsampled_global_local_query_emb) = super().vit_decode_postprocess(latent_from_vit, ret_dict, return_upsampled_residual=True)

        gaussians_upsampled_2, (gaussian_upsampled_residual_pre_activate_2, upsampled_global_local_query_emb_2) = self.superresolution['ada_CA_f4_2'](
            upsampled_global_local_query_emb,
            ret_dict['gaussians_upsampled'],
            skip_weight=self.skip_weight,
            gs_pred_fn=self.superresolution['conv_sr'],
            gs_act_fn=self._gaussian_pred_activations,
            offset_act=self.offset_act,
            gaussian_base_pre_activate=gaussian_upsampled_residual_pre_activate)


        gaussians_upsampled_3, _ = self.superresolution['ada_CA_f4_3'](
            upsampled_global_local_query_emb_2,
            gaussians_upsampled_2,
            skip_weight=self.skip_weight,
            gs_pred_fn=self.superresolution['conv_sr'],
            gs_act_fn=self._gaussian_pred_activations,
            offset_act=self.offset_act,
            gaussian_base_pre_activate=gaussian_upsampled_residual_pre_activate_2)


        ret_dict.update({
            'gaussians_upsampled_2': gaussians_upsampled_2,
            'gaussians_upsampled_3': gaussians_upsampled_3,
        }) 

        return ret_dict

