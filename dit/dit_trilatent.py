import torch.nn as nn
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from pdb import set_trace as st
from timm.models.vision_transformer import Mlp

from ldm.modules.attention import MemoryEfficientCrossAttention
from .dit_models_xformers import DiT, get_2d_sincos_pos_embed, DiTBlock, FinalLayer, t2i_modulate, PixelArtTextCondDiTBlock, T2IFinalLayer, approx_gelu

from torch.nn import LayerNorm
from vit.vit_triplane import XYZPosEmbed
import os


def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

def weighted_average(list_0, list_1, alpha):
    
    if len(list_0) != len(list_1):
        raise ValueError("两个列表的长度必须相同！")
    if not (0 <= alpha <= 1):
        raise ValueError("alpha 值必须在 [0, 1] 范围内！")
    
    # 使用列表解析对每个位置计算加权平均
    return [(1 - alpha) * a_0 + alpha * a_1 for a_0, a_1 in zip(list_0, list_1)]

def enhance_low_frequencies(input_tensor, scale_factor=5):
    
    batchsize, seq_len, channels = input_tensor.shape

    
    output_tensor = torch.zeros_like(input_tensor)

    for batch_idx in range(batchsize):
        
        data = input_tensor[batch_idx].to(dtype=torch.float32)  

        
        freq_data = torch.fft.fft(data)  

        
        num_low_freq = seq_len // 2  
        freq_data[:num_low_freq] *= scale_factor

        
        enhanced_data = torch.fft.ifft(freq_data).real  

        
        output_tensor[batch_idx] = enhanced_data.to(dtype=torch.bfloat16)

    return output_tensor

def update_tensor_with_interpolation(x, x_0, x_1, alpha):
    
    batch_size, num_points, _ = x_0.shape
    reordered_x_1 = torch.zeros_like(x_1)  
    interpolated_x = torch.zeros_like(x)  

    for batch_idx in range(batch_size):  
       
        v1_points = x_0[batch_idx].clone()  
        v2_points = x_1[batch_idx].clone()  

       
        freq_v1 = torch.fft.fft(v1_points.to(dtype=torch.float32), dim=0)  # [N, 3]
        freq_v2 = torch.fft.fft(v2_points.to(dtype=torch.float32), dim=0)  # [N, 3]

       
        low_freq_v1 = freq_v1[:num_points // 2]  # [N//2, 3]
        low_freq_v2 = freq_v2[:num_points // 2]  # [N//2, 3]
        
        
        distances = torch.cdist(torch.abs(low_freq_v1).unsqueeze(0), torch.abs(low_freq_v2).unsqueeze(0)).squeeze(0)

        
        closest_indices = torch.argmin(distances, dim=1)  # [low_freq_N]

        
        reordered_low_freq_v2 = low_freq_v2[closest_indices]  # [low_freq_N, 3]

        
        reordered_freq_v2 = freq_v2.clone()
        reordered_freq_v2[:num_points // 2] = reordered_low_freq_v2

        
        reordered_x_1[batch_idx] = torch.fft.ifft(reordered_freq_v2, dim=0).real.to(dtype=torch.bfloat16)

        
        interpolated_freq = (1 - alpha) * freq_v1 + alpha * reordered_freq_v2
        
        interpolated_x[batch_idx] = torch.fft.ifft(interpolated_freq, dim=0).real.to(dtype=torch.bfloat16)

    return interpolated_x, x_0, reordered_x_1



class DiT_TriLatent(DiT):
    # DiT with 3D_aware operations
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mixing_logit_init=-3,
        mixed_prediction=True,
        context_dim=False,
        roll_out=False,
        vit_blk=DiTBlock,
        final_layer_blk=FinalLayer,
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                         mixed_prediction, context_dim, roll_out, vit_blk,
                         final_layer_blk)

        assert self.roll_out

    def init_PE_3D_aware(self):

        self.pos_embed = nn.Parameter(torch.zeros(
            1, self.plane_n * self.x_embedder.num_patches, self.embed_dim),
                                      requires_grad=False)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        p = int(self.x_embedder.num_patches**0.5)
        D = self.pos_embed.shape[-1]
        grid_size = (self.plane_n, p * p)  # B n HW C

        pos_embed = get_2d_sincos_pos_embed(D, grid_size).reshape(
            self.plane_n * p * p, D)  # H*W, D

        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

    def initialize_weights(self):
        super().initialize_weights()

        # ! add 3d-aware PE
        self.init_PE_3D_aware()

    def forward(self,
                x,
                timesteps=None,
                context=None,
                y=None,
                get_attr='',
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # t = timesteps
        assert context is not None

        t = self.t_embedder(timesteps)  # (N, D)

        # if self.roll_out:  # !
        x = rearrange(x, 'b (c n) h w->(b n) c h w',
                      n=3)  # downsample with same conv
        x = self.x_embedder(x)  # (b n) c h/f w/f

        x = rearrange(x, '(b n) l c -> b (n l) c', n=3)
        x = x + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        # if self.roll_out:  # ! roll-out in the L dim, not B dim. add condition to all tokens.
        # x = rearrange(x, '(b n) l c ->b (n l) c', n=3)

        # assert context.ndim == 2
        if isinstance(context, dict):
            context = context['crossattn']  # sgm conditioner compat

        context = self.clip_text_proj(context)

        # c = t + context
        # else:
        # c = t  # BS 1024

        for blk_idx, block in enumerate(self.blocks):
            # if self.roll_out:
            if False:
                if blk_idx % 2 == 0:  # with-in plane self attention
                    x = rearrange(x, 'b (n l) c -> (b n) l c', n=3)
                    x = block(x, repeat(t, 'b c -> (b n) c ', n=3), # TODO, calculate once
                              repeat(context, 'b l c -> (b n) l c ', n=3))  # (N, T, D)

                else:  # global attention
                    x = rearrange(x, '(b n) l c -> b (n l) c ', n=self.plane_n)
                    x = block(x, t, context)  # (N, T, D)
            else:
                x = block(x, t, context)  # (N, T, D)

        # todo later
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)

        if self.roll_out:  # move n from L to B axis
            x = rearrange(x, 'b (n l) c ->(b n) l c', n=3)

        x = self.unpatchify(x)  # (N, out_channels, H, W)

        if self.roll_out:  # move n from L to B axis
            x = rearrange(x, '(b n) c h w -> b (c n) h w', n=3)
            # x = rearrange(x, 'b n) c h w -> b (n c) h w', n=3)

        # cast to float32 for better accuracy
        x = x.to(torch.float32).contiguous()
        # st()

        return x


class DiT_TriLatent_PixelArt(DiT_TriLatent):
    # DiT with 3D_aware operations
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mixing_logit_init=-3,
        mixed_prediction=True,
        context_dim=False,
        roll_out=False,
        vit_blk=DiTBlock,
        final_layer_blk=FinalLayer,
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                         mixed_prediction, context_dim, roll_out, vit_blk,
                         final_layer_blk)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        del self.clip_text_proj
        self.cap_embedder = nn.Sequential( # TODO, init with zero here.
            LayerNorm(context_dim),
            nn.Linear(
                context_dim,
                hidden_size,
            ),
        )
        nn.init.constant_(self.cap_embedder[-1].weight, 0)
        nn.init.constant_(self.cap_embedder[-1].bias, 0)


    def forward(self,
                x,
                timesteps=None,
                context=None,
                y=None,
                get_attr='',
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # t = timesteps
        assert context is not None

        clip_cls_token = self.cap_embedder(context['vector']) # pooled
        t = self.t_embedder(timesteps) + clip_cls_token  # (N, D)
        t0 = self.adaLN_modulation(t) # single-adaLN, B 6144

        # if self.roll_out:  # !
        x = rearrange(x, 'b (c n) h w->(b n) c h w',
                      n=3)  # downsample with same conv
        x = self.x_embedder(x)  # (b n) c h/f w/f

        x = rearrange(x, '(b n) l c -> b (n l) c', n=3)
        x = x + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        # if self.roll_out:  # ! roll-out in the L dim, not B dim. add condition to all tokens.
        # x = rearrange(x, '(b n) l c ->b (n l) c', n=3)

        # assert context.ndim == 2
        if isinstance(context, dict):
            context = context['crossattn']  # sgm conditioner compat

        # context = self.clip_text_proj(context) # ! with rmsnorm here for 

        # c = t + context
        # else:
        # c = t  # BS 1024

        for blk_idx, block in enumerate(self.blocks):
            x = block(x, t0, context)  # (N, T, D)

        # todo later
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)

        if self.roll_out:  # move n from L to B axis
            x = rearrange(x, 'b (n l) c ->(b n) l c', n=3)

        x = self.unpatchify(x)  # (N, out_channels, H, W)

        if self.roll_out:  # move n from L to B axis
            x = rearrange(x, '(b n) c h w -> b (c n) h w', n=3)
            # x = rearrange(x, 'b n) c h w -> b (n c) h w', n=3)

        # cast to float32 for better accuracy
        x = x.to(torch.float32).contiguous()
        # st()

        return x

    # ! compat issue
    def forward_with_cfg(self, x, t, context, cfg_scale):
        """
        Forward pass of SiT, but also batches the unconSiTional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        # half = x[: len(x) // 2]
        # combined = torch.cat([half, half], dim=0)
        eps = self.forward(x, t, context)
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return eps


# PCD, general single-stage model.

class DiT_PCD_PixelArt(DiT_TriLatent_PixelArt):
    # DiT with 3D_aware operations
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mixing_logit_init=-3,
        mixed_prediction=True,
        context_dim=False,
        roll_out=False,
        vit_blk=PixelArtTextCondDiTBlock,
        final_layer_blk=FinalLayer,
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                         mixed_prediction, context_dim, roll_out, vit_blk,
                         final_layer_blk)
        # an MLP to transform the input 19-dim feature to high-dim.
        self.x_embedder = Mlp(in_features=in_channels,
                          hidden_features=hidden_size,
                          out_features=hidden_size,
                          act_layer=approx_gelu,
                          drop=0)
        del self.pos_embed
        
        self.use_attn = False
        self.alpha = 1000
        self.context_0 = None
        self.context_1 = None
        self.cfg_scale = None


    def forward(self,
                x_x_0_x_1,
                timesteps=None,
                context=None,
                use_attn=None, 
                alpha=None,
                y=None,
                get_attr='',
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        
        
        step_idx = timesteps[0].item()
        step_idx = int(step_idx * 250)  
        x = torch.cat([x_x_0_x_1[:2],x_x_0_x_1[:2],x_x_0_x_1[:2]],dim=0)
        x_0 = torch.cat([x_x_0_x_1[2:4],x_x_0_x_1[2:4],x_x_0_x_1[2:4]],dim=0)
        x_1 = torch.cat([x_x_0_x_1[4:],x_x_0_x_1[4:],x_x_0_x_1[4:]],dim=0)
        
        
        
        if self.use_attn == True and step_idx >= 160:
            self.use_attn = False
        
        print('stage1 dit:', step_idx, 'use atten: ', self.use_attn, 'alpha:', self.alpha)
            
        clip_cls_token = self.cap_embedder(context['caption_vector']) # pooled
        t = self.t_embedder(timesteps)  + torch.cat([clip_cls_token,clip_cls_token,clip_cls_token],dim=0) # (N, D)
        t0 = self.adaLN_modulation(t) # single-adaLN, B 6144
        
        clip_cls_token_0 = self.cap_embedder(self.context_0['caption_vector']) # pooled
        t_0 = self.t_embedder(timesteps)  +  torch.cat([clip_cls_token_0,clip_cls_token_0,clip_cls_token_0],dim=0)  # (N, D)
        t0_0 = self.adaLN_modulation(t_0) # single-adaLN, B 6144
        
        clip_cls_token_1 = self.cap_embedder(self.context_1['caption_vector']) # pooled
        t_1 = self.t_embedder(timesteps)  +  torch.cat([clip_cls_token_1,clip_cls_token_1,clip_cls_token_1],dim=0)  # (N, D)
        t0_1 = self.adaLN_modulation(t_1) # single-adaLN, B 6144
        
        x = self.x_embedder(x)
        x_0 = self.x_embedder(x_0)
        x_1 = self.x_embedder(x_1)
        
        # assert context.ndim == 2
        if isinstance(context, dict):
            context = context['caption_crossattn']  # sgm conditioner compat
            context = torch.cat([context,context,context],dim=0)
            context_0 = self.context_0['caption_crossattn']  # sgm conditioner compat
            context_0 = torch.cat([context_0,context_0,context_0],dim=0)
            context_1 = self.context_1['caption_crossattn']  # sgm conditioner compat
            context_1 = torch.cat([context_1,context_1,context_1],dim=0)
            
        for blk_idx, block in enumerate(self.blocks):
            
            # reorder
            if (self.use_attn == True) and ( 200 <= step_idx <= 230 ) and (0 <= blk_idx <= 24):
                if self.alpha <= 0.5:
                    x, x_0,x_1 = update_tensor_with_interpolation(x.clone(), x_0.clone(), x_1.clone(), self.alpha)
                else:
                    x, x_1,x_0 = update_tensor_with_interpolation(x.clone(), x_1.clone(), x_0.clone(), 1-self.alpha)
            
            # signal strengthen
            if (self.use_attn == True) and ( 200 <= step_idx <= 230 ) :
                x = enhance_low_frequencies(x)
                x_0 = enhance_low_frequencies(x_0)
                x_1 = enhance_low_frequencies(x_1)
            
            x, x_0, x_1  = block(x=x, t=t0, context=context, 
                                            alpha=self.alpha,
                                            use_attn_interpolation=self.use_attn,
                                            t_0=t0_0, x_0=x_0, context_0=context_0,
                                            t_1=t0_1, x_1=x_1, context_1=context_1)  # (N, T, D)
            
            
        
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = x.to(torch.float32).contiguous()
        
        x_0 = self.final_layer(x_0, t_0)  # (N, T, patch_size ** 2 * out_channels)
        x_0 = x_0.to(torch.float32).contiguous()
        
        x_1 = self.final_layer(x_1, t_1)  # (N, T, patch_size ** 2 * out_channels)
        x_1 = x_1.to(torch.float32).contiguous()
        
        # print('after final layer: ', torch.equal(x,self.x_0))
        return torch.cat([x[:2],x_0[:2],x_1[:2]],dim=0) 
            
        
    
    def forward_with_cfg_attn(self, 
                              x_x_0_x_1=None, 
                              t=None, 
                              context=None, 
                              cfg_scale=0, 
                              use_attn=False, 
                              alpha=1):
        """
        Forward pass of SiT, but also batches the unconSiTional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        # half = x[: len(x) // 2]
        # combined = torch.cat([half, half], dim=0)
        
        # print('before forward check: ', x_x_0_x_1.shape)
        x_x_0_x_1 = self.forward(x_x_0_x_1, t, context, use_attn, alpha)
        # print('after forward check:', x_x_0_x_1.shape)
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        
        eps = x_x_0_x_1[:2]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        
        x_0 = x_x_0_x_1[2:4]
        cond_eps, uncond_eps = torch.split(x_0, len(x_0) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        x_0 = torch.cat([half_eps, half_eps], dim=0)
        
        x_1 = x_x_0_x_1[4:]
        cond_eps, uncond_eps = torch.split(x_1, len(x_1) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        x_1 = torch.cat([half_eps, half_eps], dim=0)
        
        eps = torch.cat([eps,x_0,x_1], dim=0)
        
        return eps

# ! two-stage version, the second-stage here, for text pretraining.
class DiT_PCD_PixelArt_tofeat(DiT_PCD_PixelArt):
    # DiT with 3D_aware operations
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mixing_logit_init=-3,
        mixed_prediction=True,
        context_dim=False,
        roll_out=False,
        vit_blk=DiTBlock,
        final_layer_blk=FinalLayer,
        use_pe_cond=True,
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                         mixed_prediction, context_dim, roll_out, PixelArtTextCondDiTBlock,
                         final_layer_blk)

        self.use_pe_cond = use_pe_cond
        if use_pe_cond:
            self.xyz_pos_embed = XYZPosEmbed(hidden_size)
        else:
            self.x_embedder = Mlp(in_features=in_channels+3,
                            hidden_features=hidden_size,
                            out_features=hidden_size,
                            act_layer=approx_gelu,
                            drop=0)
            
        
            
        self.use_attn = False
        self.alpha = 1000
        # self.x_0 = None
        # self.x_1 = None
        self.context_0 = None
        self.context_1 = None

    def forward(self,
                x_x_0_x_1,
                timesteps=None,
                context=None,
                use_attn=None, 
                alpha=None,
                y=None,
                get_attr='',
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # t = timesteps
        # assert isinstance(context, dict) # yangsonglin注释掉的
       

        # dino_spatial_token = rearrange(context['concat'], 'b v l c -> b (v l) c') # flatten MV dino features.

        # t = self.t_embedder(timesteps)
        
        step_idx = timesteps[0].item()
        step_idx = int(step_idx * 250)  # 将 step_idx 乘以 250，并取整数部分
        x = torch.cat([x_x_0_x_1[:2],x_x_0_x_1[:2],x_x_0_x_1[:2]],dim=0)
        x_0 = torch.cat([x_x_0_x_1[2:4],x_x_0_x_1[2:4],x_x_0_x_1[2:4]],dim=0)
        x_1 = torch.cat([x_x_0_x_1[4:],x_x_0_x_1[4:],x_x_0_x_1[4:]],dim=0)
        
        
        if self.use_attn == True and step_idx >= 5:
            self.use_attn = False
        
        print('stage2 dit:', step_idx, 'use atten: ', self.use_attn)
        
        # if self.use_attn == True:
        #     print('stage2 dit:', step_idx, 'step: ', torch.equal(x,x_0))
        clip_cls_token = self.cap_embedder(context['caption_vector']) # pooled
        t = self.t_embedder(timesteps)  + torch.cat([clip_cls_token,clip_cls_token,clip_cls_token],dim=0) # (N, D)
        t0 = self.adaLN_modulation(t) # single-adaLN, B 6144
        
        clip_cls_token_0 = self.cap_embedder(self.context_0['caption_vector']) # pooled
        t_0 = self.t_embedder(timesteps)  +  torch.cat([clip_cls_token_0,clip_cls_token_0,clip_cls_token_0],dim=0)  # (N, D)
        t0_0 = self.adaLN_modulation(t_0) # single-adaLN, B 6144
        
        clip_cls_token_1 = self.cap_embedder(self.context_1['caption_vector']) # pooled
        t_1 = self.t_embedder(timesteps)  +  torch.cat([clip_cls_token_1,clip_cls_token_1,clip_cls_token_1],dim=0)  # (N, D)
        t0_1 = self.adaLN_modulation(t_1) # single-adaLN, B 6144
        
        caption_crossattn, fps_xyz = context['caption_crossattn'], context['fps-xyz']
        caption_crossattn = torch.cat([caption_crossattn,caption_crossattn,caption_crossattn],dim=0)
        fps_xyz = torch.cat([fps_xyz,fps_xyz,fps_xyz],dim=0)
        
        caption_crossattn_0, fps_xyz_0 = self.context_0['caption_crossattn'], self.context_0['fps-xyz']
        caption_crossattn_0 = torch.cat([caption_crossattn_0,caption_crossattn_0,caption_crossattn_0],dim=0)
        fps_xyz_0 = torch.cat([fps_xyz_0,fps_xyz_0,fps_xyz_0],dim=0)
        
        caption_crossattn_1, fps_xyz_1 = self.context_1['caption_crossattn'], self.context_1['fps-xyz']
        caption_crossattn_1 = torch.cat([caption_crossattn_1,caption_crossattn_1,caption_crossattn_1],dim=0)
        fps_xyz_1 = torch.cat([fps_xyz_1,fps_xyz_1,fps_xyz_1],dim=0)


        if self.use_pe_cond:
            x = self.x_embedder(x) + self.xyz_pos_embed(fps_xyz) # point-wise addition
            x_0 = self.x_embedder(x_0) + self.xyz_pos_embed(fps_xyz_0) # point-wise addition
            x_1 = self.x_embedder(x_1) + self.xyz_pos_embed(fps_xyz_1) # point-wise addition
        else: # use concat to add info
            x = torch.cat([fps_xyz, x], dim=-1)
            x = self.x_embedder(x)
            x_0 = torch.cat([fps_xyz_0, x_0], dim=-1)
            x_0 = self.x_embedder(x_0)
            x_1 = torch.cat([fps_xyz_1, x_1], dim=-1)
            x_1 = self.x_embedder(x_1)
        
        for blk_idx, block in enumerate(self.blocks): 
            
            if (self.use_attn == True) and ( 200 <= step_idx <= 250 ) and (0 <= blk_idx <= 24):
                if self.alpha <= 0.5:
                    x, x_0,x_1 = update_tensor_with_interpolation(x.clone(), x_0.clone(), x_1.clone(), self.alpha)
                else:
                    x, x_1,x_0 = update_tensor_with_interpolation(x.clone(), x_1.clone(), x_0.clone(), 1-self.alpha)
            
            if (self.use_attn == True) and ( 200 <= step_idx <= 230 ) :
                x = enhance_low_frequencies(x)
                x_0 = enhance_low_frequencies(x_0)
                x_1 = enhance_low_frequencies(x_1)
               
            x, x_0, x_1 = block(x=x, t=t0, context=caption_crossattn, 
                                        alpha=self.alpha,
                                        use_attn_interpolation=self.use_attn,
                                        t_0=t0_0, x_0=x_0, context_0=caption_crossattn_0,
                                        t_1=t0_1, x_1=x_1, context_1=caption_crossattn_1)  # (N, T, D)
            
            
            
            
        x = self.final_layer(x, t) # no loss on the xyz side 
        x = x.to(torch.float32).contiguous()
        
        x_0 = self.final_layer(x_0, t_0) # no loss on the xyz side 
        x_0 = x_0.to(torch.float32).contiguous()
        
        x_1 = self.final_layer(x_1, t_1) # no loss on the xyz side 
        x_1 = x_1.to(torch.float32).contiguous()
        
        return torch.cat([x[:2],x_0[:2],x_1[:2]],dim=0)
        

        
    def forward_with_cfg_attn(self, 
                              x_x_0_x_1=None, 
                              t=None, 
                              context=None, 
                              cfg_scale=0, 
                              use_attn=False, 
                              alpha=1):
        """
        Forward pass of SiT, but also batches the unconSiTional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        # half = x[: len(x) // 2]
        # combined = torch.cat([half, half], dim=0)
        
        # print('before forward check: ', x_x_0_x_1.shape)
        x_x_0_x_1 = self.forward(x_x_0_x_1, t, context, use_attn, alpha)
        # print('after forward check:', x_x_0_x_1.shape)
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        
        eps = x_x_0_x_1[:2]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        
        x_0 = x_x_0_x_1[2:4]
        cond_eps, uncond_eps = torch.split(x_0, len(x_0) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        x_0 = torch.cat([half_eps, half_eps], dim=0)
        # print('处理完看一下是否一样：',torch.equal(eps,x_0))
        
        x_1 = x_x_0_x_1[4:]
        cond_eps, uncond_eps = torch.split(x_1, len(x_1) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        x_1 = torch.cat([half_eps, half_eps], dim=0)
        
        eps = torch.cat([eps,x_0,x_1], dim=0)
        
        return eps


#################################################################################
#                                   DiT_TriLatent Configs                                  #
#################################################################################


def DiT_XL_2(**kwargs):
    return DiT_TriLatent(depth=28,
                         hidden_size=1152,
                         patch_size=2,
                         num_heads=16,
                         **kwargs)


def DiT_L_2(**kwargs):
    return DiT_TriLatent(depth=24,
                         hidden_size=1024,
                         patch_size=2,
                         num_heads=16,
                         **kwargs)


def DiT_B_2(**kwargs):
    return DiT_TriLatent(depth=12,
                         hidden_size=768,
                         patch_size=2,
                         num_heads=12,
                         **kwargs)


def DiT_B_1(**kwargs):
    return DiT_TriLatent(depth=12,
                         hidden_size=768,
                         patch_size=1,
                         num_heads=12,
                         **kwargs)


def DiT_B_Pixelart_2(**kwargs):
    return DiT_TriLatent_PixelArt(depth=12,
                         hidden_size=768,
                         patch_size=2,
                         num_heads=12,
                        #  vit_blk=PixelArtTextCondDiTBlock,
                         final_layer_blk=T2IFinalLayer,
                         **kwargs)

def DiT_L_Pixelart_2(**kwargs):
    return DiT_TriLatent_PixelArt(depth=24,
                         hidden_size=1024,
                         patch_size=2,
                         num_heads=16,
                        #  vit_blk=PixelArtTextCondDiTBlock,
                         final_layer_blk=T2IFinalLayer,
                         **kwargs)


# PCD-DiT
def DiT_PCD_B(**kwargs):

    return DiT_PCD_PixelArt(depth=12,
                         hidden_size=768,
                         patch_size=1,
                         num_heads=12,
                         **kwargs)

def DiT_PCD_L(**kwargs):

    return DiT_PCD_PixelArt(depth=24,
                         hidden_size=1024,
                         patch_size=1,
                         num_heads=16,
                         **kwargs)

def DiT_PCD_B_tofeat(**kwargs):

    return DiT_PCD_PixelArt_tofeat(depth=12,
                         hidden_size=768,
                         patch_size=1,
                         num_heads=12,
                         **kwargs)

def DiT_PCD_L_tofeat(**kwargs):

    return DiT_PCD_PixelArt_tofeat(depth=24,
                         hidden_size=1024,
                         patch_size=1,
                         num_heads=16,
                         **kwargs)

def DiT_PCD_XL_tofeat(**kwargs):

    return DiT_PCD_PixelArt_tofeat(depth=28,
                         hidden_size=1152,
                         patch_size=1,
                         num_heads=16,
                         **kwargs)




DiT_models = {
    'DiT-XL/2': DiT_XL_2,
    'DiT-L/2': DiT_L_2,
    'DiT-PixelArt-L/2': DiT_L_Pixelart_2,
    'DiT-PixelArt-B/2': DiT_B_Pixelart_2,
    'DiT-B/2': DiT_B_2,
    'DiT-B/1': DiT_B_1,
    'DiT-PCD-B': DiT_PCD_B,
    'DiT-PCD-L': DiT_PCD_L,
    'DiT-PCD-B-stage2-xyz2feat': DiT_PCD_B_tofeat,
    'DiT-PCD-L-stage2-xyz2feat': DiT_PCD_L_tofeat,
    'DiT-PCD-XL-stage2-xyz2feat': DiT_PCD_XL_tofeat,
    # 'DiT-PCD-L-stage1-text': DiT_PCD_L_tofeat,
}
