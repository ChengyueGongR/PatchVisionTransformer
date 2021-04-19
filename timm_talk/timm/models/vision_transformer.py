""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import load_pretrained
from .layers import StdConv2dSame, DropPath, to_2tuple, trunc_normal_
from .resnet import resnet26d, resnet50d
from .resnetv2 import ResNetV2
from .registry import register_model
from .loss_ops import alpha_divergence, AdaptiveLossSoft
from .layers import SplitAttnConv2d 
from .deform_conv_v2 import DeformConv2d 
_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models (my experiments)
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),

    # patch models (weights ported from official Google JAX impl)
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),

    # patch models, imagenet21k (weights ported from official Google JAX impl)
    'vit_base_patch16_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_huge_patch14_224_in21k': _cfg(
        url='',  # FIXME I have weights for this but > 2GB limit for github release binaries
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),

    # hybrid models (weights ported from official Google JAX impl)
    'vit_base_resnet50_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_224_in21k-6f7c7740.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=0.9, first_conv='patch_embed.backbone.stem.conv'),
    'vit_base_resnet50_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_384-9fd3c705.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0, first_conv='patch_embed.backbone.stem.conv'),

    # hybrid models (my experiments)
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),

    # deit models (FB weights)
    'vit_deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'),
    'vit_deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'),
    'vit_deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',),
    'vit_deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth'),
    'vit_deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth'),
    'vit_deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth', ),
    'vit_deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
}


class AttentionDrop(nn.Module):
    def __init__(self, drop_rate=.02):
        super().__init__()
        self.drop_rate = drop_rate
    def generate_mask(self, num_token, seed=None):


        '''
        column = 2
        width = int(num_token ** .5)
        left_ind = random.randint(1 + column*width, num_token -1-column-column*width)
        right_ind = left_ind + column
        mask = torch.zeros_like(mask)
        mask[:, :, left_ind:right_ind, :] = -1e10 * torch.ones_like(mask[:, :, left_ind:right_ind, :])
        # mask[:, :, :, left_ind:right_ind] = -1e10 * torch.ones_like(mask[:, :, :,  left_ind:right_ind])
        for raw_item in range(column*2+1):
            item = raw_item - column
            # mask[:, :, left_ind+item*width:right_ind+item*width, left_ind:right_ind] = torch.zeros_like(mask[:, :, left_ind:right_ind, left_ind:right_ind])
            mask[:, :, left_ind:right_ind, left_ind+item*width:right_ind+item*width] = torch.zeros_like(mask[:, :, left_ind:right_ind, left_ind:right_ind])
        '''
        width = int(num_token ** .5)
        mask = torch.ones([width, width]).cuda()
        '''
        left_ind = random.randint(0, width-1)
        seed = random.randint(0, 3)
        if seed == 0:
            mask[:left_ind, :left_ind] = -1 * torch.ones_like(mask[:left_ind, :left_ind])
        elif seed == 1:
            mask[left_ind:, :left_ind] = -1 * torch.ones_like(mask[left_ind:, :left_ind])
        elif seed == 2:
            mask[:left_ind, left_ind:] = -1 * torch.ones_like(mask[:left_ind, left_ind:])
        else:
            mask[left_ind:, left_ind:] = -1 * torch.ones_like(mask[left_ind:, left_ind:])
        '''

        '''
        max_length = min(width, int(num_token * self.drop_rate))
        length = random.randint(1, max(1, max_length - 1))
        mask_width = int(num_token * self.drop_rate) // length
        if mask_width > width:
            mask_width = width
        mask_length = int(num_token * self.drop_rate) // mask_width
        '''
        
        # '''
        mask_length = 10
        mask_width = 10
        
        length_left_ind = random.randint(0, max(0, width - mask_length - 1))
        length_right_ind = length_left_ind + mask_length
        width_left_ind = random.randint(0, max(0, width - mask_width - 1))
        width_right_ind = width_left_ind + mask_width
        mask[length_left_ind:length_right_ind, width_left_ind:width_right_ind] = -1 * torch.ones_like(mask[length_left_ind:length_right_ind, width_left_ind:width_right_ind])
        # '''

        mask = torch.cat((torch.zeros([1]).cuda(), mask.reshape(-1)))
        mask = torch.matmul(mask.reshape(-1, 1), mask.reshape(1, -1))

        mask = torch.where(mask < torch.zeros_like(mask), torch.ones_like(mask) * -1e10, torch.zeros_like(mask))
        # mask[left_ind:left_ind+width, :left_ind] = -1e10 * torch.ones_like(mask[left_ind:left_ind+width, :left_ind])
        # mask[left_ind:left_ind+width, left_ind+width:] = -1e10 * torch.ones_like(mask[left_ind:left_ind+width, left_ind+width:])
        
        return mask

    
    def forward(self, x):
        if self.training:
            num_batch, num_head, num_token = x.shape[0], x.shape[1], x.shape[2]
            
            # mask = torch.cat([self.generate_mask(num_token).reshape(1, num_token, num_token) for item in range(num_head * num_batch)], dim=0)
            # x = torch.where(mask.reshape(num_batch, num_head, num_token, num_token) 

            mask = torch.cat([self.generate_mask(num_token, item % 4).reshape(1, num_token, num_token) for item in range(num_head)], dim=0)
            mask = mask.reshape(1, num_head, num_token, num_token).repeat(num_batch, 1, 1, 1)
            
            
            '''
            mask = torch.where(mask < torch.zeros_like(mask), torch.zeros_like(mask), torch.ones_like(mask))
            # re-scale
            x = mask * x
            x /= torch.sum(x, dim=-1, keepdim=True).detach()
            '''
            x = torch.where(mask < torch.zeros_like(mask), mask, x)
            
            # print(x.shape)
            return x
            # return torch.where(mask < torch.zeros_like(mask), torch.zeros_like(mask), torch.ones_like(mask)) * x
            # return x + mask.detach()
        return x
    
class Jigsaw(nn.Module):
    def __init__(self, jigsaw_size=2):
        super().__init__()
        self.jigsaw_size = jigsaw_size

    def forward(self, x):
        if self.training:
            num_batch, num_patchs, num_channels = x.size()
            subimg_size = int(num_patchs ** .5 // self.jigsaw_size)
            trans_x = x
            trans_x[:, 1:, :] = x[:, 1:, :].permute(0, 2, 1).reshape(num_batch, num_channels, subimg_size, self.jigsaw_size, subimg_size, self.jigsaw_size)[:, :, :, torch.randperm(self.jigsaw_size).cuda().long(), :, :][:, :, :, :, :, torch.randperm(self.jigsaw_size).cuda().long()].reshape(num_batch, num_channels, -1).permute(0, 2, 1)
            return trans_x
        
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5 
        # self.scale = qk_scale or head_dim ** -0.5 * 2 # re-scale 

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = 0. # AttentionDrop()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.proj_pre = nn.Linear(num_heads, num_heads, bias=False)
        self.proj_post = nn.Linear(num_heads, num_heads, bias=False)

        

    def forward(self, x): # /, prev_attn):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.proj_pre(attn.permute(0,2,3,1)).permute(0,3,1,2)
        attn = attn.softmax(dim=-1)
        attn = self.proj_post(attn.permute(0,2,3,1)).permute(0,3,1,2)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, 0. # , attn, self.bias_proj.abs().mean() # None # attn # cur_attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    def forward(self, x):
        post_x, proj = self.attn(self.norm1(x))
        x = x + self.drop_path(post_x)
        # x, attn = x + self.drop_path(self.attn(self.norm1(x), prev_attn))
        
        # x = x + self.mlp(self.norm2(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, proj


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # img_size = 672
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        # num_patches = self.patch_embed.num_patches
        num_patches = ( img_size // patch_size ) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        

        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # dpr = [min(0.6, max(x.item() - 0.1, 0.)) for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # decide whether two patches are neighbour
        self.patch_head = nn.Linear(embed_dim, num_classes) # nn.Sequential(nn.Linear(embed_dim, 512), nn.ReLU(inplace=True), nn.Linear(512, 1000))
        self.drop = nn.Dropout(.5)
        self.emb_drop = nn.Dropout(.0)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.emb_drop(self.patch_embed(x))
         
        # x = F.adaptive_avg_pool2d(x.reshape(B, 672//16, 672//16, -1).permute(0, 3, 1, 2), (512//16, 512//16)).reshape(B, -1, (512//16) **2).permute(0, 2, 1)
        if self.training:
            context_target = x# .detach()

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        proj = 0.
        count = 0
        for blk in self.blocks:
            x, _ = blk(x)
            if count == len(self.blocks) // 2:
                inter_patches = x[:, 1:, :]
            count += 1
            proj += _

        patches = self.norm(x)[:, 1:]
        
        x = self.norm(x)[:, 0]
        x = self.pre_logits(x)

        if self.training:
            # calculate relation loss
            # patches = patch[-1]
            num_batch, num_patch, num_dim = patches.size()
            img_size = int(num_patch ** .5)
            patches = patches.permute(0, 2, 1).reshape(num_batch, num_dim, img_size, img_size)
            return x, patches, 0., proj, context_target, inter_patches.permute(0, 2, 1).reshape(num_batch, num_dim, img_size, img_size)
        
        num_batch, num_patch, num_dim = patches.size()
        img_size = int(num_patch ** .5)
        patches = patches.permute(0, 2, 1).reshape(num_batch, num_dim, img_size, img_size)

        return x, patches

    def forward(self, x, aux_class=None, return_x=True):
        if self.training:
            x, patches, r_loss, proj, context_target, inter_patches = self.forward_features(x)
            # x, patches, patch, r_loss = self.forward_features(x)
            aux_class_left, aux_class_right, pixel_ind = aux_class
            num_batch, num_dim, img_size = patches.shape[0], patches.shape[1], patches.shape[-1] # // 2
            
            def similarity(patches, context_target):
                # high_order = patches

                # '''
                low_k, high_k = 5, 7
                low_order = F.avg_pool2d(patches, kernel_size=low_k, stride=1, padding=(low_k - 1) // 2) - patches / (low_k ** 2)
                low_order *= (low_k ** 2 * 1.0) / (low_k ** 2 - 1.)
                high_order = patches.mean(dim=(2, 3))
                # high_order = F.avg_pool2d(patches, kernel_size=high_k, stride=1, padding=(high_k - 1) // 2) - patches / (high_k ** 2)
                # high_order *= (high_k ** 2 * 1.0) / (high_k ** 2 - 1.)
                # '''

                
                # context_target = F.normalize(context_target, dim=-1)# .reshape(-1, num_dim)
                # high_order = F.normalize(high_order, dim=-1) # .reshape(num_batch, num_dim, -1).permute(0, 2, 1), dim=-1)# .reshape(-1, num_dim)
                
                pos_l = (context_target * patches.reshape(num_batch, num_dim, -1).permute(0, 2, 1)).sum(-1).reshape(-1, 1)
                neg_l = (context_target * high_order.reshape(num_batch, num_dim, -1).permute(0, 2, 1)).sum(-1).reshape(-1, 1)
                neg2_l = (context_target * low_order.reshape(num_batch, num_dim, -1).permute(0, 2, 1)).sum(-1).reshape(-1, 1)

                # neg_l = torch.cat([(context_target * high_order[:, torch.randperm(high_order.shape[1]).cuda()]).sum(-1).reshape(-1, 1) for _ in range(40)], dim=-1)

                return - torch.log(torch.cat((pos_l, neg_l, neg2_l), dim=-1).softmax(dim=-1)[:, 0] + 1e-8).mean()
                
                
                #neg_l = [torch.log( 1e-8 + (context_target * high_order[:, torch.randperm(high_order.shape[1]).cuda()]).sum(-1).sigmoid() ).mean() for _ in range(5)] 
                #pos_l = - torch.log( 1e-8 + (context_target * high_order).sum(-1).sigmoid() ).mean()
                #return pos_l + sum(neg_l) / len(neg_l)
                
                # return - (context_target * high_order).sum(-1).mean()
            
            sim_loss = similarity(patches, context_target.detach()) # inter_patches.reshape(num_batch, num_dim, -1).permute(0, 2, 1)) # context_target)
            # sim_loss = similarity(context_target.detach(), inter_patches.reshape(num_batch, num_dim, -1).permute(0, 2, 1))
            # sim_loss += similarity(inter_patches.reshape(num_batch, num_dim, -1).permute(0, 2, 1).detach(), patches.reshape(num_batch, num_dim, -1).permute(0, 2, 1))
            # sim_loss = similarity(context_target.detach(), patches.reshape(num_batch, num_dim, -1).permute(0, 2, 1))#, context_target) # + similarity(inter_patches, patches.reshape(num_batch, num_dim, -1).permute(0, 2, 1)) # , context_target)
            
            target_lst, mask_lst, targets = aux_class
            patch_loss = 0.
            for _ in range(len(mask_lst)):
                avgpool_patches = (patches * mask_lst[_].reshape(1, 1, img_size, img_size)).mean(dim=(2, 3)) * torch.ones_like(mask_lst[_]).sum() / (mask_lst[_].sum() + 1e-4)
                
                logits = self.patch_head(self.drop(avgpool_patches))
                # patch_loss += mask_lst[_].sum() / torch.ones_like(mask_lst[_]).sum() * torch.sum(-target_lst[_] * (1e-8 + logits.softmax(dim=-1)).log(), dim=-1).mean() 
                patch_loss += mask_lst[_].sum() / torch.ones_like(mask_lst[_]).sum() * nn.KLDivLoss(reduction='batchmean')((1e-8 + logits.softmax(dim=-1)).log(), target_lst[_])
            
                '''
                if _ == 0:
                    # left_term = logits.reshape(2, num_batch//2, -1)[0] 
                    left_term = F.normalize(logits.reshape(2, num_batch//2, -1)[0], dim=-1)
                    # right_term = logits.reshape(2, num_batch//2, -1)[1] 
                    right_term = F.normalize(logits.reshape(2, num_batch//2, -1)[1], dim=-1)
                    repeat_loss = (left_term - right_term).norm(dim=-1).mean() # / logits.shape[-1]**.5
                '''
            x = self.head(self.drop(x))

            if return_x:
                return x
            return x, r_loss * 0. + patch_loss, sim_loss, proj # +0*repeat_loss, proj # + repeat_loss * 2. # + cutmix_loss * 2. # + sim_loss * 2e-2

            # return x.softmax(dim=-1), r_loss * 0. + patch_loss * 4. + repeat_loss * 1. # + sim_loss * 2e-2
        else:
            x, patches = self.forward_features(x)
            
            x = self.head(x)
            if return_x:
                return x
            
            # logits = ( self.patch_head(patches.mean(dim=(2, 3))) + x ).softmax(dim=-1)
            logits = self.patch_head(patches.mean(dim=(2, 3))).softmax(dim=-1) + x.softmax(dim=-1)

            return x.softmax(dim=-1).log(), logits # self.patch_head(patches.mean(dim=(2, 3))).softmax(dim=-1) + x.softmax(dim=-1)
            # return x.softmax(dim=-1).log(), (self.patch_head(patches.mean(dim=(2, 3))) + x).softmax(dim=-1)


class DistilledVisionTransformer(VisionTransformer):
    """ Vision Transformer with distillation token.

    Paper: `Training data-efficient image transformers & distillation through attention` -
        https://arxiv.org/abs/2012.12877

    This impl of distilled ViT is taken from https://github.com/facebookresearch/deit
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        # multi_aux_head 
        self.neighbour_head = nn.Sequential(nn.Linear(self.embed_dim * 2, 512), nn.ReLU(inplace=True), nn.Linear(512, 2))
        self.crosslayer_head = nn.Sequential(nn.Linear(self.embed_dim * 2, 512), nn.ReLU(inplace=True), nn.Linear(512, 2))

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        count = 0
        for blk in self.blocks:
            x = blk(x)
            if self.training and count == len(self.blocks) // 2:
                inter_x = x
            count += 1

        if self.training:
            x = self.norm(x)
            patches = x[:, 2:]
            num_batch, num_patch, num_dim = patches.size()
            img_size = int(num_patch ** .5)
            patches = patches.permute(0, 2, 1).reshape(num_batch, num_dim, img_size, img_size)

            if random.randint(0, 1) == 0:
                ind = (random.randint(0, img_size-2), random.randint(0, img_size-1))
                pos_ind = (ind[0]+1, ind[1])
            else:
                ind = (random.randint(0, img_size-1), random.randint(0, img_size-2))
                pos_ind = (ind[0], ind[1]+1)
            neg_ind = (random.randint(0, img_size-1), random.randint(0, img_size-1))

            center = patches[:, :, ind[0], ind[1]].reshape(num_batch, num_dim)
            pos = patches[:, :, pos_ind[0], pos_ind[1]].reshape(num_batch, num_dim)
            neg = patches[:, :, neg_ind[0], neg_ind[1]].reshape(num_batch, num_dim)
            input_data = torch.cat((torch.cat((center, pos), dim=1), torch.cat((center, neg), dim=1)), dim=0)
            p_label = torch.cat((torch.ones_like(pos[:, 0]), torch.zeros_like(neg[:, 0])), dim=0).long()
            
            relation_loss =  F.cross_entropy(self.neighbour_head(input_data),  p_label)
            
            inter_patches = inter_x[:, 2:]
            inter_patches = inter_patches.permute(0, 2, 1).reshape(num_batch, num_dim, img_size, img_size)
            ind = (random.randint(0, img_size-1), random.randint(0, img_size-1))
            neg_ind = (random.randint(0, img_size-1), random.randint(0, img_size-1))
            center = inter_patches[:, :, ind[0], ind[1]].reshape(num_batch, num_dim)
            pos = patches[:, :, ind[0], ind[1]].reshape(num_batch, num_dim).detach()
            neg = inter_patches[:, :, neg_ind[0], neg_ind[1]].reshape(num_batch, num_dim)


            input_data = torch.cat((torch.cat((center, pos), dim=1), torch.cat((center, neg), dim=1)), dim=0)
            p_label = torch.cat((torch.ones_like(pos[:, 0]), torch.zeros_like(neg[:, 0])), dim=0).long()
            crosslayer_loss = F.cross_entropy(self.crosslayer_head(input_data),  p_label)
            return x[:, 0], x[:, 1], relation_loss + crosslayer_loss
        
        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x, aux_class=None):
        if self.training:
            x, x_dist, r_loss = self.forward_features(x)
            aux_class_left, aux_class_right, pixel_ind = aux_class

            num_batch, num_dim, img_size = patches.shape[0], patches.shape[1], patches.shape[-1] # // 2

            patches_left = (patches * pixel_ind.reshape(1, 1, img_size, img_size)).mean(dim=(2, 3))
            patches_right = (patches * (1. - pixel_ind.reshape(1, 1, img_size, img_size))).mean(dim=(2, 3))
            
            patch_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(self.patch_head(patches_left), dim=-1), aux_class_left) + \
                         nn.KLDivLoss(reduction='batchmean')(F.log_softmax(self.patch_head(patches_right), dim=-1), aux_class_right)

            x = self.head(x)
            x_dist = self.head_dist(x_dist)
            return (x, x_dist), r_loss + patch_loss * 2.
        else:
            x, x_dist = self.forward_features(x)
            x = self.head(x)
            x_dist = self.head_dist(x_dist)
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, distilled=False, **kwargs):
    default_cfg = default_cfgs[variant]
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-1]

    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model_cls = DistilledVisionTransformer if distilled else VisionTransformer
    model = model_cls(img_size=img_size, num_classes=num_classes, representation_size=repr_size, **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(
            model, num_classes=num_classes, in_chans=kwargs.get('in_chans', 3),
            filter_fn=partial(checkpoint_filter_fn, model=model))
    return model


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    """ My custom 'small' ViT model. Depth=8, heads=8= mlp_ratio=3."""
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3.,
        qkv_bias=False, norm_layer=nn.LayerNorm, **kwargs)
    if pretrained:
        # NOTE my scale was wrong for original weights, leaving this here until I have better ones for this model
        model_kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=768, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, representation_size=768, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, representation_size=1024, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=1024, depth=24, num_heads=16, representation_size=1024, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_huge_patch14_224_in21k(pretrained=False, **kwargs):
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model_kwargs = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, representation_size=1280, **kwargs)
    model = _create_vision_transformer('vit_huge_patch14_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_resnet50_224_in21k(pretrained=False, **kwargs):
    """ R50+ViT-B/16 hybrid model from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    # create a ResNetV2 w/o pre-activation, that uses StdConv and GroupNorm and has 3 stages, no head
    backbone = ResNetV2(
        layers=(3, 4, 9), num_classes=0, global_pool='', in_chans=kwargs.get('in_chans', 3),
        preact=False, stem_type='same', conv_layer=StdConv2dSame)
    model_kwargs = dict(
        embed_dim=768, depth=12, num_heads=12, hybrid_backbone=backbone,
        representation_size=768, **kwargs)
    model = _create_vision_transformer('vit_base_resnet50_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_resnet50_384(pretrained=False, **kwargs):
    """ R50+ViT-B/16 hybrid from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    # create a ResNetV2 w/o pre-activation, that uses StdConv and GroupNorm and has 3 stages, no head
    backbone = ResNetV2(
        layers=(3, 4, 9), num_classes=0, global_pool='', in_chans=kwargs.get('in_chans', 3),
        preact=False, stem_type='same', conv_layer=StdConv2dSame)
    model_kwargs = dict(embed_dim=768, depth=12, num_heads=12, hybrid_backbone=backbone, **kwargs)
    model = _create_vision_transformer('vit_base_resnet50_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_resnet26d_224(pretrained=False, **kwargs):
    """ Custom ViT small hybrid w/ ResNet26D stride 32. No pretrained weights.
    """
    backbone = resnet26d(pretrained=pretrained, in_chans=kwargs.get('in_chans', 3), features_only=True, out_indices=[4])
    model_kwargs = dict(embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, hybrid_backbone=backbone, **kwargs)
    model = _create_vision_transformer('vit_small_resnet26d_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_resnet50d_s3_224(pretrained=False, **kwargs):
    """ Custom ViT small hybrid w/ ResNet50D 3-stages, stride 16. No pretrained weights.
    """
    backbone = resnet50d(pretrained=pretrained, in_chans=kwargs.get('in_chans', 3), features_only=True, out_indices=[3])
    model_kwargs = dict(embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, hybrid_backbone=backbone, **kwargs)
    model = _create_vision_transformer('vit_small_resnet50d_s3_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_resnet26d_224(pretrained=False, **kwargs):
    """ Custom ViT base hybrid w/ ResNet26D stride 32. No pretrained weights.
    """
    backbone = resnet26d(pretrained=pretrained, in_chans=kwargs.get('in_chans', 3), features_only=True, out_indices=[4])
    model_kwargs = dict(embed_dim=768, depth=12, num_heads=12, hybrid_backbone=backbone, **kwargs)
    model = _create_vision_transformer('vit_base_resnet26d_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_resnet50d_224(pretrained=False, **kwargs):
    """ Custom ViT base hybrid w/ ResNet50D stride 32. No pretrained weights.
    """
    backbone = resnet50d(pretrained=pretrained, in_chans=kwargs.get('in_chans', 3), features_only=True, out_indices=[4])
    model_kwargs = dict(embed_dim=768, depth=12, num_heads=12, hybrid_backbone=backbone, **kwargs)
    model = _create_vision_transformer('vit_base_resnet50d_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_deit_tiny_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_deit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_deit_small_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_deit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_deit_base_patch16_224(pretrained=False, **kwargs):
    """ DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_deit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_deit_base_patch16_384(pretrained=False, **kwargs):
    """ DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_deit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer(
        'vit_deit_tiny_distilled_patch16_224', pretrained=pretrained,  distilled=True, **model_kwargs)
    return model


@register_model
def vit_deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        'vit_deit_small_distilled_patch16_224', pretrained=pretrained,  distilled=True, **model_kwargs)
    return model


@register_model
def vit_deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        'vit_deit_base_distilled_patch16_224', pretrained=pretrained,  distilled=True, **model_kwargs)
    return model


@register_model
def vit_deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        'vit_deit_base_distilled_patch16_384', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


def reg_loss(x, tau=5e-2):
    x_nocls = x[:, 1:, :]
    batch_size, num_patch, num_dim = x_nocls.shape[0], x_nocls.shape[1], x_nocls.shape[2]
    
    # n_outputs = nn.functional.normalize(x_nocls, dim=-1)
    n_outputs = n_outputs.transpose(1, 2).reshape(batch_size, num_dim, int(num_patch**.5), int(num_patch**.5))

    width = int(num_patch**.5)
    center = n_outputs[:, :, 1:width-1, 1:width-1].reshape(batch_size, num_dim, -1).permute(0, 2, 1)
    pos_lst = []
    select_lst = [(0, 2), (1, 1), (2, 0)]
    for first_ind in range(3):
        for second_ind in range(3):
            if first_ind == 1 and second_ind == 1:
                continue
            pos_lst.append(n_outputs[:, :, select_lst[first_ind][0]:width-select_lst[first_ind][1],
                select_lst[second_ind][0]:width-select_lst[second_ind][1]].reshape(batch_size, num_dim, -1).permute(0, 2, 1))
    pos_pair = torch.log(1e-10 + sum([torch.exp((center * item).sum(-1)) for item in pos_lst]) / 8).reshape(batch_size, (width - 2)**2, 1)
    # pos_pair = sum([torch.exp((center * item).sum(-1)) for item in pos_lst]).reshape(batch_size, (width - 2)**2, 1) / 8
    center = center.reshape(batch_size, num_dim, -1).transpose(1, 2)

    # neg_pair = torch.bmm(center, - center.permute(0, 2, 1))
    random_outputs = center.reshape(-1, num_dim)[torch.randperm(batch_size * (width - 2)**2)].reshape(batch_size, (width - 2)**2, num_dim)
    # random_outputs =  n_latent[:, :, 1:width-1, 1:width-1].reshape(batch_size, num_dim, -1).permute(0, 2, 1)
    inbatch_neg_pair = torch.bmm(center, random_outputs.permute(0, 2, 1))

    pairs = torch.cat([pos_pair, inbatch_neg_pair], dim=-1).reshape(-1, (width - 2)**2+1)
    # pairs = torch.cat([pos_pair, neg_pair, inbatch_neg_pair], dim=-1).reshape(-1, (width - 2)**2*2+1)
    pusdo_label = torch.zeros_like(pairs[:, 0]).reshape(-1).long()
    return F.cross_entropy(pairs, pusdo_label) * tau

