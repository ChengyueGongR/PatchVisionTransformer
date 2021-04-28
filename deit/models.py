# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.layers import StdConv2dSame, DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import HybridEmbed, PatchEmbed, DistilledVisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., use_talk=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5 
        # self.scale = qk_scale or head_dim ** -0.5 * 2 # re-scale 

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = 0. 
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.use_talk = use_talk
        if self.use_talk:
            self.proj_pre = nn.Linear(num_heads, num_heads, bias=False)
            self.proj_post = nn.Linear(num_heads, num_heads, bias=False)

        

    def forward(self, x): # /, prev_attn):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.use_talk:
            attn = self.proj_pre(attn.permute(0,2,3,1)).permute(0,3,1,2)
        attn = attn.softmax(dim=-1)
        if self.use_talk:
            attn = self.proj_post(attn.permute(0,2,3,1)).permute(0,3,1,2)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, 0. # , attn, self.bias_proj.abs().mean() # None # attn # cur_attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_talk=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, use_talk=use_talk)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    def forward(self, x):
        post_x, proj = self.attn(self.norm1(x))
        x = x + self.drop_path(post_x)
 
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, proj

    
class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None, use_talk=True, head_drop_rate=0.2):
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
        

        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, use_talk=use_talk)
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
        self.drop = nn.Dropout(head_drop_rate)
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
         
        if self.training:
            context_target = x.detach()

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        proj = 0.
        for blk in self.blocks:
            x, _ = blk(x)
            proj += _

        patches = self.norm(x)[:, 1:]
        
        x = self.norm(x)[:, 0]
        x = self.pre_logits(x)

        if self.training:
            num_batch, num_patch, num_dim = patches.size()
            img_size = int(num_patch ** .5)
            patches = patches.permute(0, 2, 1).reshape(num_batch, num_dim, img_size, img_size)
            return x, patches, 0., proj, context_target
        
        num_batch, num_patch, num_dim = patches.size()
        img_size = int(num_patch ** .5)
        patches = patches.permute(0, 2, 1).reshape(num_batch, num_dim, img_size, img_size)

        return x, patches
    
    def similarity(self, patches, context_target, high_k=7):
        num_batch, num_dim, img_size = patches.shape[0], patches.shape[1], patches.shape[-1]
        low_order = F.avg_pool2d(patches, kernel_size=high_k, stride=1, padding=(high_k - 1) // 2) - patches / (high_k ** 2)
        low_order *= (high_k ** 2) / (high_k ** 2 - 1)
        high_order = (patches.mean(dim=(2, 3), deepdim=True) - patches / (img_size ** 2)) * (img_size ** 2) / (img_size ** 2 - 1)

        pos_l = (context_target * patches.reshape(num_batch, num_dim, -1).permute(0, 2, 1)).sum(-1).reshape(-1, 1)
        neg_l = (context_target * high_order.reshape(num_batch, num_dim, -1).permute(0, 2, 1)).sum(-1).reshape(-1, 1)
        neg2_l = (context_target * low_order.reshape(num_batch, num_dim, -1).permute(0, 2, 1)).sum(-1).reshape(-1, 1)
        return - torch.log(torch.cat((pos_l, neg_l, neg2_l), dim=-1).softmax(dim=-1)[:, 0] + 1e-8).mean()
    def forward(self, x, aux_class=None):
        if self.training:
            x, patches, r_loss, proj, context_target = self.forward_features(x)
            # x, patches, patch, r_loss = self.forward_features(x)
            aux_class_left, aux_class_right, pixel_ind = aux_class
            num_batch, num_dim, img_size = patches.shape[0], patches.shape[1], patches.shape[-1] # // 2

            sim_loss = self.similarity(patches, context_target.detach()) 
            target_lst, mask_lst, targets = aux_class
            patch_loss = 0.
            for _ in range(len(mask_lst)):
                avgpool_patches = (patches * mask_lst[_].reshape(1, 1, img_size, img_size)).mean(dim=(2, 3)) * torch.ones_like(mask_lst[_]).sum() / (mask_lst[_].sum() + 1e-4)
                
                logits = self.patch_head(self.drop(avgpool_patches))
                patch_loss += mask_lst[_].sum() / torch.ones_like(mask_lst[_]).sum() * nn.KLDivLoss(reduction='batchmean')((1e-8 + logits.softmax(dim=-1)).log(), target_lst[_])
            
            x = self.head(self.drop(x))

            return x, patch_loss, sim_loss, proj
        else:
            x, patches = self.forward_features(x)
            
            x = self.head(x)
            logits = self.patch_head(patches.mean(dim=(2, 3))).softmax(dim=-1) + x.softmax(dim=-1)
            return x.softmax(dim=-1).log(), logits 


@register_model
def deit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
            img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def deit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
            img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def deit_downsamplesmall_patch8_224(pretrained=False, **kwargs):
    model = VisionTransformer(
            patch_size=8, embed_dim=144, depth=16, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def deit_downsampleextremetiny_patch8_224(pretrained=False, **kwargs):
    model = VisionTransformer(
            patch_size=8, embed_dim=192, depth=16, num_heads=2, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384,patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def deit_base_patch16_672(pretrained=False, **kwargs):
    model = VisionTransformer(
            img_size=672, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def deit_base_patch16_512(pretrained=False, **kwargs):
    model = VisionTransformer(
            img_size=512, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch14_224(pretrained=False, **kwargs):
    model = VisionTransformer(
            patch_size=14, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def deit_xdeepbase_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
            img_size=224, patch_size=16, embed_dim=768, depth=48, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def deit_deepbase_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
            img_size=224, patch_size=16, embed_dim=768, depth=24, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def deit_xbase_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
            img_size=224, patch_size=16, embed_dim=1536, depth=12, num_heads=24, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def deit_xbase_patch16_128(pretrained=False, **kwargs):
    model = VisionTransformer(
            img_size=128, patch_size=16, embed_dim=1536, depth=12, num_heads=24, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def deit_verydeepsmall_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
            img_size=224, patch_size=16, embed_dim=384, depth=48, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def deit_deepsmall_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
            img_size=224, patch_size=16, embed_dim=384, depth=24, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def deit_deepsmall_patch16_512(pretrained=False, **kwargs):
    model = VisionTransformer(
            img_size=512, patch_size=16, embed_dim=384, depth=24, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def deit_deepsmall_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
            img_size=384, patch_size=16, embed_dim=384, depth=24, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
            img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
                map_location="cpu", check_hash=True
            )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model
