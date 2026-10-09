# Adapted in part from V-JEPA 2 (https://github.com/facebookresearch/vjepa2),
# Copyright (c) Meta Platforms, Inc. and affiliates, MIT License.
"""Encoder architecture."""

import math

import torch
import torch.nn as nn

from toyjepa.masks import full_ids, gather_toks_by_ids
from toyjepa.models.positional_embeddings import RoPE3d
from toyjepa.models.transformer import PatchEmbed, TransformerBlock, TubeletEmbed


class ViTEncoder(nn.Module):
    def __init__(
        self,
        ref_img_size: tuple[int, int],
        patch_size: int,
        tubelet_size: int,
        depth: int,
        in_channels: int,
        d_embed: int,
        num_heads: int,
        num_registers: int,
        mlp_ratio: float,
        qkv_bias: bool,
        qk_norm: bool,
        gated_attn: bool,
        peri_norm: bool,
        use_rope: bool,
        interpolate_rope: bool,
        drop_path_rate: float,
        p_drop: float,
        p_drop_attn: float,
        hierarchical_layers: list[int],
        act_layer=nn.SiLU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.d_embed = self.d_model = d_embed
        self.vid_embed = TubeletEmbed(patch_size, tubelet_size, in_channels, self.d_embed)
        self.img_embed = PatchEmbed(patch_size, in_channels, self.d_embed)
        self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, d_embed))
        self.vid_mod_embed = nn.Parameter(torch.zeros(1, 1, d_embed))
        self.head_dim = self.d_model // num_heads
        if num_registers > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_registers, d_embed))
        else:
            self.register_tokens = None

        ref_img_h, ref_img_w = ref_img_size
        self.patch_h = self.patch_w = patch_size
        self.tubelet_size = tubelet_size
        self.ref_grid_shape = (ref_img_h // self.patch_h, ref_img_w // self.patch_w)
        self.use_rope = use_rope
        # We still use 3D rope for images, but with Tp=1, whereas Tp=T//tubelet_size for videos
        # Reason:
        # to avoid mismatch shared encoder block heads must still have 1/3 of dimensions associated with temporal dim
        self.rope = (
            RoPE3d(self.head_dim, self.ref_grid_shape, num_registers, interpolate=interpolate_rope)
            if self.use_rope
            else None
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[i],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    p_drop_attn=p_drop_attn,
                    p_drop=p_drop,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    gated_attn=gated_attn,
                    peri_norm=peri_norm,
                    rope=self.rope,
                )
                for i in range(depth)
            ]
        )
        self.hierarchical_layers = hierarchical_layers
        n_out_layers = len(self.hierarchical_layers)
        self.hierarchical_norms = nn.ModuleList([norm_layer(self.d_embed) for _ in range(n_out_layers)])

        # initialize and normalize weights
        self.apply(self._init_weights)
        nn.init.normal_(self.img_mod_embed, std=1e-6)
        nn.init.normal_(self.vid_mod_embed, std=1e-6)
        if self.register_tokens is not None:
            nn.init.trunc_normal_(self.register_tokens, std=0.02)
        self._rescale_blocks()

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.to_out.weight.data, layer_id + 1)
            rescale(layer.ffn.output_proj.weight.data, layer_id + 1)

    def forward(self, x, ids=None):
        B, _, T, H, W = x.size()
        # After embedding shape is (B, N, d_embed) and modality emb (1, 1, d_embed) broadcasts over tensor
        Hp = H // self.patch_h
        Wp = W // self.patch_w
        if T == 1:
            x = torch.squeeze(x, 2)
            x = self.img_embed(x)
            x = x + self.img_mod_embed
            Tp = 1
        else:
            x = self.vid_embed(x)
            x = x + self.vid_mod_embed
            Tp = T // self.tubelet_size
        if ids is not None:
            x = gather_toks_by_ids(x, ids)
        else:
            ids = full_ids(B, Tp * Hp * Wp, x.device)
        sin_cos = None
        if self.use_rope:
            sin_cos = self.rope.precompute_rotary_embeddings(ids, Hp, Wp)

        if self.register_tokens is not None:
            regs = self.register_tokens.expand(B, -1, -1)
            x = torch.cat([x, regs], dim=1)
        out = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, sin_cos)
            if i in self.hierarchical_layers:
                pos = self.hierarchical_layers.index(i)
                out.append(self.hierarchical_norms[pos](x))
        return torch.cat(out, dim=-1)
