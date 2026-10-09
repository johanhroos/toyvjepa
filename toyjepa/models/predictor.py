# Adapted in part from V-JEPA 2 (https://github.com/facebookresearch/vjepa2),
# Copyright (c) Meta Platforms, Inc. and affiliates, MIT License.
import math

import torch
import torch.nn as nn

from toyjepa.masks import gather_toks_by_ids
from toyjepa.models.positional_embeddings import RoPE3d
from toyjepa.models.transformer import TransformerBlock


class ViTPredictor(nn.Module):
    def __init__(
        self,
        ref_img_size: tuple[int, int],
        patch_size: int,
        depth: int,
        d_embed_enc: int,
        d_embed_pred: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        qk_norm: bool,
        gated_attn: bool,
        peri_norm: bool,
        use_rope: bool,
        interpolate_rope: bool,
        num_enc_registers: int,
        p_drop: float,
        p_drop_attn: float,
        hierarchical_layers: list[int],
        num_masking_policies: int,
        act_layer=nn.SiLU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.ref_h, self.ref_w = ref_img_size
        self.patch_size_h = self.patch_size_w = patch_size
        self.ref_grid = (self.ref_h // self.patch_size_h, self.ref_w // self.patch_size_w)
        self.d_embed_pred = d_embed_pred
        self.d_embed_enc = d_embed_enc
        self.num_enc_reg_toks = num_enc_registers
        self.vid_mod_emb = nn.Parameter(torch.zeros(1, 1, self.d_embed_pred))
        self.img_mod_emb = nn.Parameter(torch.zeros(1, 1, self.d_embed_pred))
        self.hierarchical_layers = hierarchical_layers
        self.n_hierarchical = len(self.hierarchical_layers)
        if self.n_hierarchical == 1:
            self.embed_proj = nn.Linear(d_embed_enc, d_embed_pred)
        else:
            self.embed_proj = nn.Sequential(
                nn.Linear(self.n_hierarchical * d_embed_enc, d_embed_enc),
                act_layer(),
                nn.Linear(d_embed_enc, self.d_embed_pred),
            )
        self.head_dim = self.d_embed_pred // num_heads
        self.num_masking_policies = num_masking_policies
        self.learned_mask_embeds = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1, self.d_embed_pred)) for _ in range(self.num_masking_policies)]
        )
        self.rope = (
            RoPE3d(self.head_dim, ref_grid_shape=self.ref_grid, interpolate=interpolate_rope) if use_rope else None
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_embed_pred,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=0.0,
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
                for _ in range(depth)
            ]
        )
        self.pred_norm = norm_layer(self.d_embed_pred)
        self.hierarchical_ctxt_proj = nn.Linear(self.d_embed_pred, self.n_hierarchical * self.d_embed_enc)
        self.hierarchical_pred_proj = nn.Linear(self.d_embed_pred, self.n_hierarchical * self.d_embed_enc)
        # initialize and normalize weights
        self.apply(self._init_weights)
        nn.init.normal_(self.img_mod_emb, std=1e-6)
        nn.init.normal_(self.vid_mod_emb, std=1e-6)
        for i in range(self.num_masking_policies):
            nn.init.normal_(self.learned_mask_embeds[i], std=1e-6)
        self._rescale_blocks()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
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

    def forward(self, x, mask_policy, context_ids, target_ids, grid_size):
        B, n_toks_with_regs, _ = x.size()
        n_ctxt_toks = n_toks_with_regs - self.num_enc_reg_toks
        x = x[..., :n_ctxt_toks, :]
        x = self.embed_proj(x)

        Tp, Hp, Wp = grid_size
        total_toks = Tp * Hp * Wp
        mask_toks = self.learned_mask_embeds[mask_policy].expand(B, total_toks, -1)
        mask_toks = gather_toks_by_ids(mask_toks, target_ids)
        ids = torch.cat([context_ids, target_ids], dim=1)
        argsort = torch.argsort(ids, dim=1)  # Tensor of shape (B, N)
        ids = torch.gather(ids, dim=1, index=argsort)

        x = torch.cat([x, mask_toks], dim=1)  # Tensor of shape (B, N, D)
        x = gather_toks_by_ids(x, argsort)
        x = x + self.img_mod_emb if Tp == 1 else x + self.vid_mod_emb

        sin_cos = None
        if self.rope is not None:
            sin_cos = self.rope.precompute_rotary_embeddings(ids, Hp, Wp)
        for blk in self.blocks:
            x = blk(x, sin_cos)
        x = self.pred_norm(x)
        reverse_argsort = torch.argsort(argsort, dim=1)
        x = gather_toks_by_ids(x, reverse_argsort)
        x_ctxt = self.hierarchical_ctxt_proj(x[:, :n_ctxt_toks, :])
        x_pred = self.hierarchical_pred_proj(x[:, n_ctxt_toks:, :])
        return x_ctxt, x_pred
