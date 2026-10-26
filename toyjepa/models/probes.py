"""Lightweight probes for frozen ToyJEPA representations."""

import math

import torch
import torch.nn as nn
from einops import rearrange

from toyjepa.models.transformer import CrossAttentionBlock, TransformerBlock


class MeanPoolLinearProbe(nn.Module):
    """Mean-pool patch tokens and classify a frozen encoder representation.

    ToyJEPA appends register tokens after patch tokens. They are deliberately
    excluded from the mean so variable-length videos remain comparable.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        num_registers: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        if num_registers < 0:
            raise ValueError("num_registers must be non-negative")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_registers = num_registers
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(input_dim, num_classes)

    def pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return one mean-pooled patch representation per sample."""
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens with shape B,N,D, got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.input_dim:
            raise ValueError(f"Expected token dimension {self.input_dim}, got {tokens.shape[-1]}")

        num_patch_tokens = tokens.shape[1] - self.num_registers
        if num_patch_tokens <= 0:
            raise ValueError(
                f"Expected at least one patch token before {self.num_registers} register tokens, "
                f"got sequence length {tokens.shape[1]}"
            )
        return tokens[:, :num_patch_tokens].mean(dim=1)

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        """Classify already-pooled features, as used by the feature cache."""
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(f"Expected pooled features with shape B,{self.input_dim}, got {tuple(features.shape)}")
        return self.head(self.dropout(features))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_features(self.pool_tokens(tokens))


class AttentiveProbe(nn.Module):
    """Attentive probe on frozen hierarchical tokens, following the V-JEPA 2.1 attentive classifier.

    V-JEPA 2.1 (Mur-Labadia et al. 2026, arXiv 2603.14482) probes with `depth - 1` self-attention
    blocks over the frozen tokens, then one cross-attention block in which a learnable query token
    attends to them, then a linear classifier. There is no positional embedding. For Diving-48 it
    attends to the tokens of four encoder layers side by side.

    ToyJEPA's encoder concatenates its hierarchical layers along channels. With `levels_as_tokens`,
    as in V-JEPA 2.1, the probe splits them back and lines the levels up along the token axis:
    B,K+R,L*D -> B,L*K,D. Otherwise a linear layer projects the concatenated levels to D, the levels
    the SSv2 linear probe sees: B,K+R,L*D -> B,K,D. That sequence is L times shorter, so the
    self-attention costs about L^2 times less. Register tokens are dropped first, as in
    MeanPoolLinearProbe.

    `qk_norm` adds QK-norm to every self- and cross-attention, which V-JEPA 2.1's probe doesn't have.
    The norms draw no random numbers, so a given seed initializes every other weight as without them.
    """

    def __init__(
        self,
        embed_dim: int,
        num_levels: int,
        num_classes: int,
        *,
        num_registers: int = 0,
        depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        levels_as_tokens: bool = True,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least one (the cross-attention block)")
        if num_registers < 0:
            raise ValueError("num_registers must be non-negative")

        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.num_registers = num_registers
        self.init_std = init_std
        self.level_proj = None if levels_as_tokens else nn.Linear(num_levels * embed_dim, embed_dim)
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim, num_heads, mlp_ratio=mlp_ratio, act_layer=nn.GELU, qkv_bias=qkv_bias, qk_norm=qk_norm
                )
                for _ in range(depth - 1)
            ]
        )
        self.cross_block = CrossAttentionBlock(
            embed_dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm
        )
        nn.init.trunc_normal_(self.query, std=init_std)
        if self.level_proj is not None:
            self.level_proj.apply(self._init_weights)
        self.blocks.apply(self._init_weights)
        self.cross_block.apply(self._init_weights)
        self._rescale_blocks()
        # created after the init, so it keeps PyTorch's default init as in V-JEPA 2.1
        self.head = nn.Linear(embed_dim, num_classes)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        layer_id = 0
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.to_out.weight.data, layer_id + 1)
            rescale(layer.ffn.output_proj.weight.data, layer_id + 1)
        # as in V-JEPA 2.1, the cross block shares the last self-attention block's divisor
        rescale(self.cross_block.ffn.output_proj.weight.data, layer_id + 1)

    def prepare_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Drop registers, then line the levels up as tokens (B,L*K,D) or project them to D (B,K,D)."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.num_levels * self.embed_dim:
            raise ValueError(
                f"Expected tokens with shape B,N,{self.num_levels * self.embed_dim}, got {tuple(tokens.shape)}"
            )
        num_patch_tokens = tokens.shape[1] - self.num_registers
        if num_patch_tokens <= 0:
            raise ValueError(
                f"Expected at least one patch token before {self.num_registers} register tokens, "
                f"got sequence length {tokens.shape[1]}"
            )
        patches = tokens[:, :num_patch_tokens]
        if self.level_proj is None:
            return rearrange(patches, "B K (L D) -> B (L K) D", L=self.num_levels)
        return self.level_proj(patches)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.prepare_tokens(tokens)
        for block in self.blocks:
            x = block(x)
        query = self.cross_block(self.query.expand(len(x), -1, -1), x)
        return self.head(query.squeeze(1))
