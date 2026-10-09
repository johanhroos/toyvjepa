"""Building blocks for model architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath


class PatchEmbed(nn.Module):
    """Image to patch embedding."""

    def __init__(self, patch_size, in_channels=3, d_embed=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=d_embed, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return rearrange(self.proj(x), "B C H W -> B (H W) C")


class TubeletEmbed(nn.Module):
    """Video to patch embedding."""

    def __init__(self, patch_size=16, tubelet_size=2, in_channels=3, d_embed=768):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=d_embed,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x):
        return rearrange(self.proj(x), "B C T H W -> B (T H W) C")


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer, p_drop=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(p_drop),
        )

    @property
    def output_proj(self):
        return self.layers[3]

    def forward(self, x):
        return self.layers(x)


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, wide_silu=True):
        super().__init__()
        swiglu_hidden_features = hidden_features
        if wide_silu:
            swiglu_hidden_features = int(2 / 3 * hidden_features)
            swiglu_hidden_features = (
                (swiglu_hidden_features + 7) // 8 * 8
            )  # make hidden dim divisible by 8 for efficient matmuls
        self.proj = nn.Linear(in_features, swiglu_hidden_features)
        self.gate = nn.Linear(in_features, swiglu_hidden_features)
        self.fc = nn.Linear(swiglu_hidden_features, out_features)

    @property
    def output_proj(self):
        return self.fc

    def forward(self, x):
        x1 = self.proj(x)
        x2 = self.gate(x)
        x = x1 * F.silu(x2)
        return self.fc(x)


class HeadGate(nn.Module):
    """Head-specific sigmoid gate on the attention output, computed from the attention input.

    The G1 gate of Qiu et al. 2025 (arXiv 2505.06708), with one score per head and token. It is
    zero-initialized, so every gate starts at 0.5 and creating it draws no random numbers.
    Return shape: (B N H).
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_heads, d_model))
        self.bias = nn.Parameter(torch.zeros(num_heads))

    def forward(self, x):
        return torch.sigmoid(F.linear(x, self.weight, self.bias))


class SelfAttention(nn.Module):
    """Multi-head self-attention with optional QK-norm, output gate and RoPE on patch tokens.

    Return shape: (B N d_model).
    """

    def __init__(
        self,
        d_model,
        num_heads,
        p_drop_attn=0.0,
        p_drop=0.0,
        qkv_bias=False,
        qk_norm=False,
        gated_attn=False,
        norm_layer=nn.LayerNorm,
        rope=None,
    ):
        super().__init__()
        self.p_drop_attn = p_drop_attn
        self.num_heads = num_heads
        self.rope = rope
        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        head_dim = d_model // num_heads
        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.gate = HeadGate(d_model, num_heads) if gated_attn else None
        self.to_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p_drop)

    def project_qkv(self, x, pe=None):
        """Return q, k, v shaped (B H N d), with QK-norm and RoPE applied to q and k."""
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B N (vec H d) -> vec B H N d", vec=3, H=self.num_heads)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            assert pe is not None, "No precomputed positional embeddings given"
            q, k = self.rope(q, k, pe)
        return q, k, v

    def forward(self, x, pe=None):
        dropout_p = self.p_drop_attn if self.training else 0.0
        q, k, v = self.project_qkv(x, pe)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        if self.gate is not None:
            attn = attn * rearrange(self.gate(x), "B N H -> B H N 1")
        x = rearrange(attn, "B H N d -> B N (H d)")
        x = self.to_out(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """A single pre-norm self-attention block, optionally with peri-norm.

    With `peri_norm`, the attention and MLP outputs are also normalized before they are added to the
    residual stream (Peri-LN, Kim et al. 2025; Sandwich-LN in CogView, Ding et al. 2021), so one
    branch write is bounded by its norm's gain.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        act_layer=nn.SiLU,
        norm_layer=nn.LayerNorm,
        p_drop_attn=0.0,
        p_drop=0.0,
        qkv_bias=False,
        qk_norm=False,
        gated_attn=False,
        peri_norm=False,
        rope=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(d_model)
        self.attn = SelfAttention(
            d_model,
            num_heads,
            p_drop_attn=p_drop_attn,
            p_drop=p_drop,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            gated_attn=gated_attn,
            norm_layer=norm_layer,
            rope=rope,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(d_model)
        mlp_hidden = int(mlp_ratio * d_model)
        if act_layer is nn.SiLU:
            self.ffn = SwiGLUFFN(d_model, mlp_hidden, d_model)
        else:
            self.ffn = FFN(d_model, mlp_hidden, d_model, act_layer, p_drop)
        self.attn_out_norm = norm_layer(d_model) if peri_norm else nn.Identity()
        self.ffn_out_norm = norm_layer(d_model) if peri_norm else nn.Identity()

    def forward(self, x, pe=None):
        y = self.attn_out_norm(self.attn(self.norm1(x), pe))
        x = x + self.drop_path(y)
        return x + self.drop_path(self.ffn_out_norm(self.ffn(self.norm2(x))))
