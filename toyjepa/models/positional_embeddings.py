import math

import torch
import torch.nn as nn
from einops import rearrange

from toyjepa.utils import separate_grid_positions


class RoPE3d(nn.Module):
    """3D axial Rotary Position Embedding for spatiotemporal (video) transformers.

    Splits each head's feature dimension into three equal, even-sized segments —
    one each for the temporal (T), height (H), and width (W) axes — and applies an
    independent 1D rotary rotation to each, following the 3D-RoPE scheme used in
    V-JEPA 2. Any leftover dims (when head_dim is not divisible by 6) pass through
    unrotated.

    Patch tokens are assumed to come first, with flat indices encoding positions in
    a frame-major, row-major (T, H, W) grid. Optional register tokens may be appended
    after the patch tokens; they are passed through without rotary embeddings.
    `ids` contains positions for patch tokens only and is expected to be an integer
    tensor.

    Usage: the encoder calls `precompute_rotary_embeddings(ids, Hp, Wp)` once per
    forward to build the per-axis sin/cos tables, then passes the cached tables to
    `forward(q, k, pe)` in each attention block to rotate queries and keys.

    With `interpolate=True`, the spatial (H, W) coordinates are endpoint-aligned
    onto the pretrained grid so the rotary phases stay within the range seen during
    pretraining when running at a different spatial resolution; the temporal axis is
    left unscaled.

    Args:
        head_dim:      per-head feature dimension. Three segments of
                       2 * ((head_dim // 3) // 2) are rotated; the rest pass through.
        ref_grid_shape: (H, W) patch grid the weights were trained at, used as the
                       reference for spatial interpolation.
        base:          rotary frequency base (theta).
        interpolate:   if True, endpoint-align spatial positions onto pretrain_grid.
        dtype:         dtype of the rotary tables; rotation is computed in this dtype
                       and cast back to the input dtype.
        n_registers:   number of unpositioned register tokens appended to the sequence.
    """

    def __init__(
        self,
        head_dim,
        ref_grid_shape=(16, 16),
        n_registers=0,
        base=10000.0,
        interpolate=False,
        dtype=torch.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.interpolate = interpolate
        self.n_registers = n_registers
        self.ref_grid_h, self.ref_grid_w = ref_grid_shape
        d_axis = int(2 * ((head_dim // 3) // 2))  # each axis gets equal number of dims, which must be an even number
        self.t_dim = self.h_dim = self.w_dim = d_axis
        self.remainder = head_dim - 3 * d_axis  # Leftover dims pass through unrotated

        i = torch.arange(d_axis // 2, dtype=dtype)
        self.register_buffer("theta", torch.exp(-2.0 * i / d_axis * math.log(base)), persistent=False)

    def _build_table(self, pos):
        """Precompute rotary embedding tables for a set of grid positions.

        pos: (B, N) tensor of positions along one axis (T, H or W) of video latent

        Returns emb_sin, emb_cos of shape (B, N, d_axis)
        """
        freq = pos.to(self.dtype).unsqueeze(-1) * self.theta  # (..., 1) * (d_axis // 2) -> (..., d_axis // 2)
        emb_sin = freq.sin().repeat_interleave(2, dim=-1)  # (..., d_axis)
        emb_cos = freq.cos().repeat_interleave(2, dim=-1)
        return emb_sin, emb_cos

    def precompute_rotary_embeddings(self, ids, Hp, Wp):
        """Called from forward method of the underlying neural network to apply RoPE across all layers.

        ids: mapped positions of tokens after masking
        Hp, Wp: number of height and width patches

        Returns a list of (emb_sin, emb_cos) tuples along each (T, H, W) axis.
        """
        t_pos, h_pos, w_pos = separate_grid_positions(ids, Hp, Wp)
        if self.interpolate:  # Adjustment so that RoPE doesn't extrapolate
            h_pos = h_pos * (self.ref_grid_h - 1) / (Hp - 1)
            w_pos = w_pos * (self.ref_grid_w - 1) / (Wp - 1)
        cache = []
        for pos in (t_pos, h_pos, w_pos):
            emb_sin, emb_cos = self._build_table(pos)
            cache.append((emb_sin, emb_cos))
        return cache

    def forward(self, q, k, pe):
        """Called from a RoPE attention block to rotate vectors.

        q: query vector (B, num_heads, N + n_registers, D)
        k: key vector (B, num_heads, N + n_registers, D)
        pe: List of precomputed positional embeddings for (T, H, W) axes
        pe_sin/pe_cos: (B, N, d_axis); broadcasts over batch/heads

        Returns rotated q and k with the appended register tokens unchanged.
        """

        def rotate(x):
            context_end = x.shape[-2] - self.n_registers
            x_context = x[..., :context_end, :]
            x_registers = x[..., context_end:, :] if self.n_registers else None
            out = []
            offset = 0
            for (pe_sin, pe_cos), d in zip(pe, (self.t_dim, self.h_dim, self.w_dim), strict=True):
                xi = x_context[..., offset : offset + d]
                y = rearrange(xi, "B num_heads N (d1 d2) -> B num_heads N d1 d2", d2=2)
                y1, y2 = y.unbind(
                    dim=-1
                )  # 2d rotation matrix requires taking (y1 * cos mtheta - y2 * sin mtheta) for first dimension
                y = torch.stack((-y2, y1), dim=-1).flatten(
                    -2
                )  # (y2 * cos mtheta + y1 * sin mtheta) for second dimension
                if pe_sin.dim() == 3:
                    pe_sin = pe_sin.unsqueeze(1)  # (B, N, d_axis) -> (B, 1, N, d_axis)
                    pe_cos = pe_cos.unsqueeze(1)  # to enable broadcasting over num_heads
                out.append((xi * pe_cos + y * pe_sin).to(x.dtype))
                offset += d
            if self.remainder:
                out.append(x_context[..., offset:])
            x_context = torch.cat(out, dim=-1)
            if x_registers is not None:
                return torch.cat((x_context, x_registers), dim=-2)
            return x_context

        return rotate(q), rotate(k)
