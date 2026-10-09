"""Functions to help with masking."""

import math

import torch


def gather_toks_by_ids(x, ids):
    """Gather tokens from specified positions.

    Args:
        x: Tensor of shape (B, N, D).
        ids: Integer tensor of shape (B, K) with original token positions to keep.

    Returns:
        Tensor of shape (B, K, D).
    """
    ids = ids.to(device=x.device, dtype=torch.long)
    ids = ids.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    return torch.gather(x, dim=1, index=ids)


def full_ids(batch_size, total_tokens, device):
    return torch.arange(int(total_tokens), device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)


def sample_mask_tok_ids(batch_size, tokenized_vid_dims, mask_policy, device, aspect_ratio_range=(0.75, 1.5)):
    Tp, Hp, Wp = tokenized_vid_dims
    ar_min, ar_max = aspect_ratio_range
    min_visible_context = 0.09

    if mask_policy == 0:
        num_blocks = 8
        spatial_coverage = 0.15
    elif mask_policy == 1:
        num_blocks = 2
        spatial_coverage = 0.7
    else:
        raise ValueError("No such masking policy!")

    visible_context = 0.0
    while visible_context < min_visible_context:
        pred_mask = torch.zeros((Tp, Hp, Wp), dtype=torch.bool, device=device)
        for _ in range(num_blocks):
            ar = ar_min + torch.rand((), device=device).item() * (ar_max - ar_min)

            area = max(1, round(spatial_coverage * Hp * Wp))
            h = max(1, min(Hp, round(math.sqrt(area * ar))))
            w = max(1, min(Wp, round(math.sqrt(area / ar))))

            h0 = torch.randint(0, Hp - h + 1, ()).item()
            w0 = torch.randint(0, Wp - w + 1, ()).item()

            pred_mask[:, h0 : h0 + h, w0 : w0 + w] = True
        visible_context = (~pred_mask).sum().item() / math.prod(tokenized_vid_dims)

    flat = pred_mask.flatten()
    target_ids = torch.nonzero(flat, as_tuple=False).flatten()
    context_ids = torch.nonzero(~flat, as_tuple=False).flatten()

    context_ids = context_ids.unsqueeze(0).expand(batch_size, -1).contiguous()
    target_ids = target_ids.unsqueeze(0).expand(batch_size, -1).contiguous()
    return context_ids, target_ids
