"""Compute loss function."""

import torch

from toyjepa.utils import separate_grid_positions


def loss_fn(
    pred_context: torch.Tensor,
    pred_target: torch.Tensor,
    teacher_context: torch.Tensor,
    teacher_target: torch.Tensor,
    lambda_context: float,
    context_ids: torch.Tensor,
    target_ids: torch.Tensor,
    grid_size: tuple[int, int, int],
    temporal_scale: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Works for both images and videos.

    context_ids: (B, K) flat token ids in frame-major, row-major order.
    target_ids:  (B, M) flat token ids in frame-major, row-major order.
    grid_size:   (Tp, Hp, Wp) token grid used to convert flat ids to coordinates.

    pred_context:     (B, K, D)
    pred_target:      (B, M, D)
    teacher_context:  (B, K, D)
    teacher_target:   (B, M, D)

    The target loss is ordinary L1 prediction loss.
    The context loss is distance-weighted L1 loss, where each context token
    is weighted by its distance to the nearest target/masked token.
    """
    assert pred_context.shape == teacher_context.shape
    assert pred_target.shape == teacher_target.shape

    assert context_ids.ndim == 2
    assert target_ids.ndim == 2

    # Stop grad for teacher output
    teacher_context = teacher_context.detach()
    teacher_target = teacher_target.detach()

    target_loss = torch.abs(pred_target - teacher_target).mean(dim=(1, 2))

    _, height_patches, width_patches = grid_size
    context_pos = torch.stack(
        separate_grid_positions(context_ids, height_patches, width_patches),
        dim=-1,
    ).to(device=pred_context.device, dtype=pred_context.dtype)
    target_pos = torch.stack(
        separate_grid_positions(target_ids, height_patches, width_patches),
        dim=-1,
    ).to(device=pred_context.device, dtype=pred_context.dtype)

    # temporal_scale controls how comparable one temporal step is to one spatial step.
    scale = torch.tensor(
        [temporal_scale, 1.0, 1.0],
        device=pred_context.device,
        dtype=pred_context.dtype,
    )
    context_pos = context_pos * scale
    target_pos = target_pos * scale

    distances = torch.cdist(context_pos, target_pos, p=2)  # Shape: (B, K, M)
    d_min = distances.min(dim=-1).values  # Shape: (B, K)
    weights = lambda_context / torch.sqrt(d_min.clamp_min(eps))
    weights = weights.unsqueeze(-1)  # (B, K, 1)

    context_loss = (weights * torch.abs(pred_context - teacher_context)).mean(dim=(1, 2))
    sample_losses = target_loss + context_loss
    target_loss = target_loss.mean()
    context_loss = context_loss.mean()
    loss = target_loss + context_loss

    return loss, target_loss.detach(), context_loss.detach(), sample_losses.detach()
