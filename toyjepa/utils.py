import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from toyjepa.config import OptimConfig
from toyjepa.models.transformer import HeadGate


@dataclass(frozen=True)
class ParameterCount:
    trainable: int
    frozen: int
    total: int


def count_parameters(module: nn.Module) -> ParameterCount:
    """Count trainable and frozen parameters in a module."""
    trainable = 0
    frozen = 0
    for param in module.parameters():
        if param.requires_grad:
            trainable += param.numel()
        else:
            frozen += param.numel()
    return ParameterCount(trainable=trainable, frozen=frozen, total=trainable + frozen)


def num_matmul_params(module: nn.Module) -> int:
    return sum(m.weight.numel() for m in module.modules() if isinstance(m, (nn.Linear, HeadGate)))


def estimate_flops_per_token(
    module: nn.Module,
    sequence_length: int,
    num_attn_blocks: int,
    d_model: int,
    training: bool = True,
) -> int:
    """Computes FLOPs per token according to typical formulas and approximations.

    Note that this slightly overestimates the flops for predictor architecture.
    This is because a few layers only process context or target tokens.
    """
    matmul_params = num_matmul_params(module)
    if training:
        linear_flops = 6 * matmul_params
        attn_flops = 12 * d_model * sequence_length * num_attn_blocks
    else:
        linear_flops = 2 * matmul_params
        attn_flops = 4 * d_model * sequence_length * num_attn_blocks
    return linear_flops + attn_flops


def estimate_newton_schulz_flops_per_step(muon_optimizer: torch.optim.Muon | None) -> int:
    """Newton-Schulz FLOPs per Muon step: X @ X.T, A @ A and B @ X on each matrix, ns_steps times."""
    if muon_optimizer is None:
        return 0
    flops = 0
    for group in muon_optimizer.param_groups:
        for param in group["params"]:
            short, long = sorted(param.shape)
            flops += group["ns_steps"] * (4 * short**2 * long + 2 * short**3)
    return flops


def separate_grid_positions(ids, height_patches: int, width_patches: int):
    """Map frame-major flat token ids to separate T, H, and W grid positions."""
    toks_per_frame = height_patches * width_patches
    t_ids = ids // toks_per_frame
    rem = ids - t_ids * toks_per_frame
    h_ids = rem // width_patches
    w_ids = rem - h_ids * width_patches
    return t_ids, h_ids, w_ids


def init_optim(
    encoder: nn.Module,
    predictor: nn.Module,
    cfg: OptimConfig,
) -> tuple[torch.optim.AdamW, torch.optim.Muon | None]:
    if cfg.name not in ("adamw", "muon"):
        raise ValueError(f"Unknown optimizer {cfg.name!r}; expected 'adamw' or 'muon'")
    muon_params = select_muon_params(encoder, predictor) if cfg.name == "muon" else []
    muon_param_ids = {id(param) for param in muon_params}
    encoder_decay, encoder_no_decay = split_weight_decay_params(encoder, exclude=muon_param_ids)
    predictor_decay, predictor_no_decay = split_weight_decay_params(predictor, exclude=muon_param_ids)
    param_groups = []
    if encoder_decay:
        param_groups.append(
            {
                "params": encoder_decay,
                "lr": cfg.start_lr,
                "weight_decay": cfg.wd,
                "WD_exclude": False,
                "name": "encoder_decay",
            }
        )
    if predictor_decay:
        param_groups.append(
            {
                "params": predictor_decay,
                "lr": cfg.start_lr,
                "weight_decay": cfg.wd,
                "WD_exclude": False,
                "name": "predictor_decay",
            }
        )
    if encoder_no_decay:
        param_groups.append(
            {
                "params": encoder_no_decay,
                "lr": cfg.start_lr,
                "weight_decay": 0.0,
                "WD_exclude": True,
                "name": "encoder_no_decay",
            }
        )
    if predictor_no_decay:
        param_groups.append(
            {
                "params": predictor_no_decay,
                "lr": cfg.start_lr,
                "weight_decay": 0.0,
                "WD_exclude": True,
                "name": "predictor_no_decay",
            }
        )
    adamw_optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.start_lr,
        betas=tuple(cfg.betas),
        eps=cfg.eps,
        weight_decay=0.0,
    )
    muon_optimizer = None
    if muon_params:
        muon_optimizer = torch.optim.Muon(
            muon_params,
            lr=cfg.start_lr,
            weight_decay=cfg.wd,
            adjust_lr_fn="match_rms_adamw",
        )
    return adamw_optimizer, muon_optimizer


def select_muon_params(encoder: nn.Module, predictor: nn.Module) -> list[nn.Parameter]:
    """Select matrix weights from student and predictor transformer blocks."""
    return [
        param
        for model in (encoder, predictor)
        for block in model.blocks
        for param in block.parameters()
        if param.requires_grad and param.ndim == 2
    ]


def split_weight_decay_params(
    module: nn.Module,
    exclude: set[int] | None = None,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split params using the V-JEPA rule: no WD for bias or 1D parameters."""
    exclude = set() if exclude is None else exclude
    decay = []
    no_decay = []
    for name, param in module.named_parameters():
        if not param.requires_grad or id(param) in exclude:
            continue
        if "bias" in name or param.ndim == 1:
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def update_ema(student_params, teacher_params, m: float) -> None:
    with torch.no_grad():
        alpha = 1 - m
        student = []
        teacher = []
        for param_student, param_teacher in zip(student_params, teacher_params, strict=True):
            student.append(param_student)
            teacher.append(param_teacher)
        torch._foreach_mul_(teacher, m)
        torch._foreach_add_(teacher, student, alpha=alpha)


def linear_schedule(step: int, total_steps: int, start: float, end: float) -> float:
    alpha = step / total_steps
    return start + alpha * (end - start)


def cosine_schedule(
    step: int,
    total_steps: int,
    cooldown_steps: int,
    start: float,
    end: float,
    warmup_steps: int = 0,
    warmup_start: float | None = None,
) -> float:
    """Apply cosine cooldown and linear warmup."""
    if warmup_steps > 0 and step < warmup_steps:
        warmup_start = start if warmup_start is None else warmup_start
        return linear_schedule(step=step, total_steps=warmup_steps, start=warmup_start, end=start)
    if step <= total_steps - cooldown_steps:
        return start

    constant_steps = total_steps - warmup_steps - cooldown_steps
    cosine_step = step - warmup_steps - constant_steps
    alpha = cosine_step / cooldown_steps
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * alpha))


def schedule_phase_steps(optim_cfg: OptimConfig, total_steps: int) -> tuple[int, int]:
    """Integer warmup and cooldown lengths over the planned optimizer-step horizon."""
    return int(total_steps * optim_cfg.warmup_ratio), int(total_steps * optim_cfg.cooldown_ratio)


def is_stable_lr_step(optim_cfg: OptimConfig, *, step: int, total_steps: int) -> bool:
    """Whether update `step` is past warmup and before cooldown, i.e. ran at peak LR."""
    warmup_steps, cooldown_steps = schedule_phase_steps(optim_cfg, total_steps)
    return step > 0 and warmup_steps <= step <= total_steps - cooldown_steps


def apply_optim_schedules(
    optimizers: list[torch.optim.Optimizer],
    optim_cfg: OptimConfig,
    *,
    step: int,
    total_steps: int,
) -> float:
    warmup_steps, cooldown_steps = schedule_phase_steps(optim_cfg, total_steps)
    lr = cosine_schedule(
        step=step,
        total_steps=total_steps,
        cooldown_steps=cooldown_steps,
        start=optim_cfg.lr,
        end=optim_cfg.final_lr,
        warmup_steps=warmup_steps,
        warmup_start=optim_cfg.start_lr,
    )

    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)
    return lr


def save_checkpoint(
    path: str | Path,
    *,
    student_encoder: nn.Module,
    teacher_encoder: nn.Module,
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer,
    muon_optimizer: torch.optim.Muon | None,
    completed_epochs: int,
    global_batch: int,
    optimizer_step: int,
    cumulative_estimated_flops: int,
    cumulative_train_seconds: float,
    cfg: Any,
    wandb_id: str | None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "student_encoder": student_encoder.state_dict(),
        "teacher_encoder": teacher_encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "muon_optimizer": muon_optimizer.state_dict() if muon_optimizer is not None else None,
        "completed_epochs": completed_epochs,
        "global_batch": global_batch,
        "optimizer_step": optimizer_step,
        "cumulative_estimated_flops": cumulative_estimated_flops,
        "cumulative_train_seconds": cumulative_train_seconds,
        "cfg": OmegaConf.to_container(cfg, resolve=True, enum_to_str=True),
        "wandb_id": wandb_id,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(
    path: str | Path,
    *,
    student_encoder: nn.Module,
    teacher_encoder: nn.Module,
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    muon_optimizer: torch.optim.Muon | None = None,
    map_location: str | torch.device = "cuda",
) -> dict[str, Any]:
    """Load ToyJEPA module and optimizer state, returning training metadata."""
    checkpoint = torch.load(Path(path), map_location=map_location)
    student_encoder.load_state_dict(checkpoint["student_encoder"])
    teacher_encoder.load_state_dict(checkpoint["teacher_encoder"])
    predictor.load_state_dict(checkpoint["predictor"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if muon_optimizer is not None:
        if checkpoint.get("muon_optimizer") is None:
            raise ValueError("No muon optimizer found in checkpoint!")
        muon_optimizer.load_state_dict(checkpoint["muon_optimizer"])
    return {
        "epoch": checkpoint["completed_epochs"],
        "global_batch": checkpoint["global_batch"],
        "optim_step": checkpoint["optimizer_step"],
        "cumulative_estimated_flops": checkpoint.get("cumulative_estimated_flops", 0),
        "cumulative_train_seconds": checkpoint.get("cumulative_train_seconds", 0.0),
        "cfg": checkpoint.get("cfg"),
        "wandb_id": checkpoint.get("wandb_id"),
    }


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class Metrics:
    loss: float
    target_loss: float
    context_loss: float
    max_loss: float | None = None
    tflops_per_s: float | None = None
    wall_dt: float | None = None
    toks_per_s: float | None = None
    time_elapsed: str | None = None
    eta: str | None = None


@dataclass
class Meter:
    ema_beta: float = 0.98
    window_count: int = 0
    ema_dt: float | None = None
    interval: dict[str, float] = field(default_factory=dict)
    max_loss: float = 0.0

    def update(
        self,
        loss: float,
        target_loss: float,
        context_loss: float,
        max_loss: float,
        token_count: int,
        wall_dt: float,
        estimated_flops: int,
    ) -> None:
        self.window_count += 1
        self.max_loss = max(self.max_loss, max_loss)

        values_to_upd = {
            "loss": loss,
            "target_loss": target_loss,
            "context_loss": context_loss,
        }
        if self.ema_dt is None:
            self.ema_dt = wall_dt
        else:
            self.ema_dt = self.ema_beta * self.ema_dt + (1 - self.ema_beta) * wall_dt
        values_to_upd["wall_dt"] = wall_dt
        values_to_upd["token_count"] = token_count
        values_to_upd["estimated_flops"] = estimated_flops

        for k, v in values_to_upd.items():
            self.interval[k] = self.interval.get(k, 0.0) + float(v)

    def summary(
        self,
        step_idx: int,
        total_steps: int,
        elapsed_s: float,
    ) -> Metrics:
        if self.window_count == 0:
            raise ValueError("Cannot summarize Meter before any values have been recorded")
        loss = self.interval["loss"] / self.window_count
        target_loss = self.interval["target_loss"] / self.window_count
        context_loss = self.interval["context_loss"] / self.window_count
        wall_dt = self.interval.get("wall_dt")
        token_count = self.interval.get("token_count")
        estimated_flops = self.interval.get("estimated_flops")

        eta = None
        if self.ema_dt is not None:
            remaining_steps = max(0, total_steps - step_idx)
            eta = remaining_steps * self.ema_dt

        return Metrics(
            loss=loss,
            target_loss=target_loss,
            context_loss=context_loss,
            max_loss=self.max_loss,
            wall_dt=wall_dt / self.window_count,
            toks_per_s=token_count / wall_dt,
            time_elapsed=format_seconds(elapsed_s),
            eta=None if eta is None else format_seconds(eta),
            tflops_per_s=estimated_flops / (wall_dt * 1e12),
        )

    def reset_window(self) -> None:
        self.interval.clear()
        self.window_count = 0
        self.max_loss = 0.0
