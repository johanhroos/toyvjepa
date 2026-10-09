"""Pretraining script."""

import logging
import math
import time
from dataclasses import fields
from itertools import islice
from pathlib import Path

import hydra
import torch
import torch._inductor.config as inductor_config
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from toyjepa.build import build_dataloader, build_toyjepa, seed_dataloader
from toyjepa.config import LinearProbeConfig, LossConfig, OptimConfig, PretrainConfig
from toyjepa.datasets import Batch
from toyjepa.diagnostics import GradNormTracker, forward_diagnostics, grad_norm_groups, layer_lines
from toyjepa.eval import evaluate_linear_probe
from toyjepa.loss import loss_fn
from toyjepa.masks import gather_toks_by_ids
from toyjepa.utils import (
    Meter,
    Metrics,
    apply_optim_schedules,
    count_parameters,
    estimate_flops_per_token,
    estimate_newton_schulz_flops_per_step,
    init_optim,
    is_stable_lr_step,
    load_checkpoint,
    save_checkpoint,
    update_ema,
)

DTYPE_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}
WANDB_PROJECT = "toyjepa"
FLOPS_PER_PFLOP = 1e15
FLOP_AXIS = "compute/estimated_model_pflops"
TRAIN_TIME_AXIS = "compute/train_hours"
SECONDS_PER_HOUR = 3600
# Config values that must match the checkpoint for exact resume and paired runs.
RESUME_MATCH_KEYS = (
    ("arch",),
    ("optim",),
    ("mask",),
    ("loss",),
    ("ema",),
    ("trainer", "seed"),
    ("trainer", "compile"),
    ("trainer", "epochs"),
    ("trainer", "gradient_accumulation_steps"),
    ("trainer", "train_split"),
    ("trainer", "train_shuffle"),
    ("loader", "num_workers"),
    ("loader", "token_budget"),
    ("trainer", "val_split"),
    ("trainer", "val_shuffle"),
    ("trainer", "num_val_batches"),
    *(("probe", field.name) for field in fields(LinearProbeConfig) if field.name != "log_every"),
)
RESUME_MATCH_DATA_FIELDS = ("dataset_cls", "target_fps", "quantization_interval")


def epoch_seed(seed: int, epoch: int) -> int:
    """Seed for one run-epoch; distinct for every (seed, epoch) pair with epoch < 1000."""
    return seed * 1000 + epoch


def epoch_checkpoint_name(
    completed_epochs: int, *, optim_step: int, total_optim_steps: int, optim_cfg: OptimConfig
) -> str:
    """Checkpoint file name for an epoch boundary, tagged when its last update ran at peak LR."""
    stable = is_stable_lr_step(optim_cfg, step=optim_step, total_steps=total_optim_steps)
    return f"epoch_{completed_epochs:03d}{'_stablelr' if stable else ''}.pt"


def wandb_fork_step(source_run_id: str, *, optimizer_step: int) -> int:
    """W&B `_step` of the source run's last row at or before `optimizer_step`, for `fork_from`.

    Selecting by `optimizer_step` keeps the checkpoint epoch's probe row, which is logged after
    the checkpoint is saved and carries that epoch's final `optimizer_step`.
    """
    source = wandb.Api().run(f"{WANDB_PROJECT}/{source_run_id}")
    steps = [
        row["_step"]
        for row in source.scan_history()
        if row.get("optimizer_step") is not None and row["optimizer_step"] <= optimizer_step
    ]
    if not steps:
        raise ValueError(f"W&B run {source_run_id} has no rows at or before optimizer_step {optimizer_step}")
    return max(steps)


def forward_loss(
    batch: Batch,
    *,
    student_enc: torch.nn.Module,
    teacher_enc: torch.nn.Module,
    predictor: torch.nn.Module,
    loss_cfg: LossConfig,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one ToyJEPA forward pass and compute the pretraining loss."""
    inputs = batch["inputs"].to(device)
    context_ids = batch["context_ids"]
    target_ids = batch["target_ids"]
    if context_ids is None or target_ids is None:
        raise ValueError("Enable mask generation in the collator")
    context_ids = context_ids.to(device)
    target_ids = target_ids.to(device)
    grid_size = tuple(batch["tokenized_shape"])

    # first make pixels have values in range [0,1]
    inputs = inputs.float().div_(255.0)
    # then normalize everything with mean 0.5 and std 0.5 i.e. range is then [-1, 1]
    inputs = inputs.sub_(0.5).div_(0.5)

    if inputs.shape[1] == 1:
        inputs = inputs.expand(-1, 3, -1, -1, -1)

    h_student = student_enc(inputs, ids=context_ids)
    pred_context, pred_target = predictor(
        h_student,
        mask_policy=batch["mask_policy"],
        context_ids=context_ids,
        target_ids=target_ids,
        grid_size=grid_size,
    )
    with torch.no_grad():
        h_teacher = teacher_enc(inputs)
    teacher_context = gather_toks_by_ids(h_teacher, context_ids)
    teacher_target = gather_toks_by_ids(h_teacher, target_ids)

    return loss_fn(
        pred_context=pred_context,
        pred_target=pred_target,
        teacher_context=teacher_context,
        teacher_target=teacher_target,
        lambda_context=loss_cfg.lambda_context,
        context_ids=context_ids,
        target_ids=target_ids,
        grid_size=grid_size,
        temporal_scale=loss_cfg.temporal_scale,
        eps=loss_cfg.eps,
    )


def run_validation(
    *,
    val_loader: DataLoader,
    student_enc: torch.nn.Module,
    teacher_enc: torch.nn.Module,
    predictor: torch.nn.Module,
    loss_cfg: LossConfig,
    device: torch.device,
    num_batches: int,
    precision: str,
) -> Metrics:
    """Evaluate loss on a fixed validation set."""
    batches_to_eval = min(num_batches, len(val_loader))

    student_enc.eval(), teacher_enc.eval(), predictor.eval()
    tot_val_loss, tot_val_tgt_loss, tot_val_ctxt_loss = 0.0, 0.0, 0.0

    # same batches and masks at every validation
    seed_dataloader(val_loader, 1337)
    for val_batch in tqdm(islice(val_loader, batches_to_eval), total=batches_to_eval, desc="Processing val batches"):
        with (
            torch.no_grad(),
            torch.autocast(device_type=device.type, dtype=DTYPE_MAP.get(precision, torch.bfloat16)),
        ):
            val_loss, val_target_loss, val_context_loss, _ = forward_loss(
                batch=val_batch,
                student_enc=student_enc,
                teacher_enc=teacher_enc,
                predictor=predictor,
                loss_cfg=loss_cfg,
                device=device,
            )
            tot_val_loss += val_loss.item()
            tot_val_tgt_loss += val_target_loss.item()
            tot_val_ctxt_loss += val_context_loss.item()
    student_enc.train(), teacher_enc.eval(), predictor.train()
    return Metrics(
        loss=tot_val_loss / batches_to_eval,
        target_loss=tot_val_tgt_loss / batches_to_eval,
        context_loss=tot_val_ctxt_loss / batches_to_eval,
    )


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="pretrain")
def main(cfg: PretrainConfig) -> None:
    if not cfg.mask.enabled:
        raise ValueError("ToyJEPA pretraining requires mask.enabled=true")
    if cfg.trainer.epochs >= 1000:
        raise ValueError("epoch_seed assumes fewer than 1000 epochs")
    if cfg.loader.num_workers == 0:
        raise ValueError("Seeded pretraining requires loader.num_workers > 0")
    log.info("Logging output dir: %s", HydraConfig.get().runtime.output_dir)
    log.info("Training with config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # create train loaders
    train_loader = build_dataloader(
        cfg,
        split=cfg.trainer.train_split,
        shuffle=cfg.trainer.train_shuffle,
        augment=True,
    )
    train_meter = Meter()
    num_batches = len(train_loader)
    log.info(
        "Created train loader with %d batches",
        num_batches,
    )

    # create val loader for specific dataset
    val_loader = build_dataloader(
        cfg,
        split=cfg.trainer.val_split,
        shuffle=cfg.trainer.val_shuffle,
        dataset=cfg.trainer.val_dataset,
        augment=False,
    )
    num_val_batches = len(val_loader)
    log.info(
        "Validation will use the first %d/%d batches from split=%s of %s dataset",
        min(cfg.trainer.num_val_batches, num_val_batches),
        num_val_batches,
        cfg.trainer.val_split,
        cfg.trainer.val_dataset,
    )
    # the first validation batches, with their masks, kept for the per-layer forward diagnostics
    seed_dataloader(val_loader, 1337)
    diagnostic_batches = list(islice(val_loader, cfg.trainer.diagnostic_batches))

    probe_train_loader = build_dataloader(
        cfg,
        split=cfg.probe.train_split,
        shuffle=True,
        dataset=cfg.probe.dataset,
        augment=False,
        generate_masks=False,
    )
    probe_val_loader = build_dataloader(
        cfg,
        split=cfg.probe.val_split,
        shuffle=False,
        dataset=cfg.probe.dataset,
        augment=False,
        generate_masks=False,
    )
    seed_dataloader(probe_train_loader, cfg.probe.seed)
    seed_dataloader(probe_val_loader, cfg.probe.seed)
    probe_train_batches = min(cfg.probe.max_train_batches or len(probe_train_loader), len(probe_train_loader))
    probe_val_batches = min(cfg.probe.max_val_batches or len(probe_val_loader), len(probe_val_loader))
    log.info(
        "Linear probe will use %d train and %d validation source batches from %s",
        probe_train_batches,
        probe_val_batches,
        cfg.probe.dataset,
    )

    # create and init models
    device = torch.device(cfg.trainer.accelerator)
    torch.manual_seed(cfg.trainer.seed)
    toyjepa = build_toyjepa(cfg.arch)
    student_enc = toyjepa.student_encoder.to(device)
    teacher_enc = toyjepa.teacher_encoder.to(device)
    predictor = toyjepa.predictor.to(device)
    # performance boost with torch.compile
    if cfg.trainer.compile:
        inductor_config.triton.coalesce_tiling_analysis = False
        torch._dynamo.config.recompile_limit = 64
        for blk in (*student_enc.blocks, *teacher_enc.blocks, *predictor.blocks):
            blk.compile(dynamic=True)
    adamw_optimizer, muon_optimizer = init_optim(
        encoder=student_enc,
        predictor=predictor,
        cfg=cfg.optim,
    )
    optimizers: list[torch.optim.Optimizer] = [adamw_optimizer]
    if muon_optimizer is not None:
        optimizers.append(muon_optimizer)
    newton_schulz_flops_per_step = estimate_newton_schulz_flops_per_step(muon_optimizer)
    diagnostic_params = [
        param
        for model in (student_enc, predictor)
        for param in model.parameters()
        if param.requires_grad and param.ndim > 1
    ]
    grad_norms = GradNormTracker(
        diagnostic_params, grad_norm_groups(student_enc, predictor, diagnostic_params), device=device
    )
    # Count params
    student_params = count_parameters(student_enc)
    predictor_params = count_parameters(predictor)
    log.info(
        "Parameter counts: encoder=%s | predictor=%s | total_trainable_params=%s",
        f"{student_params.total:,}",
        f"{predictor_params.total:,}",
        f"{student_params.trainable + predictor_params.trainable:,}",
    )

    # go train!
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    start_epoch = 0
    optim_step = 0
    global_batch = 0
    cumulative_estimated_flops = 0
    cumulative_train_seconds = 0.0  # excludes validation, checkpointing and probes.
    optim_seconds = 0.0
    timed_optim_steps = 0

    # training resume logic
    resume_state = None
    if cfg.checkpoint.resume_path is not None:
        resume_path = Path(cfg.paths.project_root) / "outputs" / "pretrain" / cfg.checkpoint.resume_path
        resume_state = load_checkpoint(
            resume_path,
            student_encoder=student_enc,
            teacher_encoder=teacher_enc,
            predictor=predictor,
            optimizer=optimizers[0],
            muon_optimizer=optimizers[1] if cfg.optim.name == "muon" else None,
            map_location=device,
        )
        start_epoch = resume_state["epoch"]
        global_batch = resume_state["global_batch"]
        optim_step = resume_state["optim_step"]
        cumulative_estimated_flops = resume_state["cumulative_estimated_flops"]
        cumulative_train_seconds = resume_state["cumulative_train_seconds"]
        saved_cfg = resume_state["cfg"]
        # checkpoints saved before arch.*.gated_attn and arch.*.peri_norm existed trained without them
        for part in ("encoder", "predictor"):
            saved_cfg["arch"][part].setdefault("gated_attn", False)
            saved_cfg["arch"][part].setdefault("peri_norm", False)
        current_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
        saved_datasets, current_datasets = sorted(saved_cfg["data"]), sorted(current_cfg["data"])
        if saved_datasets != current_datasets:
            raise ValueError(f"Resume config mismatch at data: checkpoint={saved_datasets} current={current_datasets}")
        data_keys = [("data", name, field) for name in current_datasets for field in RESUME_MATCH_DATA_FIELDS]
        for keys in (*RESUME_MATCH_KEYS, *data_keys):
            saved_value, current_value = saved_cfg, current_cfg
            for key in keys:
                saved_value, current_value = saved_value.get(key), current_value[key]
            if saved_value != current_value:
                raise ValueError(
                    f"Resume config mismatch at {'.'.join(keys)}: checkpoint={saved_value} current={current_value}"
                )

    # stop_at_epoch only ends the loop early for shorter test runs.
    # LR schedule still works according to cfg.trainer.epochs
    stop_at_epoch = cfg.trainer.stop_at_epoch
    if stop_at_epoch != -1 and not start_epoch < stop_at_epoch <= cfg.trainer.epochs:
        raise ValueError(
            f"trainer.stop_at_epoch={stop_at_epoch} must be -1 or in ({start_epoch}, {cfg.trainer.epochs}]"
        )
    final_epoch = cfg.trainer.epochs if stop_at_epoch == -1 else stop_at_epoch

    grad_accum_steps = cfg.trainer.gradient_accumulation_steps
    total_batches = num_batches * cfg.trainer.epochs
    total_optim_steps = math.ceil(num_batches / grad_accum_steps) * cfg.trainer.epochs
    final_optim_step = math.ceil(num_batches / grad_accum_steps) * final_epoch
    run_name = Path(cfg.paths.output_dir).name.replace("_", "-")
    if cfg.trainer.run_name is not None:
        run_name = f"{cfg.trainer.run_name}-{cfg.arch.name}"
    checkpoint_wandb_id = resume_state.get("wandb_id") if resume_state else None
    if cfg.checkpoint.fork_wandb_run and checkpoint_wandb_id is None:
        raise ValueError("checkpoint.fork_wandb_run needs a resume_path checkpoint that stores a W&B run id")
    wandb_id = None if cfg.checkpoint.fork_wandb_run else checkpoint_wandb_id
    fork_from = None
    if cfg.checkpoint.fork_wandb_run:
        # The new run inherits the source run's history up to and including this step
        fork_step = wandb_fork_step(checkpoint_wandb_id, optimizer_step=optim_step)
        fork_from = f"{checkpoint_wandb_id}?_step={fork_step}"
        log.info("Forking W&B run %s at _step=%d (optimizer_step=%d)", checkpoint_wandb_id, fork_step, optim_step)
    wandb_run = wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True, enum_to_str=True),
        dir=cfg.paths.output_dir,
        id=wandb_id,
        resume="must" if wandb_id else None,
        fork_from=fork_from,
        reinit="finish_previous",
    )
    wandb_run.define_metric("epoch")
    wandb_run.define_metric("optimizer_step")
    wandb_run.define_metric(FLOP_AXIS)
    wandb_run.define_metric(TRAIN_TIME_AXIS)
    wandb_run.define_metric("train/*", step_metric=FLOP_AXIS)
    wandb_run.define_metric("diagnostics/*", step_metric=FLOP_AXIS)
    wandb_run.define_metric("probe/*", step_metric="epoch")

    if start_epoch == 0:
        save_path = Path(cfg.checkpoint.dir) / epoch_checkpoint_name(
            0, optim_step=optim_step, total_optim_steps=total_optim_steps, optim_cfg=cfg.optim
        )
        save_checkpoint(
            path=save_path,
            student_encoder=student_enc,
            teacher_encoder=teacher_enc,
            predictor=predictor,
            optimizer=optimizers[0],
            muon_optimizer=optimizers[1] if cfg.optim.name == "muon" else None,
            completed_epochs=0,
            global_batch=global_batch,
            optimizer_step=optim_step,
            cumulative_estimated_flops=cumulative_estimated_flops,
            cumulative_train_seconds=cumulative_train_seconds,
            cfg=cfg,
            wandb_id=wandb_run.id,
        )
        log.info("Saved initial checkpoint in path %s", save_path)

    student_enc.train(), teacher_enc.eval(), predictor.train()
    synchronize()
    train_start_time = time.perf_counter()
    last_step_end = train_start_time
    for epoch in range(start_epoch, final_epoch):
        log.info("Starting epoch %d/%d", epoch + 1, cfg.trainer.epochs)
        seed = epoch_seed(cfg.trainer.seed, epoch)
        seed_dataloader(train_loader, seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for i, batch in enumerate(train_loader):
            global_batch += 1
            batch_idx = i + 1

            is_window_start = i % grad_accum_steps == 0
            is_accum_boundary = batch_idx % grad_accum_steps == 0
            is_last_batch = batch_idx == num_batches
            should_step = is_accum_boundary or is_last_batch
            diagnostics_due = should_step and optim_step % cfg.trainer.diagnostics_interval == 0

            # divide loss by number of accumulation steps
            window_start_zero = (i // grad_accum_steps) * grad_accum_steps
            accum_window_size = min(grad_accum_steps, num_batches - window_start_zero)

            # start compute timer
            if is_window_start:
                window_token_count = 0
                window_estimated_flops = 0
                window_losses = []

            # forward + backward
            with torch.autocast(device_type=device.type, dtype=DTYPE_MAP.get(cfg.trainer.precision, torch.bfloat16)):
                loss, target_loss, context_loss, loss_for_logging = forward_loss(
                    batch=batch,
                    student_enc=student_enc,
                    teacher_enc=teacher_enc,
                    predictor=predictor,
                    loss_cfg=cfg.loss,
                    device=device,
                )
                loss = loss / accum_window_size
            loss.backward()
            window_losses.append(loss_for_logging)
            batch_size = batch["inputs"].shape[0]
            num_patches = math.prod(batch["tokenized_shape"])
            window_token_count += batch_size * num_patches

            # estimate flops
            num_context = batch["context_ids"].shape[1]
            num_target = batch["target_ids"].shape[1]
            num_registers = cfg.arch.encoder.num_registers
            student_toks = num_context + num_registers
            teacher_toks = num_context + num_target + num_registers
            predictor_toks = num_context + num_target
            student_flops_per_tok = estimate_flops_per_token(
                module=student_enc,
                sequence_length=student_toks,
                num_attn_blocks=cfg.arch.encoder.depth,
                d_model=cfg.arch.encoder.d_embed,
            )
            teacher_flops_per_tok = estimate_flops_per_token(
                module=teacher_enc,
                sequence_length=teacher_toks,
                num_attn_blocks=cfg.arch.encoder.depth,
                d_model=cfg.arch.encoder.d_embed,
                training=False,
            )
            predictor_flops_per_tok = estimate_flops_per_token(
                module=predictor,
                sequence_length=predictor_toks,
                num_attn_blocks=cfg.arch.predictor.depth,
                d_model=cfg.arch.predictor.d_embed,
            )
            window_estimated_flops += batch_size * (
                student_toks * student_flops_per_tok
                + teacher_toks * teacher_flops_per_tok
                + predictor_toks * predictor_flops_per_tok
            )

            # decide whether to continue accumulating or not
            if not should_step:
                continue
            window_estimated_flops += newton_schulz_flops_per_step
            cumulative_estimated_flops += window_estimated_flops

            # optimize
            should_log = cfg.trainer.log_every > 0 and (optim_step + 1) % cfg.trainer.log_every == 0
            log_due = should_log or is_last_batch
            grad_norms.update()
            params_before = None
            if diagnostics_due:
                params_before = [param.detach().clone() for param in diagnostic_params]

            optim_step += 1
            lr = apply_optim_schedules(
                optimizers,
                cfg.optim,
                step=optim_step,
                total_steps=total_optim_steps,
            )
            synchronize()
            optim_start = time.perf_counter()
            for optimizer in optimizers:
                optimizer.step()
            synchronize()
            optim_seconds += time.perf_counter() - optim_start
            timed_optim_steps += 1
            if params_before is not None:
                old_norm_sq = torch.stack([param.float().square().sum() for param in params_before]).sum()
                update_norm_sq = torch.stack(
                    [
                        (param.detach() - old).float().square().sum()
                        for param, old in zip(diagnostic_params, params_before, strict=True)
                    ]
                ).sum()
                update_to_weight = torch.log10((update_norm_sq / old_norm_sq.clamp_min(1e-24)).sqrt().clamp_min(1e-12))
                params_before = None
            update_ema(student_enc.parameters(), teacher_enc.parameters(), m=cfg.ema.momentum)
            for optimizer in optimizers:
                optimizer.zero_grad()

            window_losses = torch.cat(window_losses)
            window_mean_loss = window_losses.mean().item()
            window_max_loss = window_losses.max().item()
            # target and context loss: just take final at the end
            target_loss_f = target_loss.item()
            context_loss_f = context_loss.item()

            # compute timings
            synchronize()
            step_end = time.perf_counter()
            wall_dt = step_end - last_step_end
            last_step_end = step_end
            cumulative_train_seconds += wall_dt

            if diagnostics_due:
                layer_stats = forward_diagnostics(
                    diagnostic_batches,
                    student_enc=student_enc,
                    predictor=predictor,
                    device=device,
                    dtype=DTYPE_MAP.get(cfg.trainer.precision, torch.bfloat16),
                )
                log.info("DIAGNOSTIC step=%d  log10_update_to_weight=%.2f", optim_step, update_to_weight.item())
                for line in layer_lines(layer_stats):
                    log.info("  %s", line)
                wandb_run.log(
                    {
                        FLOP_AXIS: cumulative_estimated_flops / FLOPS_PER_PFLOP,
                        TRAIN_TIME_AXIS: cumulative_train_seconds / SECONDS_PER_HOUR,
                        "optimizer_step": optim_step,
                        "diagnostics/log10_update_to_weight": update_to_weight.item(),
                        **{f"diagnostics/{key}": value for key, value in layer_stats.items()},
                    }
                )
                # the diagnostic forward is not training time
                last_step_end = time.perf_counter()

            # log
            train_meter.update(
                loss=window_mean_loss,
                target_loss=target_loss_f,
                context_loss=context_loss_f,
                max_loss=window_max_loss,
                token_count=window_token_count,
                wall_dt=wall_dt,
                estimated_flops=window_estimated_flops,
            )

            if log_due:
                metrics = train_meter.summary(
                    step_idx=optim_step,
                    total_steps=final_optim_step,
                    elapsed_s=time.perf_counter() - train_start_time,
                )
                grad_norm = grad_norms.total_norm()
                max_mem_bytes = torch.cuda.memory.max_memory_allocated(device=cfg.trainer.accelerator)
                log.info(
                    "elapsed=%s  |  epoch=%d/%d  step=%d/%d  batch=%d/%d  |  adamw_lr=%.2e",
                    metrics.time_elapsed,
                    epoch + 1,
                    cfg.trainer.epochs,
                    optim_step,
                    math.ceil(total_batches / grad_accum_steps),
                    global_batch,
                    total_batches,
                    lr,
                )
                log.info(
                    "  eta=%s  |  tflops/s=%.1f  max_mem=%.0fMib  toks/s=%.0f  |  step_time=%.3fs",
                    metrics.eta,
                    metrics.tflops_per_s,
                    max_mem_bytes / 1024**2,
                    metrics.toks_per_s,
                    metrics.wall_dt,
                )
                log.info(
                    "  loss: mean=%.4f (tgt=%.3f ctxt=%.3f)  max=%.3f",
                    metrics.loss,
                    metrics.target_loss,
                    metrics.context_loss,
                    metrics.max_loss,
                )
                grad_norm_max = grad_norms.pop_window_max()
                grad_norm_max_all = grad_norm_max.pop("all")
                log.info("  grad_norm=%.2e  window max=%.2e", grad_norm, grad_norm_max_all)
                grad_norm_max = {f"grad_norm_max/{group}": value for group, value in grad_norm_max.items()}
                for line in layer_lines(grad_norm_max, fmt="8.2e"):
                    log.info("    %s", line)
                wandb_run.log(
                    {
                        FLOP_AXIS: cumulative_estimated_flops / FLOPS_PER_PFLOP,
                        TRAIN_TIME_AXIS: cumulative_train_seconds / SECONDS_PER_HOUR,
                        "optimizer_step": optim_step,
                        "train/loss": metrics.loss,
                        "train/max_loss": metrics.max_loss,
                        "train/step_time_s": metrics.wall_dt,
                        "train/lr": lr,
                        "train/max_mem_gib": max_mem_bytes / 1024**3,
                        "diagnostics/grad_norm": grad_norm,
                        "diagnostics/grad_norm_max/all": grad_norm_max_all,
                        **{f"diagnostics/{key}": value for key, value in grad_norm_max.items()},
                    },
                )

                train_meter.reset_window()

            # compute validation loss
            if optim_step % cfg.trainer.val_check_interval == 0:
                val_losses = run_validation(
                    val_loader=val_loader,
                    student_enc=student_enc,
                    teacher_enc=teacher_enc,
                    predictor=predictor,
                    loss_cfg=cfg.loss,
                    device=device,
                    num_batches=cfg.trainer.num_val_batches,
                    precision=cfg.trainer.precision,
                )
                log.info(
                    "validation_loss=%.4f (target_loss=%.3f, context_loss=%.3f)",
                    val_losses.loss,
                    val_losses.target_loss,
                    val_losses.context_loss,
                )
                wandb_run.log(
                    {
                        FLOP_AXIS: cumulative_estimated_flops / FLOPS_PER_PFLOP,
                        TRAIN_TIME_AXIS: cumulative_train_seconds / SECONDS_PER_HOUR,
                        "optimizer_step": optim_step,
                        "train/val_loss": val_losses.loss,
                    }
                )
                last_step_end = time.perf_counter()

        completed_epoch = epoch + 1
        save_path = Path(cfg.checkpoint.dir) / epoch_checkpoint_name(
            completed_epoch, optim_step=optim_step, total_optim_steps=total_optim_steps, optim_cfg=cfg.optim
        )
        save_checkpoint(
            path=save_path,
            student_encoder=student_enc,
            teacher_encoder=teacher_enc,
            predictor=predictor,
            optimizer=optimizers[0],
            muon_optimizer=optimizers[1] if cfg.optim.name == "muon" else None,
            completed_epochs=completed_epoch,
            global_batch=global_batch,
            optimizer_step=optim_step,
            cumulative_estimated_flops=cumulative_estimated_flops,
            cumulative_train_seconds=cumulative_train_seconds,
            cfg=cfg,
            wandb_id=wandb_run.id,
        )
        log.info(
            "Saved epoch %d checkpoint in path %s",
            completed_epoch,
            save_path,
        )
        mean_optim_step_ms = 1e3 * optim_seconds / timed_optim_steps
        wandb_run.summary["train/mean_optim_step_ms"] = mean_optim_step_ms

        if completed_epoch % cfg.trainer.probe_every == 0:
            log.info("Evaluating epoch %d teacher representations with a fresh linear probe", completed_epoch)
            probe_metrics, probe_rankme, probe_rankme_centered = evaluate_linear_probe(
                train_source_loader=probe_train_loader,
                val_source_loader=probe_val_loader,
                encoder=teacher_enc,
                input_dim=cfg.arch.encoder.d_embed * len(teacher_enc.hierarchical_layers),
                num_registers=cfg.arch.encoder.num_registers,
                cfg=cfg.probe,
                device=device,
                precision=cfg.trainer.precision,
                pin_memory=cfg.loader.pin_memory,
                log=log,
                pretrain_epoch=completed_epoch,
            )
            log.info(
                "epoch=%d/%d probe_val_loss=%.4f probe_top1=%.2f%% probe_top5=%.2f%% "
                "probe_macro_precision=%.2f%% probe_macro_recall=%.2f%%",
                completed_epoch,
                cfg.trainer.epochs,
                probe_metrics.loss,
                100.0 * probe_metrics.top1,
                100.0 * probe_metrics.top5,
                100.0 * probe_metrics.macro_precision,
                100.0 * probe_metrics.macro_recall,
            )
            log.info(
                "epoch=%d/%d RankMe of probe train features=%.1f (centered=%.1f)",
                completed_epoch,
                cfg.trainer.epochs,
                probe_rankme,
                probe_rankme_centered,
            )
            wandb_run.log(
                {
                    "epoch": completed_epoch,
                    FLOP_AXIS: cumulative_estimated_flops / FLOPS_PER_PFLOP,
                    TRAIN_TIME_AXIS: cumulative_train_seconds / SECONDS_PER_HOUR,
                    "optimizer_step": optim_step,
                    "probe/top1": 100.0 * probe_metrics.top1,
                    "probe/top5": 100.0 * probe_metrics.top5,
                    "probe/recall": 100.0 * probe_metrics.macro_recall,
                    "probe/precision": 100.0 * probe_metrics.macro_precision,
                    "probe/rankme": probe_rankme,
                    "probe/rankme_centered": probe_rankme_centered,
                }
            )
        last_step_end = time.perf_counter()
    if timed_optim_steps > 0:
        log.info("Mean optimizer.step() time over %d steps: %.1f ms", timed_optim_steps, mean_optim_step_ms)
    if final_epoch < cfg.trainer.epochs:
        log.info(
            "Stopped after epoch %d of the %d-epoch schedule (trainer.stop_at_epoch)", final_epoch, cfg.trainer.epochs
        )
    wandb_run.finish()


if __name__ == "__main__":
    main()
