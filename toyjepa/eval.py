"""Deterministic frozen-feature evaluation for ToyJEPA encoders."""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from toyjepa.config import LinearProbeConfig
from toyjepa.datasets import Batch
from toyjepa.models.probes import MeanPoolLinearProbe

DTYPE_MAP = {"bf16": torch.bfloat16}
CACHE_DTYPE_MAP = {"float16": torch.float16, "float32": torch.float32}


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    loss: float
    top1: float
    top5: float
    macro_precision: float
    macro_recall: float


def prepare_inputs(inputs: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move uint8 clips to the device and normalize pixels to [-1, 1]."""
    inputs = inputs.to(device, non_blocking=True).float().div_(255.0).sub_(0.5).div_(0.5)
    if inputs.shape[1] == 1:
        inputs = inputs.expand(-1, 3, -1, -1, -1)
    return inputs


def _limited_batches(loader: DataLoader, max_batches: int | None) -> Iterable[Batch]:
    if max_batches is None:
        return loader
    if max_batches <= 0:
        raise ValueError("max_batches must be positive or null")
    return islice(loader, max_batches)


@torch.inference_mode()
def extract_features(
    *,
    loader: DataLoader,
    encoder: nn.Module,
    probe: MeanPoolLinearProbe,
    device: torch.device,
    precision: str,
    cache_dtype: torch.dtype,
    max_batches: int | None,
    description: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a frozen encoder once and return CPU pooled features and labels."""
    encoder.eval()
    feature_batches = []
    label_batches = []
    batches = _limited_batches(loader, max_batches)
    total = len(loader) if max_batches is None else min(max_batches, len(loader))

    for batch in tqdm(batches, total=total, desc=description):
        labels = batch["labels"]
        if labels is None:
            raise ValueError("The selected probe split does not contain labels")
        inputs = prepare_inputs(batch["inputs"], device)
        with torch.autocast(
            device_type=device.type,
            dtype=DTYPE_MAP[precision],
            enabled=device.type == "cuda",
        ):
            tokens = encoder(inputs)
            features = probe.pool_tokens(tokens)
        feature_batches.append(features.to(device="cpu", dtype=cache_dtype))
        label_batches.append(labels.cpu())

    if not feature_batches:
        raise ValueError("Feature extraction produced no batches")
    return torch.cat(feature_batches), torch.cat(label_batches)


def rankme(features: torch.Tensor, *, device: torch.device, center: bool = False, eps: float = 1e-7) -> float:
    """RankMe effective rank (Garrido et al. 2023) of an N,D embedding matrix.

    exp of the entropy of the L1-normalized singular values; at most D. Uncentered, as in the paper,
    unless `center`: subtracting the per-dimension mean removes the feature average's influence,
    which otherwise dominates one singular value.
    """
    if features.ndim != 2:
        raise ValueError(f"Expected features with shape N,D, got {tuple(features.shape)}")
    # Singular values from the D,D Gram matrix; float64 because N is large
    embeddings = features.to(device=device, dtype=torch.float64)
    if center:
        embeddings -= embeddings.mean(dim=0)
    singular_values = torch.linalg.eigvalsh(embeddings.T @ embeddings).clamp_min(0.0).sqrt()
    p = singular_values / singular_values.sum() + eps
    return torch.exp(-(p * p.log()).sum()).item()


def build_cached_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("probe.batch_size must be positive")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
        generator=generator,
    )


def _update_counts(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int, torch.Tensor]:
    predictions = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
    matches = predictions.eq(labels.unsqueeze(1))
    return (
        int(matches[:, :1].sum().item()),
        int(matches.any(dim=1).sum().item()),
        predictions[:, 0],
    )


def macro_precision_recall(confusion: torch.Tensor) -> tuple[float, float]:
    """Compute zero-safe macro precision and recall from a confusion matrix."""
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError(f"Expected a square confusion matrix, got {tuple(confusion.shape)}")
    confusion = confusion.to(dtype=torch.float64)
    true_positives = confusion.diag()
    predicted = confusion.sum(dim=0)
    actual = confusion.sum(dim=1)
    precision = torch.where(predicted > 0, true_positives / predicted, 0.0)
    recall = torch.where(actual > 0, true_positives / actual, 0.0)
    return precision.mean().item(), recall.mean().item()


def run_probe_epoch(
    *,
    loader: DataLoader,
    probe: MeanPoolLinearProbe,
    device: torch.device,
    precision: str,
    optimizer: torch.optim.Optimizer | None,
    log_every: int,
    log: logging.Logger,
) -> ClassificationMetrics:
    """Run one training or validation epoch over cached frozen features."""
    training = optimizer is not None
    probe.train(training)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total_samples = 0
    confusion = torch.zeros(
        (probe.num_classes, probe.num_classes),
        dtype=torch.int64,
        device=device,
    )

    for batch_idx, (features, labels) in enumerate(loader, start=1):
        features = features.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if labels.numel() and int(labels.max()) >= probe.num_classes:
            raise ValueError(f"Found label {int(labels.max())}, but probe has {probe.num_classes} classes")

        grad_context = torch.enable_grad() if training else torch.no_grad()
        with (
            grad_context,
            torch.autocast(
                device_type=device.type,
                dtype=DTYPE_MAP[precision],
                enabled=device.type == "cuda",
            ),
        ):
            logits = probe.forward_features(features)
            loss = loss_fn(logits, labels)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_size = labels.shape[0]
        top1, top5, predictions = _update_counts(logits.detach(), labels)
        confusion_indices = labels * probe.num_classes + predictions
        confusion += torch.bincount(
            confusion_indices,
            minlength=probe.num_classes**2,
        ).reshape(probe.num_classes, probe.num_classes)
        total_loss += loss.item() * batch_size
        total_top1 += top1
        total_top5 += top5
        total_samples += batch_size

        if training and log_every > 0 and batch_idx % log_every == 0:
            log.info(
                "probe_step=%d/%d loss=%.4f top1=%.2f%%",
                batch_idx,
                len(loader),
                total_loss / total_samples,
                100.0 * total_top1 / total_samples,
            )

    if total_samples == 0:
        raise ValueError("Probe loader produced no samples")
    macro_precision, macro_recall = macro_precision_recall(confusion.cpu())
    return ClassificationMetrics(
        loss=total_loss / total_samples,
        top1=total_top1 / total_samples,
        top5=total_top5 / total_samples,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
    )


def evaluate_linear_probe(
    *,
    train_source_loader: DataLoader,
    val_source_loader: DataLoader,
    encoder: nn.Module,
    input_dim: int,
    num_registers: int,
    cfg: LinearProbeConfig,
    device: torch.device,
    precision: str,
    pin_memory: bool,
    log: logging.Logger,
    pretrain_epoch: int,
) -> tuple[ClassificationMetrics, float, float]:
    """Fit a fresh frozen-feature linear probe and return its best validation metrics.

    Also returns the RankMe of the pooled train features the probe classifies.
    """
    if cfg.epochs <= 0:
        raise ValueError("probe.epochs must be positive")

    encoder_was_training = encoder.training
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]

    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(cfg.seed)
            probe = MeanPoolLinearProbe(
                input_dim=input_dim,
                num_classes=cfg.num_classes,
                num_registers=num_registers,
                dropout=cfg.dropout,
            )
            cache_dtype = CACHE_DTYPE_MAP[cfg.cache_dtype]
            train_features, train_labels = extract_features(
                loader=train_source_loader,
                encoder=encoder,
                probe=probe,
                device=device,
                precision=precision,
                cache_dtype=cache_dtype,
                max_batches=cfg.max_train_batches,
                description=f"Epoch {pretrain_epoch}: caching probe train features",
            )
            val_features, val_labels = extract_features(
                loader=val_source_loader,
                encoder=encoder,
                probe=probe,
                device=device,
                precision=precision,
                cache_dtype=cache_dtype,
                max_batches=cfg.max_val_batches,
                description=f"Epoch {pretrain_epoch}: caching probe validation features",
            )
            log.info(
                "Cached probe features: train=%d validation=%d dim=%d",
                len(train_labels),
                len(val_labels),
                input_dim,
            )
            train_rankme = rankme(train_features, device=device)
            train_rankme_centered = rankme(train_features, device=device, center=True)
            train_loader = build_cached_loader(
                train_features,
                train_labels,
                batch_size=cfg.batch_size,
                shuffle=True,
                pin_memory=pin_memory,
                seed=cfg.seed,
            )
            val_loader = build_cached_loader(
                val_features,
                val_labels,
                batch_size=cfg.batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                seed=cfg.seed,
            )

            probe.to(device)
            optimizer = torch.optim.AdamW(
                probe.parameters(),
                lr=cfg.optim.lr,
                betas=tuple(cfg.optim.betas),
                weight_decay=cfg.optim.wd,
                eps=cfg.optim.eps,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.epochs,
                eta_min=cfg.optim.final_lr,
            )
            best_val_metrics = None

            for probe_epoch in range(1, cfg.epochs + 1):
                epoch_lr = optimizer.param_groups[0]["lr"]
                train_metrics = run_probe_epoch(
                    loader=train_loader,
                    probe=probe,
                    device=device,
                    precision=precision,
                    optimizer=optimizer,
                    log_every=cfg.log_every,
                    log=log,
                )
                val_metrics = run_probe_epoch(
                    loader=val_loader,
                    probe=probe,
                    device=device,
                    precision=precision,
                    optimizer=None,
                    log_every=0,
                    log=log,
                )
                log.info(
                    "pretrain_epoch=%d probe_epoch=%d/%d lr=%.2e train_loss=%.4f train_top1=%.2f%% "
                    "val_loss=%.4f val_top1=%.2f%% val_top5=%.2f%%",
                    pretrain_epoch,
                    probe_epoch,
                    cfg.epochs,
                    epoch_lr,
                    train_metrics.loss,
                    100.0 * train_metrics.top1,
                    val_metrics.loss,
                    100.0 * val_metrics.top1,
                    100.0 * val_metrics.top5,
                )
                if best_val_metrics is None or val_metrics.top1 > best_val_metrics.top1:
                    best_val_metrics = val_metrics
                scheduler.step()

            assert best_val_metrics is not None
            return best_val_metrics, train_rankme, train_rankme_centered
    finally:
        encoder.train(encoder_was_training)
        if device.type == "cuda":
            torch.cuda.empty_cache()
