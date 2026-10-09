"""Typed config schemas for ToyJEPA experiments."""

from dataclasses import dataclass
from enum import StrEnum, auto

from hydra.core.config_store import ConfigStore


class Split(StrEnum):
    train = auto()
    validation = auto()
    test = auto()


class Accelerator(StrEnum):
    cpu = auto()
    cuda = auto()
    mps = auto()


class Precision(StrEnum):
    bf16 = auto()


class OptimName(StrEnum):
    adamw = auto()
    muon = auto()


class CacheDtype(StrEnum):
    float16 = auto()
    float32 = auto()


class Modality(StrEnum):
    image = auto()
    video = auto()


@dataclass
class EncoderConfig:
    in_channels: int
    d_embed: int
    depth: int
    num_heads: int
    mlp_ratio: float
    patch_size: int
    tubelet_size: int
    num_registers: int
    qkv_bias: bool
    qk_norm: bool
    gated_attn: bool
    peri_norm: bool
    drop_path_rate: float
    p_drop: float
    p_drop_attn: float


@dataclass
class PredictorConfig:
    d_embed: int
    depth: int
    num_heads: int
    mlp_ratio: float
    qkv_bias: bool
    qk_norm: bool
    gated_attn: bool
    peri_norm: bool
    p_drop: float
    p_drop_attn: float


@dataclass
class ModelConfig:
    name: str
    ref_img_size: list[int]
    num_masking_policies: int
    use_rope: bool
    interpolate_rope: bool
    encoder: EncoderConfig
    predictor: PredictorConfig


@dataclass
class PathsConfig:
    project_root: str
    data_root: str
    output_dir: str


@dataclass
class TrainerConfig:
    run_name: str | None
    seed: int
    compile: bool
    train_split: Split
    val_split: Split
    val_dataset: str
    train_shuffle: bool
    val_shuffle: bool
    devices: int
    accelerator: Accelerator
    epochs: int
    stop_at_epoch: int
    precision: Precision
    gradient_accumulation_steps: int
    val_check_interval: int
    num_val_batches: int
    log_every: int
    diagnostics_interval: int
    diagnostic_batches: int
    probe_every: int


@dataclass
class LoaderConfig:
    token_budget: int
    prefetch_factor: int
    num_workers: int
    pin_memory: bool


@dataclass
class MaskConfig:
    enabled: bool
    policy_weights: list[float]


@dataclass
class LossConfig:
    lambda_context: float
    temporal_scale: float
    eps: float


@dataclass
class EMAConfig:
    momentum: float


@dataclass
class OptimConfig:
    name: OptimName
    start_lr: float
    lr: float
    final_lr: float
    warmup_ratio: float
    cooldown_ratio: float
    betas: list[float]
    wd: float
    eps: float


@dataclass
class CheckpointConfig:
    dir: str
    resume_path: str | None
    fork_wandb_run: bool


@dataclass
class LinearProbeOptimConfig:
    lr: float
    final_lr: float
    betas: list[float]
    wd: float
    eps: float


@dataclass
class LinearProbeConfig:
    dataset: str
    train_split: Split
    val_split: Split
    epochs: int
    seed: int
    log_every: int
    max_train_batches: int | None
    max_val_batches: int | None
    num_classes: int
    dropout: float
    batch_size: int
    cache_dtype: CacheDtype
    optim: LinearProbeOptimConfig


@dataclass
class DataConfig:
    name: str
    dataset_cls: str
    modality: Modality
    root: str
    target_fps: int | None
    quantization_interval: int | None
    overwrite_manifest: bool


@dataclass
class PretrainConfig:
    paths: PathsConfig
    trainer: TrainerConfig
    loader: LoaderConfig
    mask: MaskConfig
    loss: LossConfig
    ema: EMAConfig
    optim: OptimConfig
    probe: LinearProbeConfig
    checkpoint: CheckpointConfig
    arch: ModelConfig
    data: dict[str, DataConfig]


ConfigStore.instance().store(name="pretrain_config", node=PretrainConfig)
