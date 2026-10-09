"""Build models and dataloaders."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import torch
import torch.nn as nn
from hydra.utils import get_class
from torch.utils.data import DataLoader, Dataset, get_worker_info

from toyjepa.config import ModelConfig, PretrainConfig
from toyjepa.datasets import JEPABatchSampler, JEPACollator, JEPAConcatDataset
from toyjepa.masks import sample_mask_tok_ids
from toyjepa.models.encoder import ViTEncoder
from toyjepa.models.predictor import ViTPredictor


@dataclass
class ToyJEPA:
    student_encoder: nn.Module
    teacher_encoder: nn.Module
    predictor: nn.Module


def give_hierarchical_layers(depth: int) -> list[int]:
    if depth == 4:
        hierarchical_layers = [0, 1, 2, 3]
    elif depth == 8:
        hierarchical_layers = [1, 3, 5, 7]
    elif depth == 12:
        hierarchical_layers = [2, 5, 8, 11]
    elif depth == 20:
        hierarchical_layers = [4, 9, 14, 19]
    elif depth == 24:
        hierarchical_layers = [4, 11, 17, 23]
    else:
        raise ValueError("ViT depth must be one of [4, 8, 12, 20, 24].")
    return hierarchical_layers


def build_encoder(config: ModelConfig) -> ViTEncoder:
    """Build one ToyJEPA encoder from an architecture config."""
    enc = config.encoder
    return ViTEncoder(
        ref_img_size=tuple(config.ref_img_size),
        patch_size=enc.patch_size,
        tubelet_size=enc.tubelet_size,
        depth=enc.depth,
        in_channels=enc.in_channels,
        d_embed=enc.d_embed,
        num_heads=enc.num_heads,
        num_registers=enc.num_registers,
        mlp_ratio=enc.mlp_ratio,
        qkv_bias=enc.qkv_bias,
        qk_norm=enc.qk_norm,
        gated_attn=enc.gated_attn,
        peri_norm=enc.peri_norm,
        use_rope=config.use_rope,
        interpolate_rope=config.interpolate_rope,
        drop_path_rate=enc.drop_path_rate,
        p_drop=enc.p_drop,
        p_drop_attn=enc.p_drop_attn,
        hierarchical_layers=give_hierarchical_layers(enc.depth),
    )


def build_toyjepa(config: ModelConfig) -> ToyJEPA:
    enc = config.encoder
    pred = config.predictor

    student_encoder = build_encoder(config)
    teacher_encoder = deepcopy(student_encoder).requires_grad_(False)
    predictor = ViTPredictor(
        ref_img_size=tuple(config.ref_img_size),
        patch_size=enc.patch_size,
        depth=pred.depth,
        d_embed_enc=enc.d_embed,
        d_embed_pred=pred.d_embed,
        num_heads=pred.num_heads,
        mlp_ratio=pred.mlp_ratio,
        qkv_bias=pred.qkv_bias,
        qk_norm=pred.qk_norm,
        gated_attn=pred.gated_attn,
        peri_norm=pred.peri_norm,
        use_rope=config.use_rope,
        interpolate_rope=config.interpolate_rope,
        num_enc_registers=enc.num_registers,
        p_drop=pred.p_drop,
        p_drop_attn=pred.p_drop_attn,
        hierarchical_layers=give_hierarchical_layers(enc.depth),
        num_masking_policies=config.num_masking_policies,
    )
    return ToyJEPA(
        student_encoder=student_encoder,
        teacher_encoder=teacher_encoder,
        predictor=predictor,
    )


def build_image_transforms(output_size: tuple[int, int], *, training: bool = True) -> A.Compose:
    output_h, output_w = output_size
    crop = A.RandomCrop(height=output_h, width=output_w) if training else A.CenterCrop(height=output_h, width=output_w)
    flip = A.HorizontalFlip(p=0.5 if training else 0.0)
    return A.Compose(
        [
            A.Resize(height=256, width=256),
            crop,
            flip,
        ]
    )


def build_video_transforms(output_size: tuple[int, int], *, training: bool = True) -> A.Compose:
    output_h, output_w = output_size
    crop = A.RandomCrop(height=output_h, width=output_w) if training else A.CenterCrop(height=output_h, width=output_w)
    flip = A.HorizontalFlip(p=0.5 if training else 0.0)
    return A.Compose(
        [
            A.Resize(height=256, width=256),
            crop,
            flip,
        ]
    )


def build_datasets(
    cfg: PretrainConfig,
    split: str,
    data: str | None,
    *,
    augment: bool,
) -> Dataset:
    datasets: list[Dataset] = []
    output_size = tuple(cfg.arch.ref_img_size)
    data_configs = cfg.data.values()

    for dataset in data_configs:
        if data is not None and dataset.name != data:
            continue
        dataset_cls = get_class(dataset.dataset_cls)
        if dataset.modality == "image":
            datasets.append(
                dataset_cls(
                    root=Path(dataset.root),
                    split=split,
                    output_size=output_size,
                    patch_size=cfg.arch.encoder.patch_size,
                    transforms=build_image_transforms(output_size, training=augment),
                    overwrite_manifest=dataset.overwrite_manifest,
                )
            )
        else:
            datasets.append(
                dataset_cls(
                    root=Path(dataset.root),
                    split=split,
                    target_fps=dataset.target_fps,
                    quantization_interval=dataset.quantization_interval,
                    output_size=output_size,
                    tubelet_size=cfg.arch.encoder.tubelet_size,
                    patch_size=cfg.arch.encoder.patch_size,
                    transforms=build_video_transforms(output_size, training=augment),
                    overwrite_manifest=dataset.overwrite_manifest,
                )
            )
    if len(datasets) == 0:
        raise ValueError("No datasets were built!")
    return JEPAConcatDataset(datasets)


def seed_worker_transforms(worker_id: int) -> None:
    """Reseed each dataset's Albumentations pipeline from this worker's torch seed.

    Albumentations keeps private generators seeded from system entropy, and forked
    workers would otherwise share one copy of that state. torch.initial_seed() is
    base_seed + worker_id, where base_seed is drawn from the loader's generator when
    the iterator is created, so crops and flips follow seed_dataloader.
    """
    worker_info = get_worker_info()
    assert worker_info is not None, "seed_worker_transforms must run inside a DataLoader worker"
    seed = torch.initial_seed() % 2**32
    for dataset in worker_info.dataset.datasets:
        if dataset.transforms is not None:
            dataset.transforms.set_random_seed(seed)


def build_dataloader(
    cfg: PretrainConfig,
    split: str,
    shuffle: bool,
    dataset: str | None = None,
    *,
    augment: bool,
    generate_masks: bool | None = None,
) -> DataLoader:
    dataset = build_datasets(cfg, split, dataset, augment=augment)
    use_masks = cfg.mask.enabled if generate_masks is None else generate_masks

    collate_fn = JEPACollator(
        mask_generator=sample_mask_tok_ids if use_masks else None,
        mask_policy_weights=tuple(cfg.mask.policy_weights),
    )

    sampler = JEPABatchSampler(
        sample_specs=dataset.sample_specs,
        token_budget=cfg.loader.token_budget,
        shuffle=shuffle,
    )
    return DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        prefetch_factor=cfg.loader.prefetch_factor,
        worker_init_fn=seed_worker_transforms,
        generator=torch.Generator(),
    )


def seed_dataloader(loader: DataLoader, seed: int) -> None:
    """Fix the batch order and worker seeds (masks, crops) of the loader's next iterator.

    Workers are not persistent, so every iterator starts fresh workers seeded from
    one draw of loader.generator. Call this before each pass that must be reproducible.
    """
    loader.batch_sampler.set_seed(seed)
    loader.generator.manual_seed(seed)
