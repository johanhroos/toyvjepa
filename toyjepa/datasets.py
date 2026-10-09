"""Create manifests and prepare datasets for models. Custom collator and batch sampler."""

import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import cv2
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset, Sampler
from torchcodec.decoders import VideoDecoder
from torchvision.io import ImageReadMode, decode_image
from tqdm import tqdm

cv2.setNumThreads(0)

type Modality = Literal["image", "video"]
type BoundingBox = dict[str, int]
type ItemMetadataDict = dict[str, int | float | str | None | list[BoundingBox]]
type TokenizedShape = tuple[int, int, int]
type Split = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class SampleSpec:
    dataset_name: str
    modality: Modality
    color_channels: int
    tokenized_shape: TokenizedShape


class DatasetItem(TypedDict):
    dataset_name: str
    item: Tensor
    metadata: ItemMetadataDict
    modality: Modality
    color_channels: int
    tokenized_shape: TokenizedShape


class Batch(TypedDict):
    dataset: str
    inputs: Tensor
    labels: Tensor | None
    context_ids: Tensor | None
    target_ids: Tensor | None
    mask_policy: int | None
    modality: Modality
    tokenized_shape: TokenizedShape


def normalize_ssv2_label(label: str) -> str:
    return " ".join(label.lower().replace("[", "").replace("]", "").split())


def read_jsonl(path: Path) -> list[ItemMetadataDict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def create_ssv2_manifest(
    *,
    root: Path,
    split: Split,
    target_fps: int,
    quantization_interval: int,
    tubelet_size: int = 2,
    overwrite: bool = False,
) -> Path:
    """Creates jsonl file with information about each video in the data split with the chosen frame sampling strategy.

    This makes it easier to do batch sampling for fixed tensor sizes without padding.

    Label files in dataset:
        labels.json
            -> maps label (without brackets) to index e.g.
                Approaching something with your camera: "0"
        train.json / validation.json
            -> hold labels for data splits e.g.
                "id": "78687"
                "template": "Holding [something] next to [something]"
        test.json
            -> only holds test split id's
        test-answers.csv
            -> holds id and label as string (without brackets) with semicolon separator
                50058;Stacking number of something
    Video files in dataset:
        named according to id as webm files e.g. 101.webm
    """
    if quantization_interval % tubelet_size != 0:
        raise ValueError("quantization_interval must be divisible by tubelet_size")
    labels_dir = root / "labels"
    video_dir = root / "videos" / "20bn-something-something-v2"
    manifest_path = root / "manifests" / f"{split}_fps{target_fps}_q{quantization_interval}.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and not overwrite:
        return manifest_path

    print(f"Creating {split} split manifest. This may take up to a few minutes...")

    label_path = labels_dir / "test-answers.csv" if split == "test" else labels_dir / f"{split}.json"
    if not label_path.exists():
        raise FileNotFoundError(f"Could not find Something-Something label file: {label_path}")

    if split == "test":
        df_labels = pd.read_csv(
            label_path,
            sep=";",
            names=["id", "template"],
        )
        labels = df_labels.to_dict(orient="records")
    else:
        with label_path.open("r", encoding="utf-8") as f:
            labels = json.load(f)

    label_mapping_path = labels_dir / "labels.json"
    if not label_mapping_path.exists():
        raise FileNotFoundError(f"Could not find Something-Something label map: {label_mapping_path}")
    class_to_idx: dict[str, int] = {}
    with label_mapping_path.open("r", encoding="utf-8") as f:
        class_to_idx = {normalize_ssv2_label(label): int(idx) for label, idx in json.load(f).items()}

    rows_written = 0
    missing_videos = 0
    missing_labels = 0
    tmp_path = manifest_path.with_name(f"{manifest_path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in tqdm(labels, desc=f"Processing SSv2 {split} split"):
            video_id = int(row["id"])
            video_path = video_dir / f"{video_id}.webm"
            if not video_path.exists():
                missing_videos += 1
                continue

            label_name = row.get("template")
            if label_name is not None:
                label_name = normalize_ssv2_label(label_name)
                label = class_to_idx[label_name]
            else:
                label = None
                missing_labels += 1
            decoder = VideoDecoder(
                video_path,
                dimension_order="NCHW",
                num_ffmpeg_threads=1,
                device="cpu",
            )
            decoder_metadata = decoder.metadata
            duration_seconds = decoder_metadata.duration_seconds
            source_fps = decoder_metadata.average_fps
            num_frames = len(decoder)

            sampled_fps = min(float(target_fps), float(source_fps))
            raw_frames = round(duration_seconds * sampled_fps)
            frames_sampled = round(raw_frames / quantization_interval) * quantization_interval
            max_frames = num_frames // quantization_interval * quantization_interval
            if max_frames == 0:
                missing_videos += 1
                continue
            frames_sampled = min(max(quantization_interval, frames_sampled), max_frames)

            manifest_row = {
                "video_id": video_id,
                "video_path": str(video_path.relative_to(root)),
                "modality": "video",
                "label_name": label_name,
                "label": label,
                "height": decoder_metadata.height,
                "width": decoder_metadata.width,
                "duration_seconds": duration_seconds,
                "source_fps": source_fps,
                "num_frames": num_frames,
                "frames_sampled": frames_sampled,
            }
            f.write(json.dumps(manifest_row, ensure_ascii=False))
            f.write("\n")
            rows_written += 1
    tmp_path.replace(manifest_path)
    print(
        f"Created {manifest_path}\n",
        f"{split} split length: {rows_written}\n",
        f"Missing videos count: {missing_videos}\n",
        f"Missing labels count: {missing_labels}",
    )
    return manifest_path


class SomethingSomethingV2(Dataset):
    """Something-Something V2 clips decoded with TorchCodec.

    Args:
        fps
            target fps; can cause variable length outputs due to differing video durations
        transforms
            an Albumentations transform called with images=video
        quantization_interval


    Output:
        Each item is a dict with:
            video: float tensor with shape ()
            label: integer class id
            metadata: custom type ItemMetadataDict

    Frames are sampled based on fps as opposed to a fixed number of frames sampled per video.
    The dataset class handles fetching and decoding dataset.
    """

    def __init__(
        self,
        *,
        root: Path,
        split: Split,
        target_fps: int = 8,
        quantization_interval: int = 4,
        output_size: tuple[int, int] = (224, 224),
        tubelet_size: int = 2,
        patch_size: int = 16,
        transforms=None,
        overwrite_manifest: bool = False,
    ) -> None:
        self.root = root.resolve()
        self.manifest_path = create_ssv2_manifest(
            root=self.root,
            split=split,
            target_fps=target_fps,
            quantization_interval=quantization_interval,
            tubelet_size=tubelet_size,
            overwrite=overwrite_manifest,
        )

        self.output_h, self.output_w = output_size
        self.transforms = transforms
        self.metadata = read_jsonl(self.manifest_path)
        self.sample_specs = [
            SampleSpec(
                dataset_name="ssv2",
                modality="video",
                color_channels=3,
                tokenized_shape=(
                    row["frames_sampled"] // tubelet_size,
                    self.output_h // patch_size,
                    self.output_w // patch_size,
                ),
            )
            for row in self.metadata
        ]

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(
        self,
        idx: int,
    ) -> DatasetItem:
        sample = self.metadata[idx]
        spec = self.sample_specs[idx]

        decoder = VideoDecoder(
            self.root / sample["video_path"],
            dimension_order="NCHW",
            num_ffmpeg_threads=1,
            device="cpu",
            output_dtype=torch.uint8,
        )
        indices = torch.linspace(start=0, end=sample["num_frames"] - 1, steps=sample["frames_sampled"]).round().long()
        frame_batch = decoder.get_frames_at(indices)
        video = rearrange(frame_batch.data, "T C H W -> T H W C").contiguous().numpy()
        if self.transforms is not None:
            video = self.transforms(images=video)["images"]
        video = rearrange(torch.from_numpy(np.ascontiguousarray(video)), "T H W C -> C T H W").contiguous()
        _, _, h, w = video.shape
        if (h, w) != (self.output_h, self.output_w):
            raise ValueError(
                f"Output dimensions in the video tensor don't match given output shape ({self.output_h, self.output_w})"
            )

        item: DatasetItem = {
            "dataset_name": spec.dataset_name,
            "item": video,
            "metadata": sample,
            "modality": spec.modality,
            "color_channels": spec.color_channels,
            "tokenized_shape": spec.tokenized_shape,
        }
        return item


@dataclass(frozen=True, slots=True)
class ImageNetDataSidecar:
    label: str
    height: int
    width: int
    depth: int
    bounding_boxes: list[BoundingBox]


def parse_imagenet_xml(path: Path) -> ImageNetDataSidecar:
    """ImageNet annotation formats are in xml.

     <annotation>
        <folder>val</folder>
        <filename>ILSVRC2012_val_00005512</filename>
        <source>
            <database>ILSVRC_2012</database>
        </source>
        <size>
            <width>200</width>
            <height>200</height>
            <depth>3</depth>
        </size>
        <segmented>0</segmented>
        <object>
            <name>n02916936</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>15</xmin>
                <ymin>0</ymin>
                <xmax>187</xmax>
                <ymax>199</ymax>
            </bndbox>
        </object>
    </annotation>
    """
    xml = ET.parse(path).getroot()

    size = xml.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))
    depth = int(size.findtext("depth"))

    objects = xml.findall("object")
    if not objects:
        raise ValueError(f"Missing objects in ImageNet annotation: {path}")

    label = objects[0].findtext("name")
    if label is None:
        raise ValueError(f"Missing object label in ImageNet annotation: {path}")

    bounding_boxes: list[BoundingBox] = []
    for obj in objects:
        box = obj.find("bndbox")
        bounding_box = {
            "xmin": int(box.findtext("xmin")),
            "ymin": int(box.findtext("ymin")),
            "xmax": int(box.findtext("xmax")),
            "ymax": int(box.findtext("ymax")),
        }
        bounding_boxes.append(bounding_box)
    return ImageNetDataSidecar(
        label=label,
        width=width,
        height=height,
        depth=depth,
        bounding_boxes=bounding_boxes,
    )


def create_imagenet_manifest(
    *,
    root: Path,
    split: Split,
    overwrite: bool = False,
) -> Path:
    """Create an ImageNet CLS-LOC manifest.

    Note: test labels aren't available. Evaluations must be submitted to
    https://www.kaggle.com/c/imagenet-object-localization-challenge.

    Data example path:
        ILSVRC/Data/CLS-LOC/train/n12998815/n12998815_9974.JPEG
        -> n12998815 is the synset
        ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00049999.JPEG
        ILSVRC/Data/CLS-LOC/test/ILSVRC2012_test_00099991.JPEG
    Annotations example path:
        ILSVRC/Annotations/CLS-LOC/train/n15075141/n15075141_9942.xml
        ILSVRC/Annotations/CLS-LOC/val/ILSVRC2012_val_00050000.xml
    Labels mappings:
        ILSVRC/Annotations/LOC_synset_mapping.txt
        -> e.g. n01484850 great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias
    """
    data_split = "val" if split == "validation" else split
    image_root = root / "Data" / "CLS-LOC" / data_split
    if not image_root.exists():
        raise FileNotFoundError(f"Could not find ImageNet image directory: {image_root}")
    annotation_root = None if data_split == "test" else root / "Annotations" / "CLS-LOC" / data_split
    manifest_path = root / "manifests" / f"{split}.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and not overwrite:
        return manifest_path

    label2name: dict[str, str] = {}
    class_to_idx: dict[str, int] = {}

    label_mapping_path = root / "LOC_synset_mapping.txt"
    if label_mapping_path.exists():
        with label_mapping_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                label, name = line.split(maxsplit=1)
                label2name[label] = name
                class_to_idx[label] = idx
    else:
        print("Couldn't find ImageNet synset mapping. Generating manifest without natural language labels.")
        train_image_root = root / "Data" / "CLS-LOC" / "train"
        synset_dirs = [
            class_dir
            for class_dir in sorted(train_image_root.iterdir())
            if class_dir.is_dir() and class_dir.name.startswith("n") and class_dir.name[1:].isdigit()
        ]
        class_to_idx = {class_dir.name: idx for idx, class_dir in enumerate(synset_dirs)}
        assert len(class_to_idx) == 1000, "Should be 1000 ImageNet classes."

    if data_split == "train":
        image_paths: list[Path] = []
        for class_dir in sorted(image_root.iterdir()):
            if not class_dir.is_dir() or class_dir.name not in class_to_idx:
                continue
            image_paths.extend(sorted(class_dir.glob("*.JPEG")))
    else:
        image_paths = sorted(image_root.glob("*.JPEG"))

    rows_written = 0
    tmp_path = manifest_path.with_name(f"{manifest_path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for image_path in tqdm(image_paths, desc=f"Processing ImageNet {split} split"):
            img_id = image_path.stem
            sidecar = None

            if data_split == "train":
                synset = image_path.parent.name
                xml_path = annotation_root / synset / f"{img_id}.xml"
                # Only 544,546 of the 1,281,167 train images have box annotations; labels come from the folder
                sidecar = parse_imagenet_xml(xml_path) if xml_path.exists() else None
            elif data_split == "val":
                xml_path = annotation_root / f"{img_id}.xml"
                sidecar = parse_imagenet_xml(xml_path)
                synset = sidecar.label
            else:
                synset = None

            if synset is None:
                label = None
                label_name = None
            else:
                label = class_to_idx[synset]
                label_name = label2name.get(synset)

            manifest_row = {
                "img_id": img_id,
                "img_path": str(image_path.relative_to(root)),
                "modality": "image",
                "label_name": label_name,
                "label": label,
                "synset": synset,
                "height": sidecar.height if sidecar is not None else None,
                "width": sidecar.width if sidecar is not None else None,
                "depth": sidecar.depth if sidecar is not None else None,
                "bounding_boxes": sidecar.bounding_boxes if sidecar is not None else None,
            }
            f.write(json.dumps(manifest_row, ensure_ascii=False))
            f.write("\n")
            rows_written += 1
    tmp_path.replace(manifest_path)
    print(
        f"Created {manifest_path}\n",
        f"{split} split length: {rows_written}\n",
    )
    return manifest_path


class ImageNet(Dataset):
    """ImageNet CLS-LOC images using the same sample contract as the video datasets."""

    def __init__(
        self,
        *,
        root: Path,
        split: Split,
        output_size: tuple[int, int] = (224, 224),
        patch_size: int = 16,
        transforms=None,
        overwrite_manifest: bool = False,
    ) -> None:
        self.root = root.resolve()
        self.manifest_path = create_imagenet_manifest(
            root=self.root,
            split=split,
            overwrite=overwrite_manifest,
        )

        self.output_h, self.output_w = output_size
        if self.output_h % patch_size != 0 or self.output_w % patch_size != 0:
            raise ValueError("ImageNet output_size must be divisible by patch_size")

        self.transforms = transforms
        self.metadata = read_jsonl(self.manifest_path)
        self.sample_specs = [
            SampleSpec(
                dataset_name="imagenet",
                modality="image",
                color_channels=3,
                tokenized_shape=(
                    1,
                    self.output_h // patch_size,
                    self.output_w // patch_size,
                ),
            )
            for _ in self.metadata
        ]

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(
        self,
        idx: int,
    ) -> DatasetItem:
        sample = self.metadata[idx]
        spec = self.sample_specs[idx]

        image = decode_image(
            self.root / sample["img_path"],
            mode=ImageReadMode.RGB,
        )
        image = rearrange(image, "C H W -> H W C").contiguous().numpy()
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        image = rearrange(torch.from_numpy(np.ascontiguousarray(image)), "H W C -> C H W").contiguous()
        _, h, w = image.shape
        if (h, w) != (self.output_h, self.output_w):
            raise ValueError(
                f"Output dimensions in the image tensor don't match given output shape ({self.output_h, self.output_w})"
            )

        item: DatasetItem = {
            "dataset_name": spec.dataset_name,
            "item": image.unsqueeze(1).contiguous(),
            "metadata": sample,
            "modality": spec.modality,
            "color_channels": spec.color_channels,
            "tokenized_shape": spec.tokenized_shape,
        }
        return item


class JEPABatchSampler(Sampler[list[int]]):
    """Groups dataset by modality and data shape to ensure even batch dimensions.

    Computes batch size to utilize approximately constant number of input tokens per batch.
    Must remember to set seed from training loop to get different randomized order.
    """

    def __init__(
        self,
        sample_specs: list[SampleSpec],
        token_budget: int,
        shuffle: bool = True,
    ) -> None:
        self.sample_specs = sample_specs
        self.token_budget = token_budget
        self.shuffle = shuffle
        self.seed = 0
        self.buckets: dict[
            tuple[
                str,
                Modality,
                int,
                TokenizedShape,
            ],
            list[int],
        ] = defaultdict(list)

        for index, spec in enumerate(self.sample_specs):
            key = (
                spec.dataset_name,
                spec.modality,
                spec.color_channels,
                spec.tokenized_shape,
            )
            self.buckets[key].append(index)

    def _batch_size(self, token_shape: TokenizedShape) -> int:
        num_tokens = math.prod(token_shape)
        return max(1, self.token_budget // num_tokens)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        batches: list[list[int]] = []
        for key, bucket_indices in self.buckets.items():
            indices = bucket_indices.copy()
            if self.shuffle:
                permutation = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in permutation]
            batch_size = self._batch_size(key[3])
            for start in range(0, len(indices), batch_size):
                batch = indices[start : start + batch_size]
                batches.append(batch)
        # shuffle batch order (remove correlation caused by token_shape)
        if self.shuffle:
            permutation = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in permutation]
        yield from batches

    def __len__(self) -> int:
        total = 0
        for key, indices in self.buckets.items():
            batch_size = self._batch_size(key[3])
            total += math.ceil(len(indices) / batch_size)
        return total

    def set_seed(self, seed: int) -> None:
        self.seed = seed


class JEPACollator:
    """Stack decoded clips and optionally add JEPA mask token ids.

    Dataset items should already contain fixed-size video tensors with shape C,T,H,W.
    If mask_generator is provided:
        mask_policy is randomly selected out of possibilities during call of batch
    """

    def __init__(
        self,
        mask_generator=None,
        mask_policy_weights: tuple[float, ...] = (0.5, 0.5),
    ) -> None:
        self.mask_generator = mask_generator
        self.weights = torch.tensor(mask_policy_weights, dtype=torch.float32)

    def __call__(self, samples: list[DatasetItem]) -> Batch:
        inputs = torch.stack([sample["item"] for sample in samples], dim=0)
        sample_labels = [sample["metadata"]["label"] for sample in samples]
        if any(label is None for label in sample_labels):
            labels = None
        else:
            labels = torch.tensor(sample_labels, dtype=torch.long)
        names = [sample["dataset_name"] for sample in samples]
        name = names[0]
        assert all(value == name for value in names), "Batch sampler returned a mixed-dataset batch"
        modalities = [sample["modality"] for sample in samples]
        modality = modalities[0]
        assert all(value == modality for value in modalities), "Batch sampler returned a mixed-modality batch"
        tokenized_shapes = [sample["tokenized_shape"] for sample in samples]
        tokenized_shape = tokenized_shapes[0]
        assert all(value == tokenized_shape for value in tokenized_shapes), (
            "Batch sampler received differing tokenized shapes"
        )

        if self.mask_generator:
            mask_policy = torch.multinomial(self.weights, num_samples=1, replacement=True).item()
            context_ids, target_ids = self.mask_generator(
                batch_size=len(samples),
                tokenized_vid_dims=tokenized_shape,
                mask_policy=mask_policy,
                device=inputs.device,
            )
        else:
            context_ids, target_ids = None, None
            mask_policy = None

        batch: Batch = {
            "dataset": name,
            "inputs": inputs,
            "labels": labels,
            "context_ids": context_ids,
            "target_ids": target_ids,
            "mask_policy": mask_policy,
            "modality": modality,
            "tokenized_shape": tokenized_shape,
        }
        return batch


class JEPAConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.sample_specs = [spec for dataset in datasets for spec in dataset.sample_specs]
