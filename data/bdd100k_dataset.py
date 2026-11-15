"""Reusable PyTorch datasets for the prepared BDD100K 10K subset.

The datasets read CSV manifests produced by ``python -m data.prepare_bdd100k``.
Source images, 2018 detection JSONs and segmentation masks stay in their existing
sibling directories; the manifest keeps their paths relative to the project root.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "bdd100k"

# Stable 2018 BDD100K 2D detection category order used for training targets.
DETECTION_CLASS_NAMES = (
    "bike", "bus", "car", "motor", "person", "rider",
    "traffic light", "traffic sign", "train", "truck",
)
DETECTION_CLASS_TO_ID = {name: i for i, name in enumerate(DETECTION_CLASS_NAMES)}

# Official BDD100K semantic train IDs 0..18. Official void/ignore is 255.
TRAIN_ID_NAMES = (
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
)
SEGMENTATION_CLASS_NAMES = (
    "background", "road", "sidewalk", "vehicles", "vulnerable road users",
)
SEGMENTATION_IGNORE_INDEX = 255
# train_id -> project class_id: built environment/other -> background;
# person/rider -> vulnerable; vehicle types -> vehicles. Void remains ignored.
TRAIN_ID_TO_CLASS_ID = {
    0: 1, 1: 2,
    **{i: 0 for i in range(2, 11)},
    11: 4, 12: 4,
    **{i: 3 for i in range(13, 19)},
    255: SEGMENTATION_IGNORE_INDEX,
}

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def normalize_bdd100k_image(image: Image.Image) -> Image.Image:
    """Return an RGB image with EXIF orientation applied and BDD portrait frames fixed.

    Ten source JPEGs in this extracted 10K set are physically 720x1280; six of
    them have 1280x720 semantic masks. Rotating portrait frames 90 degrees
    counter-clockwise restores the common image/mask orientation.
    """
    image = ImageOps.exif_transpose(image)
    if image.width < image.height:
        image = image.transpose(Image.Transpose.ROTATE_90)
    return image.convert("RGB")


def read_train_id_mask(path: str | Path) -> np.ndarray:
    """Read train-ID indices without interpreting palette indices as RGB colors."""
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[2] not in (3, 4) or not np.all(array[..., :3] == array[..., :1]):
            raise ValueError(f"Expected a single-channel train-ID mask, got {array.shape} at {path}")
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D train-ID mask, got shape {array.shape} at {path}")
    if array.size and (int(array.min()) < 0 or int(array.max()) > 255):
        raise ValueError(f"Train-ID mask values must fit in uint8: {path}")
    return array.astype(np.uint8, copy=False)


def map_train_id_mask(mask: np.ndarray) -> np.ndarray:
    """Map official semantic train IDs to five project classes; keep void ignored.

    All unrecognized values, including official 255, map to 255 so they are
    excluded from the segmentation loss and metrics rather than treated as class 0.
    """
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D train-ID mask, got shape {array.shape}")
    if array.size and (int(array.min()) < 0 or int(array.max()) > 255):
        raise ValueError("Train-ID mask values must fit in uint8")
    array = array.astype(np.uint8, copy=False)
    lookup = np.full(256, SEGMENTATION_IGNORE_INDEX, dtype=np.uint8)
    for train_id, class_id in TRAIN_ID_TO_CLASS_ID.items():
        lookup[train_id] = class_id
    return lookup[array]


def assert_no_split_leakage(split_to_ids: Mapping[str, object]) -> None:
    """Raise ValueError if any image ID appears in more than one named split."""
    seen: dict[str, str] = {}
    conflicts: list[tuple[str, str, str]] = []
    for split, ids in split_to_ids.items():
        for image_id in ids:
            ident = str(image_id)
            if ident in seen:
                conflicts.append((ident, seen[ident], split))
            else:
                seen[ident] = split
    if conflicts:
        preview = ", ".join(f"{i} ({a}/{b})" for i, a, b in conflicts[:5])
        raise ValueError(f"Image IDs leak across splits: {preview}")


def audit_multitask_split_ids(
    detection_ids: Mapping[str, object],
    segmentation_ids: Mapping[str, object],
    *,
    reject_leakage: bool = True,
) -> dict:
    """Measure task-local and cross-task split intersections by exact image ID.

    Same-role sharing (for example, an ID used to train both heads) is allowed.
    Any ID used for training by either task and evaluation by either task is
    rejected, as is any validation/test intersection in the combined evaluation.
    """
    det = {name: {str(item) for item in detection_ids.get(name, ())}
           for name in ("train", "val", "test")}
    seg = {name: {str(item) for item in segmentation_ids.get(name, ())}
           for name in ("train", "val", "test")}

    def count(left: set[str], right: set[str]) -> int:
        return len(left & right)

    within_task = {
        f"detection_{left}_x_{right}": count(det[left], det[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    } | {
        f"segmentation_{left}_x_{right}": count(seg[left], seg[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    cross_task = {
        f"detection_{left}_x_segmentation_{right}": count(det[left], seg[right])
        for left in ("train", "val", "test")
        for right in ("train", "val", "test")
    }
    train_vs_evaluation = {
        "detection_train_x_detection_validation": count(det["train"], det["val"]),
        "detection_train_x_detection_test": count(det["train"], det["test"]),
        "segmentation_train_x_segmentation_validation": count(seg["train"], seg["val"]),
        "segmentation_train_x_segmentation_test": count(seg["train"], seg["test"]),
        "detection_train_x_segmentation_validation": count(det["train"], seg["val"]),
        "detection_train_x_segmentation_test": count(det["train"], seg["test"]),
        "segmentation_train_x_detection_validation": count(seg["train"], det["val"]),
        "segmentation_train_x_detection_test": count(seg["train"], det["test"]),
    }
    validation_vs_test = {
        "detection_validation_x_detection_test": count(det["val"], det["test"]),
        "segmentation_validation_x_segmentation_test": count(seg["val"], seg["test"]),
        "detection_validation_x_segmentation_test": count(det["val"], seg["test"]),
        "segmentation_validation_x_detection_test": count(seg["val"], det["test"]),
    }
    combined_train = det["train"] | seg["train"]
    combined_validation = det["val"] | seg["val"]
    combined_test = det["test"] | seg["test"]
    combined = {
        "train_ids": len(combined_train),
        "validation_ids": len(combined_validation),
        "test_ids": len(combined_test),
        "train_vs_validation": count(combined_train, combined_validation),
        "train_vs_test": count(combined_train, combined_test),
        "validation_vs_test": count(combined_validation, combined_test),
    }
    result = {
        "within_task_intersections": within_task,
        "cross_task_intersections": cross_task,
        "training_vs_evaluation_intersections": train_vs_evaluation,
        "validation_vs_test_intersections": validation_vs_test,
        "same_split_shared_ids": {
            "train": count(det["train"], seg["train"]),
            "validation": count(det["val"], seg["val"]),
            "test": count(det["test"], seg["test"]),
        },
        "combined_splits": combined,
    }
    failures = {name: value for group in (train_vs_evaluation, validation_vs_test)
                for name, value in group.items() if value}
    failures.update({f"combined_{name}": value for name, value in (
        ("train_vs_validation", combined["train_vs_validation"]),
        ("train_vs_test", combined["train_vs_test"]),
        ("validation_vs_test", combined["validation_vs_test"]),
    ) if value})
    result["leakage_free"] = not failures
    if failures and reject_leakage:
        details = ", ".join(f"{name}={value}" for name, value in sorted(failures.items()))
        raise ValueError(f"Combined BDD100K splits leak image IDs: {details}")
    return result


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing BDD100K manifest {path}. Run `python -m data.prepare_bdd100k` from the project root."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _image_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float()


def _as_image_tensor(value) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value
        if tensor.ndim == 3 and tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        if tensor.dtype == torch.uint8:
            tensor = tensor.float().div_(255.0)
            mean = torch.as_tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device)[:, None, None]
            std = torch.as_tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device)[:, None, None]
            tensor = (tensor - mean) / std
        return tensor.float()
    array = np.asarray(value)
    if array.dtype != np.uint8 and array.size and np.nanmax(array) <= 1.0:
        image = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8))
    else:
        image = Image.fromarray(array.astype(np.uint8))
    return _image_tensor(image)


def _load_boxes(annotation_path: Path, width: int, height: int):
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    boxes: list[list[float]] = []
    labels: list[int] = []
    for frame in data.get("frames", []):
        for obj in frame.get("objects", []):
            category = obj.get("category")
            box = obj.get("box2d")
            if category not in DETECTION_CLASS_TO_ID or not isinstance(box, dict):
                continue
            try:
                values = [float(box[k]) for k in ("x1", "y1", "x2", "y2")]
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in values):
                continue
            x1, y1, x2, y2 = values
            x1, x2 = min(max(x1, 0.0), float(width)), min(max(x2, 0.0), float(width))
            y1, y2 = min(max(y1, 0.0), float(height)), min(max(y2, 0.0), float(height))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(DETECTION_CLASS_TO_ID[category])
    return boxes, labels, data


class BDD100KDetectionDataset(Dataset):
    """Load matched 10K images with per-image 2018 2D detection annotations.

    ``split`` is ``train``, ``val`` or ``test``. Test lives in its own manifest
    and is never merged with train/validation. Targets use pixel ``xyxy`` boxes.
    A custom transform receives ``image``, ``bboxes`` and ``labels``.
    """

    def __init__(
        self,
        split: str = "train",
        manifest_path: str | Path | None = None,
        project_root: str | Path = PROJECT_ROOT,
        image_size: tuple[int, int] | None = (448, 800),
        transform: Optional[Callable] = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val' or 'test'")
        self.project_root = Path(project_root).resolve()
        self.split = split
        self.image_size = image_size
        self.transform = transform
        path = Path(manifest_path) if manifest_path else DEFAULT_PROCESSED_ROOT / f"detection_{split}.csv"
        if not path.is_absolute():
            path = self.project_root / path
        self.records = _read_manifest(path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        image_path = _resolve_path(row["image_path"], self.project_root)
        annotation_path = _resolve_path(row["annotation_path"], self.project_root)
        with Image.open(image_path) as source:
            image = normalize_bdd100k_image(source)
        width, height = image.size
        boxes, labels, annotation = _load_boxes(annotation_path, width, height)
        expected_id = row.get("image_id", "")
        annotation_name = annotation.get("name", "")
        if annotation_name and expected_id and annotation_name != expected_id:
            raise ValueError(f"Annotation ID {annotation_name!r} does not match manifest ID {expected_id!r}")
        if self.transform is not None:
            transformed = self.transform(image=np.asarray(image), bboxes=boxes, labels=labels)
            image_tensor = _as_image_tensor(transformed["image"])
            boxes = [list(map(float, b)) for b in transformed.get("bboxes", boxes)]
            labels = list(map(int, transformed.get("labels", labels)))
        else:
            if self.image_size is not None:
                out_h, out_w = self.image_size
                image = image.resize((out_w, out_h), Image.Resampling.BILINEAR)
                sx, sy = out_w / width, out_h / height
                boxes = [[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for x1, y1, x2, y2 in boxes]
            image_tensor = _image_tensor(image)
        return image_tensor, {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.long).reshape(-1),
            "image_id": expected_id,
            "meta": {
                "image_id": expected_id,
                "image_split": row.get("image_split", ""),
                "label_split": row.get("label_split", ""),
                "image_path": str(image_path),
                "annotation_path": str(annotation_path),
                "annotation_name": annotation_name,
            },
        }


class BDD100KSegmentationDataset(Dataset):
    """Load an official train/val pair and its mapped five-class train-ID mask."""

    def __init__(
        self,
        split: str = "train",
        manifest_path: str | Path | None = None,
        project_root: str | Path = PROJECT_ROOT,
        image_size: tuple[int, int] | None = (448, 800),
        transform: Optional[Callable] = None,
    ):
        if split not in {"train", "val"}:
            raise ValueError("BDD100K semantic masks are available only for train and val")
        self.project_root = Path(project_root).resolve()
        self.split = split
        self.image_size = image_size
        self.transform = transform
        path = Path(manifest_path) if manifest_path else DEFAULT_PROCESSED_ROOT / f"segmentation_{split}.csv"
        if not path.is_absolute():
            path = self.project_root / path
        self.records = _read_manifest(path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        image_path = _resolve_path(row["image_path"], self.project_root)
        mask_path = _resolve_path(row["train_id_mask_path"], self.project_root)
        with Image.open(image_path) as source:
            image = normalize_bdd100k_image(source)
        raw = read_train_id_mask(mask_path)
        if tuple(raw.shape[::-1]) != image.size:
            raise ValueError(
                f"Image/mask spatial mismatch for {row.get('image_id', index)}: "
                f"image {image.size}, mask {(raw.shape[1], raw.shape[0])}"
            )
        mapped = map_train_id_mask(raw)
        if self.transform is not None:
            transformed = self.transform(image=np.asarray(image), mask=mapped)
            image_tensor = _as_image_tensor(transformed["image"])
            mask_value = transformed["mask"]
            mask_tensor = mask_value.long() if torch.is_tensor(mask_value) else torch.as_tensor(np.asarray(mask_value).copy(), dtype=torch.long)
        else:
            if self.image_size is not None:
                out_h, out_w = self.image_size
                image = image.resize((out_w, out_h), Image.Resampling.BILINEAR)
                mapped_image = Image.fromarray(mapped, mode="L")
                mapped = np.asarray(mapped_image.resize((out_w, out_h), Image.Resampling.NEAREST), dtype=np.uint8)
            image_tensor = _image_tensor(image)
            mask_tensor = torch.from_numpy(np.ascontiguousarray(mapped.astype(np.int64)))
        if tuple(image_tensor.shape[-2:]) != tuple(mask_tensor.shape[-2:]):
            raise ValueError(f"Transform returned misaligned image/mask shapes: {image_tensor.shape}, {mask_tensor.shape}")
        # This is a CPU dataset transform. NumPy's membership test avoids the
        # relatively expensive torch.unique sort/kernel launch for every mask.
        valid_ids = np.asarray([*range(len(SEGMENTATION_CLASS_NAMES)), SEGMENTATION_IGNORE_INDEX], dtype=np.int64)
        mask_values = mask_tensor.detach().cpu().numpy()
        invalid_values = mask_values[~np.isin(mask_values, valid_ids)]
        if invalid_values.size:
            invalid_ids = sorted(int(v) for v in np.unique(invalid_values))
            raise ValueError(f"Mapped mask contains invalid class IDs: {invalid_ids}")
        return image_tensor, mask_tensor


def detection_collate_fn(batch):
    """Stack same-sized images while leaving variable-length targets as a list."""
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]


def segmentation_collate_fn(batch):
    """Stack image and semantic-mask tensors into batches."""
    return torch.stack([item[0] for item in batch]), torch.stack([item[1] for item in batch])


def build_bdd100k_dataloaders(
    processed_root: str | Path = DEFAULT_PROCESSED_ROOT,
    project_root: str | Path = PROJECT_ROOT,
    batch_size: int = 4,
    image_size: tuple[int, int] | None = (448, 800),
    num_workers: int = 0,
    seed: int = 42,
):
    """Return detection {train,val,test} and segmentation {train,val} loaders.

    Test is a separate non-shuffled loader. Before returning, this function
    rejects within-task or cross-task train/evaluation leakage and any combined
    validation/test overlap. This only constructs loaders; it starts no training.
    """
    processed_root = Path(processed_root)
    if not processed_root.is_absolute():
        processed_root = Path(project_root) / processed_root
    root = Path(project_root)
    detection = {
        split: BDD100KDetectionDataset(
            split=split, manifest_path=processed_root / f"detection_{split}.csv",
            project_root=root, image_size=image_size,
        ) for split in ("train", "val", "test")
    }
    segmentation = {
        split: BDD100KSegmentationDataset(
            split=split, manifest_path=processed_root / f"segmentation_{split}.csv",
            project_root=root, image_size=image_size,
        ) for split in ("train", "val")
    }
    audit_multitask_split_ids(
        {split: [row["image_id"] for row in dataset.records]
         for split, dataset in detection.items()},
        {split: [row["image_id"] for row in dataset.records]
         for split, dataset in segmentation.items()},
        reject_leakage=True,
    )
    generator = torch.Generator().manual_seed(seed)
    return {
        "detection": {
            split: DataLoader(
                ds, batch_size=batch_size, shuffle=(split == "train"),
                generator=generator if split == "train" else None,
                num_workers=num_workers, collate_fn=detection_collate_fn,
            ) for split, ds in detection.items()
        },
        "segmentation": {
            split: DataLoader(
                ds, batch_size=batch_size, shuffle=(split == "train"),
                generator=generator if split == "train" else None,
                num_workers=num_workers, collate_fn=segmentation_collate_fn,
            ) for split, ds in segmentation.items()
        },
    }
