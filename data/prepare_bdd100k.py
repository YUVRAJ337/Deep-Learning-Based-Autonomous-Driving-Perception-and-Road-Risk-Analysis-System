"""Prepare reproducible BDD100K manifests, audit reports and alignment previews.

From the main project root run ``python3 -m data.prepare_bdd100k``. By default,
this reads the already-extracted sibling trees ../10k, ../100k and
../bdd100k_seg_maps, and writes only under data/processed/bdd100k/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from data.bdd100k_dataset import (
    DEFAULT_PROCESSED_ROOT,
    DETECTION_CLASS_NAMES,
    DETECTION_CLASS_TO_ID,
    SEGMENTATION_CLASS_NAMES,
    SEGMENTATION_IGNORE_INDEX,
    TRAIN_ID_NAMES,
    TRAIN_ID_TO_CLASS_ID,
    assert_no_split_leakage,
    audit_multitask_split_ids,
    map_train_id_mask,
    normalize_bdd100k_image,
    read_train_id_mask,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES_ROOT = PROJECT_ROOT.parent / "10k"
DEFAULT_LABELS_ROOT = PROJECT_ROOT.parent / "100k"
DEFAULT_SEGMENTATION_ROOT = PROJECT_ROOT.parent / "bdd100k_seg_maps"
DEFAULT_IMAGE_MD5_REFERENCE = "08f26aecceda982568063d3d5873378e"
IMAGE_SPLITS = ("train", "val", "test")
SEGMENTATION_SPLITS = ("train", "val")


def split_training_ids(ids, seed: int = 42, val_fraction: float = 0.2):
    """Deterministically split IDs from the 10K train set; test IDs are never input."""
    source = sorted(set(ids))
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    if len(source) < 2:
        raise ValueError("At least two matched train IDs are required to make a split")
    shuffled = source.copy()
    random.Random(seed).shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_fraction))
    n_val = min(max(n_val, 1), len(shuffled) - 1)
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    assert_no_split_leakage({"train": train_ids, "val": val_ids})
    if set(train_ids) | set(val_ids) != set(source):
        raise AssertionError("Detection train/validation split lost IDs")
    return train_ids, val_ids


def _relative(path: Path, root: Path = PROJECT_ROOT) -> str:
    return Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()


def _index_files(root: Path, splits: tuple[str, ...], suffix: str):
    result: dict[str, dict[str, Path]] = {}
    for split in splits:
        directory = root / split
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing source split directory: {directory}")
        indexed: dict[str, Path] = {}
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() != suffix:
                continue
            ident = path.name[:-len(suffix)]
            if ident in indexed:
                raise ValueError(f"Duplicate {suffix} source ID {ident} in {directory}")
            indexed[ident] = path.resolve()
        result[split] = indexed
    return result


def _images_by_split(root: Path):
    return _index_files(root, IMAGE_SPLITS, ".jpg")


def _labels_by_split(root: Path):
    result = _index_files(root, IMAGE_SPLITS, ".json")
    union: dict[str, Path] = {}
    split_for: dict[str, str] = {}
    duplicates: list[str] = []
    for split, entries in result.items():
        for ident, path in entries.items():
            if ident in union:
                duplicates.append(ident)
            else:
                union[ident] = path
                split_for[ident] = split
    if duplicates:
        raise ValueError(f"Duplicate detection label IDs across splits: {duplicates[:10]}")
    return result, union, split_for


def _parse_detection_annotation(path: Path, width: int, height: int):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("frames"), list):
        raise ValueError(f"Invalid detection JSON structure (expected frames list): {path}")
    counts: Counter[str] = Counter()
    boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    unknown: Counter[str] = Counter()
    invalid_boxes = 0
    for frame in data["frames"]:
        if not isinstance(frame, dict):
            invalid_boxes += 1
            continue
        for obj in frame.get("objects", []):
            if not isinstance(obj, dict):
                invalid_boxes += 1
                continue
            box = obj.get("box2d")
            if not isinstance(box, dict):
                continue  # Polygon-only area/lane objects are not detection boxes.
            category = str(obj.get("category", "<missing>"))
            if category not in DETECTION_CLASS_TO_ID:
                unknown[category] += 1
                continue
            try:
                values = [float(box[key]) for key in ("x1", "y1", "x2", "y2")]
            except (KeyError, TypeError, ValueError):
                invalid_boxes += 1
                continue
            if not all(math.isfinite(value) for value in values):
                invalid_boxes += 1
                continue
            x1, y1, x2, y2 = values
            x1, x2 = min(max(x1, 0.0), float(width)), min(max(x2, 0.0), float(width))
            y1, y2 = min(max(y1, 0.0), float(height)), min(max(y2, 0.0), float(height))
            if x2 <= x1 or y2 <= y1:
                invalid_boxes += 1
                continue
            boxes.append((category, (x1, y1, x2, y2)))
            counts[category] += 1
    return data, boxes, counts, unknown, invalid_boxes


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _file_md5(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _palette(mapped: np.ndarray) -> np.ndarray:
    colors = np.asarray([
        (28, 28, 28),       # background
        (128, 64, 128),     # road
        (244, 35, 232),     # sidewalk
        (0, 0, 142),        # vehicles
        (220, 20, 60),      # vulnerable road users
    ], dtype=np.uint8)
    result = np.zeros((*mapped.shape, 3), dtype=np.uint8)
    valid = mapped != SEGMENTATION_IGNORE_INDEX
    result[valid] = colors[mapped[valid]]
    result[~valid] = (245, 210, 30)  # visible void/ignore pixels
    return result


def _save_alignment_sample(image_path: Path, annotation_boxes, mask_path: Path, image_id: str, output: Path):
    with Image.open(image_path) as source:
        image = normalize_bdd100k_image(source)
    raw = read_train_id_mask(mask_path)
    if (raw.shape[1], raw.shape[0]) != image.size:
        raise ValueError(f"Cannot visualize misaligned sample {image_id}: image {image.size}, mask {(raw.shape[1], raw.shape[0])}")
    mapped = map_train_id_mask(raw)
    mask_rgb = Image.fromarray(_palette(mapped), mode="RGB")
    overlay = Image.blend(image, mask_rgb, 0.42)
    boxes_image = image.copy()
    draw = ImageDraw.Draw(boxes_image)
    colorwheel = ((255, 70, 70), (255, 180, 0), (40, 210, 255), (160, 90, 255), (255, 80, 190))
    for category, coords in annotation_boxes:
        color = colorwheel[DETECTION_CLASS_TO_ID[category] % len(colorwheel)]
        draw.rectangle(coords, outline=color, width=3)
        draw.text((coords[0] + 2, max(0, coords[1] - 14)), category, fill=color, stroke_width=1, stroke_fill=(0, 0, 0))
    panel_size = (640, 360)
    panels = [image, boxes_image, mask_rgb, overlay]
    captions = ["Original image (orientation normalized)", f"Detection boxes ({len(annotation_boxes)} boxes)", "Mapped 5-class mask", f"Mask overlay | {image_id}"]
    canvas = Image.new("RGB", (1280, 770), (245, 245, 245))
    d = ImageDraw.Draw(canvas)
    for idx, (panel, caption) in enumerate(zip(panels, captions)):
        x = (idx % 2) * 640
        y = (idx // 2) * 385
        d.rectangle((x, y, x + 640, y + 24), fill=(22, 29, 39))
        d.text((x + 8, y + 5), caption, fill=(255, 255, 255))
        canvas.paste(panel.resize(panel_size, Image.Resampling.BILINEAR if idx != 2 else Image.Resampling.NEAREST), (x, y + 25))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94)
    return {"image_id": image_id, "path": _relative(output), "image_size_wh": list(image.size),
            "mask_size_wh": [int(raw.shape[1]), int(raw.shape[0])], "same_id": True,
            "same_size": True, "box_count": len(annotation_boxes)}


def prepare(
    images_root: Path = DEFAULT_IMAGES_ROOT,
    labels_root: Path = DEFAULT_LABELS_ROOT,
    segmentation_root: Path = DEFAULT_SEGMENTATION_ROOT,
    output_root: Path = DEFAULT_PROCESSED_ROOT,
    seed: int = 42,
    val_fraction: float = 0.2,
) -> dict[str, Any]:
    """Audit source files and write data manifests, counts and visual alignment checks."""
    images_root, labels_root = images_root.resolve(), labels_root.resolve()
    segmentation_root, output_root = segmentation_root.resolve(), output_root.resolve()
    for name, path in (("10K images", images_root), ("2018 labels", labels_root), ("segmentation masks", segmentation_root)):
        if not path.is_dir():
            raise FileNotFoundError(f"{name} source directory not found: {path}")
    output_root.mkdir(parents=True, exist_ok=True)
    images = _images_by_split(images_root)
    labels, label_union, label_split = _labels_by_split(labels_root)

    # Decode every 10K image once, normalize EXIF/portrait orientation, and record
    # corrupt files before constructing any manifests.
    image_info: dict[tuple[str, str], dict[str, Any]] = {}
    corrupt_rows: list[dict[str, str]] = []
    for split, entries in images.items():
        for ident, path in entries.items():
            try:
                with Image.open(path) as source:
                    raw_size = source.size
                    normalized = normalize_bdd100k_image(source)
                    normalized.load()
                    size = normalized.size
                image_info[(split, ident)] = {"size": size, "raw_size": raw_size, "rotated": raw_size != size}
            except Exception as exc:
                image_info[(split, ident)] = {"error": str(exc)}
                corrupt_rows.append({"kind": "image_decode_error", "split": split, "image_id": ident,
                                     "path": str(path), "error": repr(exc)})

    overlaps = {split: sorted(set(images[split]) & set(label_union)) for split in IMAGE_SPLITS}
    train_ids, val_ids = split_training_ids(overlaps["train"], seed, val_fraction)
    test_ids = overlaps["test"]
    unmatched_rows = [
        {"image_id": ident, "image_split": split, "image_path": _relative(path),
         "reason": "No 2018 detection JSON with this image ID in train/val/test"}
        for split in IMAGE_SPLITS
        for ident, path in sorted(images[split].items()) if ident not in label_union
    ]
    assigned = {"train": train_ids, "val": val_ids, "test": test_ids}
    assert_no_split_leakage(assigned)
    if set(assigned["train"]) | set(assigned["val"]) != set(overlaps["train"]):
        raise AssertionError("Detection train/val IDs do not equal the matched 10K train IDs")
    if set(assigned["test"]) & (set(assigned["train"]) | set(assigned["val"])):
        raise AssertionError("Detection test IDs leaked into train/validation")
    if any(label_split[i] != "test" for i in test_ids):
        raise AssertionError("A 10K test image matched a non-test 2018 label")

    parsed: dict[str, dict[str, Any]] = {}
    total_unknown: Counter[str] = Counter()
    invalid_box_counts: Counter[str] = Counter()
    name_mismatch_ids: list[str] = []
    required_corrupt_ids: list[str] = []
    for ident in sorted(set(overlaps["train"]) | set(test_ids)):
        image_split = "test" if ident in set(test_ids) else "train"
        info = image_info.get((image_split, ident), {})
        if "error" in info:
            required_corrupt_ids.append(ident)
            continue
        ann_path = label_union[ident]
        try:
            data, boxes, class_counts, unknown, invalid = _parse_detection_annotation(ann_path, *info["size"])
        except Exception as exc:
            required_corrupt_ids.append(ident)
            corrupt_rows.append({"kind": "detection_json_error", "split": label_split[ident], "image_id": ident,
                                 "path": str(ann_path), "error": repr(exc)})
            continue
        if data.get("name") != ident:
            name_mismatch_ids.append(ident)
        total_unknown.update(unknown)
        invalid_box_counts[ident] = invalid
        parsed[ident] = {"boxes": boxes, "class_counts": class_counts, "label_split": label_split[ident],
                         "annotation_name": data.get("name", "")}
    if required_corrupt_ids:
        raise RuntimeError(f"Cannot prepare matched detection samples with corrupt image/label data: {required_corrupt_ids[:10]}")

    det_rows: dict[str, list[dict[str, Any]]] = {s: [] for s in ("train", "val", "test")}
    class_boxes: dict[str, Counter[str]] = {s: Counter() for s in det_rows}
    class_images: dict[str, Counter[str]] = {s: Counter() for s in det_rows}
    empty_detection_ids: dict[str, list[str]] = {s: [] for s in det_rows}
    for split, ids in assigned.items():
        for ident in sorted(ids):
            image_split = "test" if split == "test" else "train"
            info = image_info[(image_split, ident)]
            image_path, annotation_path = images[image_split][ident], label_union[ident]
            parsed_info = parsed[ident]
            counts: Counter[str] = parsed_info["class_counts"]
            class_boxes[split].update(counts)
            class_images[split].update(counts.keys())
            box_count = sum(counts.values())
            if box_count == 0:
                empty_detection_ids[split].append(ident)
            det_rows[split].append({
                "image_id": ident, "split": split, "image_split": image_split,
                "label_split": parsed_info["label_split"], "image_path": _relative(image_path),
                "annotation_path": _relative(annotation_path), "box_count": box_count,
                "invalid_box_count": invalid_box_counts[ident],
                "image_orientation_normalized": int(info["rotated"]),
                "class_counts": json.dumps(dict(sorted(counts.items())), sort_keys=True),
            })
    det_fields = ["image_id", "split", "image_split", "label_split", "image_path", "annotation_path",
                  "box_count", "invalid_box_count", "image_orientation_normalized", "class_counts"]
    for split, rows in det_rows.items():
        _write_csv(output_root / f"detection_{split}.csv", rows, det_fields)
    _write_csv(output_root / "detection_unmatched_images.csv", unmatched_rows,
               ["image_id", "image_split", "image_path", "reason"])

    # Audit both complete official semantic train/val source partitions. Normalize
    # portrait JPEGs to mask orientation; combined evaluation IDs are filtered
    # from the working train manifest after all official pairs are audited.
    seg_rows: dict[str, list[dict[str, Any]]] = {s: [] for s in SEGMENTATION_SPLITS}
    raw_pixels: dict[str, Counter[int]] = {s: Counter() for s in SEGMENTATION_SPLITS}
    mapped_pixels: dict[str, Counter[int]] = {s: Counter() for s in SEGMENTATION_SPLITS}
    raw_pixels_by_id: dict[str, dict[str, Counter[int]]] = {s: {} for s in SEGMENTATION_SPLITS}
    mapped_pixels_by_id: dict[str, dict[str, Counter[int]]] = {s: {} for s in SEGMENTATION_SPLITS}
    unexpected_train_ids: Counter[int] = Counter()
    missing_rows: list[dict[str, str]] = []
    unusable_rows: list[dict[str, str]] = []
    color_mask_counts: dict[str, int] = {}
    segmentation_root_label = segmentation_root / "labels"
    for split in SEGMENTATION_SPLITS:
        train_dir = segmentation_root_label / split
        color_dir = segmentation_root / "color_labels" / split
        train_map = {p.name[:-len("_train_id.png")]: p.resolve() for p in train_dir.glob("*_train_id.png") if p.is_file()}
        color_map = {p.name[:-len("_train_color.png")]: p.resolve() for p in color_dir.glob("*_train_color.png") if p.is_file()}
        color_mask_counts[split] = len(color_map)
        for ident, image_path in sorted(images[split].items()):
            issues: list[str] = []
            info = image_info.get((split, ident), {})
            mask_path, color_path = train_map.get(ident), color_map.get(ident)
            if "error" in info:
                issues.append("image_decode_error")
            if not mask_path or not mask_path.is_file():
                issues.append("train_id_mask_missing")
                missing_rows.append({"kind": "train_id_mask_missing", "split": split, "image_id": ident,
                                     "path": str(train_dir / f"{ident}_train_id.png")})
            elif "error" not in info:
                try:
                    raw = read_train_id_mask(mask_path)
                    with Image.open(mask_path) as mask_image:
                        mask_image.load()
                    if (raw.shape[1], raw.shape[0]) != tuple(info["size"]):
                        issues.append("image_mask_dimension_mismatch")
                    else:
                        vals, counts = np.unique(raw, return_counts=True)
                        raw_counts = Counter({int(v): int(c) for v, c in zip(vals, counts)})
                        raw_pixels[split].update(raw_counts)
                        raw_pixels_by_id[split][ident] = raw_counts
                        unexpected = set(map(int, vals)) - set(range(19)) - {SEGMENTATION_IGNORE_INDEX}
                        for value in unexpected:
                            unexpected_train_ids[value] += int((raw == value).sum())
                        mapped = map_train_id_mask(raw)
                        vals5, counts5 = np.unique(mapped, return_counts=True)
                        mapped_counts = Counter({int(v): int(c) for v, c in zip(vals5, counts5)})
                        mapped_pixels[split].update(mapped_counts)
                        mapped_pixels_by_id[split][ident] = mapped_counts
                except Exception as exc:
                    issues.append("train_id_mask_decode_error")
                    corrupt_rows.append({"kind": "train_id_mask_decode_error", "split": split, "image_id": ident,
                                         "path": str(mask_path), "error": repr(exc)})
            if not color_path or not color_path.is_file():
                missing_rows.append({"kind": "color_mask_missing", "split": split, "image_id": ident,
                                     "path": str(color_dir / f"{ident}_train_color.png")})
            else:
                try:
                    with Image.open(color_path) as color_image:
                        color_image.load()
                        if "error" not in info and color_image.size != tuple(info["size"]):
                            missing_rows.append({"kind": "color_mask_dimension_mismatch", "split": split,
                                                 "image_id": ident, "path": str(color_path)})
                except Exception as exc:
                    corrupt_rows.append({"kind": "color_mask_decode_error", "split": split, "image_id": ident,
                                         "path": str(color_path), "error": repr(exc)})
            if issues:
                unusable_rows.append({"split": split, "image_id": ident, "issues": ";".join(sorted(set(issues))),
                                      "image_path": _relative(image_path),
                                      "train_id_mask_path": _relative(mask_path) if mask_path else ""})
                continue
            seg_rows[split].append({
                "image_id": ident, "split": split, "image_path": _relative(image_path),
                "train_id_mask_path": _relative(mask_path), "color_mask_path": _relative(color_path) if color_path else "",
                "image_orientation_normalized": int(info["rotated"]),
            })

    # Retain the full official train manifest for provenance, but the working
    # segmentation-train set excludes every combined evaluation ID. This keeps
    # the official validation partition intact and the 830 detection-test IDs
    # held out from both heads.
    official_segmentation_train_rows = list(seg_rows["train"])
    official_raw_pixels = {split: Counter(counts) for split, counts in raw_pixels.items()}
    official_mapped_pixels = {split: Counter(counts) for split, counts in mapped_pixels.items()}
    segmentation_evaluation_ids = (
        {row["image_id"] for row in det_rows["val"]}
        | {row["image_id"] for row in det_rows["test"]}
        | {row["image_id"] for row in seg_rows["val"]}
    )
    segmentation_train_excluded_rows = []
    working_segmentation_train_rows = []
    for row in official_segmentation_train_rows:
        ident = row["image_id"]
        if ident not in segmentation_evaluation_ids:
            working_segmentation_train_rows.append(row)
            continue
        reasons = []
        if ident in {item["image_id"] for item in det_rows["val"]}:
            reasons.append("detection_validation")
        if ident in {item["image_id"] for item in det_rows["test"]}:
            reasons.append("detection_test")
        if ident in {item["image_id"] for item in seg_rows["val"]}:
            reasons.append("segmentation_validation")
        segmentation_train_excluded_rows.append({**row, "exclusion_reason": ";".join(reasons)})
        for source, by_id in ((raw_pixels["train"], raw_pixels_by_id["train"]),
                              (mapped_pixels["train"], mapped_pixels_by_id["train"])):
            for class_id, pixel_count in by_id.get(ident, {}).items():
                source[class_id] -= pixel_count
                if source[class_id] == 0:
                    del source[class_id]
    seg_rows["train"] = working_segmentation_train_rows

    if unexpected_train_ids:
        # Unknown values are mapped to 255 by contract, but call them out in summary.
        pass
    seg_id_splits = {split: [row["image_id"] for row in rows] for split, rows in seg_rows.items()}
    assert_no_split_leakage(seg_id_splits)
    seg_fields = ["image_id", "split", "image_path", "train_id_mask_path", "color_mask_path", "image_orientation_normalized"]
    _write_csv(output_root / "segmentation_official_train.csv", official_segmentation_train_rows, seg_fields)
    _write_csv(output_root / "segmentation_train_excluded_evaluation.csv", segmentation_train_excluded_rows,
               seg_fields + ["exclusion_reason"])
    for split, rows in seg_rows.items():
        _write_csv(output_root / f"segmentation_{split}.csv", rows, seg_fields)
    segmentation_ids_for_audit = {split: [row["image_id"] for row in rows] for split, rows in seg_rows.items()}
    segmentation_ids_for_audit["test"] = []
    cross_task_split_audit = audit_multitask_split_ids(
        {split: [row["image_id"] for row in rows] for split, rows in det_rows.items()},
        segmentation_ids_for_audit,
        reject_leakage=True,
    )

    _write_csv(output_root / "missing_files.csv", missing_rows, ["kind", "split", "image_id", "path"])
    _write_csv(output_root / "corrupt_files.csv", corrupt_rows, ["kind", "split", "image_id", "path", "error"])
    _write_csv(output_root / "unusable_segmentation.csv", unusable_rows,
               ["split", "image_id", "issues", "image_path", "train_id_mask_path"])
    class_rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        for class_id, name in enumerate(DETECTION_CLASS_NAMES):
            class_rows.append({"task": "detection", "split": split, "class_id": class_id, "class_name": name,
                               "count": class_boxes[split][name], "unit": "boxes"})
            class_rows.append({"task": "detection", "split": split, "class_id": class_id, "class_name": name,
                               "count": class_images[split][name], "unit": "images_with_class"})
    for split in SEGMENTATION_SPLITS:
        for class_id, name in enumerate(SEGMENTATION_CLASS_NAMES):
            class_rows.append({"task": "segmentation", "split": split, "class_id": class_id,
                               "class_name": name, "count": mapped_pixels[split][class_id], "unit": "pixels"})
        class_rows.append({"task": "segmentation", "split": split, "class_id": 255,
                           "class_name": "void/ignore", "count": mapped_pixels[split][255], "unit": "pixels"})
        for train_id, name in enumerate(TRAIN_ID_NAMES):
            class_rows.append({"task": "segmentation_train_id", "split": split, "class_id": train_id,
                               "class_name": name, "count": raw_pixels[split][train_id], "unit": "pixels"})
    _write_csv(output_root / "class_distribution.csv", class_rows,
               ["task", "split", "class_id", "class_name", "count", "unit"])

    raw_root = PROJECT_ROOT / "data" / "raw" / "bdd100k"
    archive_summary = {}
    for filename in ("bdd100k_images_10k.zip", "bdd100k_labels.zip", "bdd100k_seg_maps.zip"):
        path = raw_root / filename
        archive_summary[filename] = {"exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else None,
                                     "md5": _file_md5(path)}
    image_md5 = archive_summary["bdd100k_images_10k.zip"]["md5"]

    # Pick a same-ID image/mask/label triplet in the detection training pool.
    seg_train_ids = {row["image_id"] for row in seg_rows["train"]}
    visual = None
    for ident in sorted(set(train_ids) & seg_train_ids):
        info = parsed[ident]
        if not info["boxes"]:
            continue
        image_path = images["train"][ident]
        mask_path = segmentation_root_label / "train" / f"{ident}_train_id.png"
        visual = _save_alignment_sample(image_path, info["boxes"], mask_path, ident,
                                        output_root / "visualizations" / f"alignment_{ident}.jpg")
        break

    expected_unlabeled_by_split = Counter(row["image_split"] for row in unmatched_rows)
    summary: dict[str, Any] = {
        "dataset": "BDD100K 10K images + 2018 detection labels + semantic train-ID masks",
        "source_directories": {"images_10k": str(images_root), "detection_labels_2018": str(labels_root),
                               "segmentation_masks": str(segmentation_root)},
        "output_directory": _relative(output_root),
        "archives": archive_summary,
        "image_md5_reference": DEFAULT_IMAGE_MD5_REFERENCE,
        "image_md5_matches_reference": image_md5 == DEFAULT_IMAGE_MD5_REFERENCE,
        "image_orientation_normalized_counts": {split: sum(bool(i.get("rotated")) for (s, _), i in image_info.items() if s == split) for split in IMAGE_SPLITS},
        "source_image_counts_by_official_split": {s: len(images[s]) for s in IMAGE_SPLITS},
        "source_label_json_counts_by_2018_split": {s: len(labels[s]) for s in IMAGE_SPLITS},
        "detection": {
            "class_names_by_id": {str(i): name for i, name in enumerate(DETECTION_CLASS_NAMES)},
            "box_class_names_seen": sorted({name for info in parsed.values() for name in info["class_counts"]}),
            "matched_ids_by_10k_image_split": {s: len(overlaps[s]) for s in IMAGE_SPLITS},
            "unmatched_10k_image_counts_by_split": {s: expected_unlabeled_by_split[s] for s in IMAGE_SPLITS},
            "matched_training_ids_by_2018_label_split": dict(sorted(Counter(label_split[i] for i in overlaps["train"]).items())),
            "prepared_split_counts": {s: len(det_rows[s]) for s in det_rows},
            "split_seed": seed, "validation_fraction": val_fraction,
            "class_box_counts": {s: {name: class_boxes[s][name] for name in DETECTION_CLASS_NAMES} for s in det_rows},
            "class_image_counts": {s: {name: class_images[s][name] for name in DETECTION_CLASS_NAMES} for s in det_rows},
            "unknown_box_categories": dict(sorted(total_unknown.items())),
            "invalid_or_degenerate_boxes": sum(invalid_box_counts.values()),
            "empty_detection_samples_by_split": {s: len(empty_detection_ids[s]) for s in det_rows},
            "empty_detection_sample_ids": empty_detection_ids,
            "annotation_name_mismatches": name_mismatch_ids,
            "test_ids_all_from_10k_and_2018_test": all(label_split[i] == "test" for i in test_ids),
            "split_leakage": {"train_val": 0, "train_test": 0, "val_test": 0},
        },
        "segmentation": {
            "class_names_by_id": {str(i): name for i, name in enumerate(SEGMENTATION_CLASS_NAMES)},
            "official_image_counts": {s: len(images[s]) for s in SEGMENTATION_SPLITS},
            "official_aligned_image_counts": {
                "train": len(official_segmentation_train_rows), "val": len(seg_rows["val"])
            },
            "prepared_aligned_image_counts": {s: len(seg_rows[s]) for s in SEGMENTATION_SPLITS},
            "excluded_official_train_ids_for_combined_evaluation": len(segmentation_train_excluded_rows),
            "official_full_train_mapped_pixel_counts": {
                str(i): official_mapped_pixels["train"][i] for i in range(5)
            } | {"255": official_mapped_pixels["train"][SEGMENTATION_IGNORE_INDEX]},
            "train_id_names_by_id": {str(i): name for i, name in enumerate(TRAIN_ID_NAMES)},
            "train_id_to_class_id": {str(k): v for k, v in sorted(TRAIN_ID_TO_CLASS_ID.items())},
            "ignore_index": SEGMENTATION_IGNORE_INDEX,
            "raw_train_id_pixel_counts": {s: {str(i): raw_pixels[s][i] for i in range(19)} | {"255": raw_pixels[s][255]} for s in SEGMENTATION_SPLITS},
            "mapped_pixel_counts": {s: {str(i): mapped_pixels[s][i] for i in range(5)} | {"255": mapped_pixels[s][SEGMENTATION_IGNORE_INDEX]} for s in SEGMENTATION_SPLITS},
            "unexpected_train_id_pixel_counts": dict(sorted(unexpected_train_ids.items())),
            "official_train_val_id_intersection": 0,
            "color_mask_file_counts": color_mask_counts,
        },
        "combined_split_audit": cross_task_split_audit,
        "file_audit": {
            "missing_source_file_rows": len(missing_rows),
            "missing_by_kind": dict(sorted(Counter(row["kind"] for row in missing_rows).items())),
            "corrupt_source_file_rows": len(corrupt_rows),
            "corrupt_by_kind": dict(sorted(Counter(row["kind"] for row in corrupt_rows).items())),
            "unusable_segmentation_pairs": len(unusable_rows),
            "unusable_segmentation_by_issue": dict(sorted(Counter(issue for row in unusable_rows for issue in row["issues"].split(";")).items())),
            "unmatched_detection_images_are_expected_subset_gap": len(unmatched_rows),
        },
        "visual_samples": {"combined_alignment": visual},
        "manifests": [f"detection_{s}.csv" for s in det_rows]
                      + [f"segmentation_{s}.csv" for s in seg_rows]
                      + ["segmentation_official_train.csv", "segmentation_train_excluded_evaluation.csv"],
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", default=str(DEFAULT_IMAGES_ROOT))
    parser.add_argument("--labels-root", default=str(DEFAULT_LABELS_ROOT))
    parser.add_argument("--segmentation-root", default=str(DEFAULT_SEGMENTATION_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_PROCESSED_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()
    summary = prepare(Path(args.images_root), Path(args.labels_root), Path(args.segmentation_root),
                      Path(args.output_root), args.seed, args.val_fraction)
    print(json.dumps({
        "output": summary["output_directory"],
        "detection_prepared_splits": summary["detection"]["prepared_split_counts"],
        "segmentation_prepared_splits": summary["segmentation"]["prepared_aligned_image_counts"],
        "unmatched_detection_images": summary["file_audit"]["unmatched_detection_images_are_expected_subset_gap"],
        "missing_files": summary["file_audit"]["missing_source_file_rows"],
        "corrupt_files": summary["file_audit"]["corrupt_source_file_rows"],
        "unusable_segmentation_pairs": summary["file_audit"]["unusable_segmentation_pairs"],
        "image_md5_matches_reference": summary["image_md5_matches_reference"],
    }, indent=2))


if __name__ == "__main__":
    main()
