#!/usr/bin/env python3
"""Bounded-memory sampled validation and profiling for prepared BDD100K loaders.

Example (from project root):
    python3 tools/bdd100k_runtime_audit.py --samples-per-split 64 --batch-size 8 --workers 0 2

This validates only seeded samples; it never materializes a whole dataset.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import math
import pstats
import random
import resource
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.bdd100k_dataset import (
    BDD100KDetectionDataset,
    BDD100KSegmentationDataset,
    DETECTION_CLASS_NAMES,
    SEGMENTATION_CLASS_NAMES,
    detection_collate_fn,
    segmentation_collate_fn,
)

OUTPUT = PROJECT_ROOT / "data/processed/bdd100k/runtime_validation.json"


def rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(value / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


def profile_items(dataset, indexes: list[int], label: str) -> dict:
    profiler = cProfile.Profile()
    durations = []
    for index in indexes:
        start = time.perf_counter()
        profiler.runcall(dataset.__getitem__, index)
        durations.append(time.perf_counter() - start)
    stats = pstats.Stats(profiler)
    top = sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:12]
    rows = []
    for (filename, line, function), (primitive, calls, self_s, cumulative_s, callers) in top:
        rows.append({
            "function": f"{Path(filename).name}:{line}:{function}",
            "calls": calls,
            "self_seconds": round(self_s, 4),
            "cumulative_seconds": round(cumulative_s, 4),
        })
    return {
        "task_split": label,
        "profiled_samples": len(indexes),
        "mean_item_ms": round(statistics.mean(durations) * 1000, 2),
        "median_item_ms": round(statistics.median(durations) * 1000, 2),
        "p95_item_ms": round(sorted(durations)[max(0, math.ceil(.95 * len(durations)) - 1)] * 1000, 2),
        "top_cumulative_functions": rows,
    }


def validate_detection_batch(images: torch.Tensor, targets: list[dict]) -> tuple[list[str], dict]:
    if images.ndim != 4 or images.shape[1] != 3 or not torch.isfinite(images).all():
        raise AssertionError(f"Invalid detection batch image tensor: {tuple(images.shape)}")
    ids, empty, boxes_n, classes = [], 0, 0, set()
    height, width = images.shape[-2:]
    for target in targets:
        image_id = str(target["image_id"])
        ids.append(image_id)
        boxes = target["boxes"]
        labels = target["labels"]
        if boxes.ndim != 2 or boxes.shape[1] != 4 or labels.ndim != 1 or len(boxes) != len(labels):
            raise AssertionError(f"Malformed target for {image_id}")
        if not torch.isfinite(boxes).all():
            raise AssertionError(f"Non-finite box for {image_id}")
        if len(boxes) == 0:
            empty += 1
        if len(boxes):
            if (boxes[:, 0] < 0).any() or (boxes[:, 1] < 0).any() or (boxes[:, 2] > width + 1e-3).any() or (boxes[:, 3] > height + 1e-3).any():
                raise AssertionError(f"Out-of-bounds box for {image_id}")
            if (boxes[:, 2] <= boxes[:, 0]).any() or (boxes[:, 3] <= boxes[:, 1]).any():
                raise AssertionError(f"Degenerate box for {image_id}")
            if (labels < 0).any() or (labels >= len(DETECTION_CLASS_NAMES)).any():
                raise AssertionError(f"Invalid detection class id for {image_id}")
            boxes_n += len(boxes)
            classes.update(int(v) for v in labels.tolist())
        if target.get("meta", {}).get("annotation_name") not in ("", image_id):
            raise AssertionError(f"Annotation ID mismatch for {image_id}")
    return ids, {"empty_samples": empty, "boxes": boxes_n, "class_ids_seen": sorted(classes)}


def validate_segmentation_batch(images: torch.Tensor, masks: torch.Tensor) -> dict:
    if images.ndim != 4 or images.shape[1] != 3 or not torch.isfinite(images).all():
        raise AssertionError(f"Invalid segmentation image tensor: {tuple(images.shape)}")
    if masks.ndim != 3 or tuple(images.shape[0:1] + images.shape[-2:]) != tuple(masks.shape):
        raise AssertionError(f"Image/mask batch shape mismatch: {tuple(images.shape)}, {tuple(masks.shape)}")
    valid = set(range(len(SEGMENTATION_CLASS_NAMES))) | {255}
    found = set(int(value) for value in torch.unique(masks).tolist())
    if not found <= valid:
        raise AssertionError(f"Unexpected mapped mask values: {sorted(found - valid)}")
    empty = sum(not bool((mask != 255).any()) for mask in masks)
    return {"mapped_ids_seen": sorted(found), "empty_valid_pixel_samples": int(empty)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-split", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.samples_per_split <= 0 or args.batch_size <= 0:
        parser.error("sample and batch sizes must be positive")

    definitions = [
        ("detection", split, BDD100KDetectionDataset(split=split), detection_collate_fn)
        for split in ("train", "val", "test")
    ] + [
        ("segmentation", split, BDD100KSegmentationDataset(split=split), segmentation_collate_fn)
        for split in ("train", "val")
    ]
    sample_sets = {}
    for task, split, dataset, collate in definitions:
        rng = random.Random(args.seed + len(sample_sets))
        count = min(args.samples_per_split, len(dataset))
        indexes = sorted(rng.sample(range(len(dataset)), count))
        sample_sets[(task, split)] = (dataset, indexes, collate)

    results = {
        "method": "Seeded index sampling per manifest split; each sample is read through a bounded DataLoader Subset. No full dataset tensors are retained.",
        "seed": args.seed,
        "samples_per_split_requested": args.samples_per_split,
        "batch_size": args.batch_size,
        "workers_compared": args.workers,
        "splits": {},
        "profiles": [],
        "process_peak_rss_mb": None,
    }
    for (task, split), (dataset, indexes, collate) in sample_sets.items():
        key = f"{task}_{split}"
        row = {"dataset_size": len(dataset), "sampled": len(indexes), "sampled_indices_min_max": [indexes[0], indexes[-1]], "runs": []}
        if task == "detection":
            records = dataset.records
            row["sampled_ids"] = [records[i]["image_id"] for i in indexes]
        else:
            records = dataset.records
            row["sampled_ids"] = [records[i]["image_id"] for i in indexes]
        for workers in args.workers:
            loader = DataLoader(Subset(dataset, indexes), batch_size=args.batch_size, shuffle=False,
                                num_workers=workers, collate_fn=collate)
            started = time.perf_counter()
            item_count = batch_count = box_count = empty_count = 0
            observed_classes, observed_mask_ids, observed_empty_masks = set(), set(), 0
            seen_ids = []
            for batch in loader:
                batch_count += 1
                if task == "detection":
                    images, targets = batch
                    ids, counts = validate_detection_batch(images, targets)
                    seen_ids.extend(ids)
                    item_count += len(ids)
                    box_count += counts["boxes"]
                    empty_count += counts["empty_samples"]
                    observed_classes.update(counts["class_ids_seen"])
                else:
                    images, masks = batch
                    counts = validate_segmentation_batch(images, masks)
                    start = item_count
                    seen_ids.extend(row["sampled_ids"][start:start + len(masks)])
                    item_count += len(masks)
                    observed_mask_ids.update(counts["mapped_ids_seen"])
                    observed_empty_masks += counts["empty_valid_pixel_samples"]
            elapsed = time.perf_counter() - started
            if item_count != len(indexes) or len(seen_ids) != len(set(seen_ids)):
                raise AssertionError(f"Sample coverage or duplicate IDs failed: {key}, workers={workers}")
            row["runs"].append({
                "workers": workers, "batches": batch_count, "items_checked": item_count,
                "seconds": round(elapsed, 3), "items_per_second": round(item_count / elapsed, 2),
                "ids_unique": True, "empty_detection_samples": empty_count if task == "detection" else None,
                "detection_boxes_checked": box_count if task == "detection" else None,
                "detection_class_ids_seen": sorted(observed_classes) if task == "detection" else None,
                "segmentation_mapped_ids_seen": sorted(observed_mask_ids) if task == "segmentation" else None,
                "empty_segmentation_samples": observed_empty_masks if task == "segmentation" else None,
            })
        results["splits"][key] = row

    for key in ("detection_train", "segmentation_train"):
        dataset, indexes, _ = sample_sets[(key.split("_")[0], "train")]
        results["profiles"].append(profile_items(dataset, indexes[:min(32, len(indexes))], key))
    results["process_peak_rss_mb"] = rss_mb()
    out = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    compact = {key: [{"workers": run["workers"], "items": run["items_checked"], "s": run["seconds"], "items/s": run["items_per_second"]} for run in value["runs"]] for key, value in results["splits"].items()}
    print(json.dumps({"output": str(out), "samples_per_split": args.samples_per_split, "runs": compact,
                      "peak_rss_mb": results["process_peak_rss_mb"], "profiles": results["profiles"]}, indent=2))


if __name__ == "__main__":
    main()
