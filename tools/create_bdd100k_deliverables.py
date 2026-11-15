#!/usr/bin/env python3
"""Create BDD100K ID lists and native-size image/box/mask visual examples."""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from data.bdd100k_dataset import (
    normalize_bdd100k_image, read_train_id_mask, map_train_id_mask,
    _load_boxes, DETECTION_CLASS_NAMES, audit_multitask_split_ids,
)
from data.prepare_bdd100k import _palette


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    root = PROJECT_ROOT
    processed = root / "data/processed/bdd100k"
    sample_dir = root / "dataset_samples"
    split_dir = root / "split_manifest"
    sample_dir.mkdir(exist_ok=True)
    split_dir.mkdir(exist_ok=True)
    detection = {s: read_rows(processed / f"detection_{s}.csv") for s in ("train", "val", "test")}
    segmentation = {s: read_rows(processed / f"segmentation_{s}.csv") for s in ("train", "val")}
    official_segmentation_train = read_rows(processed / "segmentation_official_train.csv")

    manifests = {}
    for task, table, task_splits in (
        ("detection", detection, ("train", "val", "test")),
        ("segmentation", segmentation, ("train", "val")),
    ):
        for split in task_splits:
            label = "validation" if split == "val" else split
            name = f"{task}_{label}_ids.txt"
            ids = sorted(row["image_id"] for row in table[split])
            content = "".join(image_id + "\n" for image_id in ids)
            (split_dir / name).write_text(content, encoding="utf-8")
            manifests[name] = {
                "count": len(ids),
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
            }
    (split_dir / "segmentation_test_ids.txt").write_text("", encoding="utf-8")
    manifests["segmentation_test_ids.txt"] = {
        "count": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
    }
    det_ids = {split: {row["image_id"] for row in rows} for split, rows in detection.items()}
    seg_ids = {split: {row["image_id"] for row in rows} for split, rows in segmentation.items()}
    seg_ids["test"] = set()
    split_audit = audit_multitask_split_ids(det_ids, seg_ids, reject_leakage=True)
    additional_id_lists = {
        "segmentation_official_train_ids.txt": sorted(row["image_id"] for row in official_segmentation_train),
        "segmentation_excluded_combined_evaluation_ids.txt": sorted(
            {row["image_id"] for row in official_segmentation_train} - seg_ids["train"]
        ),
        "combined_train_ids.txt": sorted(det_ids["train"] | seg_ids["train"]),
        "combined_validation_ids.txt": sorted(det_ids["val"] | seg_ids["val"]),
        "combined_test_ids.txt": sorted(det_ids["test"] | seg_ids["test"]),
        "cross_task_shared_train_ids.txt": sorted(det_ids["train"] & seg_ids["train"]),
        "cross_task_detection_validation_in_segmentation_train_ids.txt": sorted(det_ids["val"] & seg_ids["train"]),
        "cross_task_detection_test_in_segmentation_train_ids.txt": sorted(det_ids["test"] & seg_ids["train"]),
        "cross_task_detection_train_in_segmentation_validation_ids.txt": sorted(det_ids["train"] & seg_ids["val"]),
        "cross_task_segmentation_validation_in_detection_test_ids.txt": sorted(seg_ids["val"] & det_ids["test"]),
    }
    for name, ids in additional_id_lists.items():
        content = "".join(image_id + "\n" for image_id in ids)
        (split_dir / name).write_text(content, encoding="utf-8")
        manifests[name] = {"count": len(ids), "sha256": hashlib.sha256(content.encode()).hexdigest()}
    (split_dir / "manifest_summary.json").write_text(json.dumps({
        "note": "Segmentation test masks are not provided. Segmentation training excludes all combined validation/test IDs; official segmentation validation and all 830 detection test IDs remain held out.",
        "split_counts": {
            "detection": {split: len(ids) for split, ids in det_ids.items()},
            "segmentation_working": {split: len(ids) for split, ids in seg_ids.items()},
            "segmentation_official_train": len(official_segmentation_train),
        },
        "split_audit": split_audit,
        "files": manifests,
    }, indent=2) + "\n", encoding="utf-8")

    seg_train_by_id = {row["image_id"]: row for row in segmentation["train"]}
    common = [row for row in detection["train"]
              if row["image_id"] in seg_train_by_id and int(row["box_count"]) > 0]
    chosen = []
    known = "00067cfb-e535423e"
    if any(row["image_id"] == known for row in common):
        chosen.append(next(row for row in common if row["image_id"] == known))
    candidates = []
    for row in common:
        if row["image_id"] in {item["image_id"] for item in chosen}:
            continue
        class_counts = json.loads(row["class_counts"])
        candidates.append((row, len(class_counts), int(row["box_count"])))
    for predicate in (
        lambda item: item[1] >= 4 and 5 <= item[2] <= 20,
        lambda item: 5 <= item[2] <= 40,
        lambda item: True,
    ):
        pool = [item for item in candidates if predicate(item)]
        pool.sort(key=lambda item: (-item[1], abs(item[2] - 12), item[0]["image_id"]))
        while pool and len(chosen) < 3:
            selected = pool.pop(0)[0]
            chosen.append(selected)
            candidates = [item for item in candidates if item[0]["image_id"] != selected["image_id"]]
            pool = [item for item in candidates if predicate(item)]
            pool.sort(key=lambda item: (-item[1], abs(item[2] - 12), item[0]["image_id"]))
    if len(chosen) < 3:
        raise RuntimeError(f"Only {len(chosen)} aligned image/box/mask triplets available")

    metadata = []
    palette = ((255, 70, 70), (255, 180, 0), (40, 210, 255),
               (160, 90, 255), (255, 80, 190), (60, 220, 120))
    contact = Image.new("RGB", (1920, 3 * 385), (235, 238, 242))
    contact_draw = ImageDraw.Draw(contact)
    for index, row in enumerate(chosen, 1):
        image_id = row["image_id"]
        image_path = Path(row["image_path"])
        annotation_path = Path(row["annotation_path"])
        mask_path = Path(seg_train_by_id[image_id]["train_id_mask_path"])
        image_path = image_path if image_path.is_absolute() else (root / image_path).resolve()
        annotation_path = annotation_path if annotation_path.is_absolute() else (root / annotation_path).resolve()
        mask_path = mask_path if mask_path.is_absolute() else (root / mask_path).resolve()
        with Image.open(image_path) as source:
            image = normalize_bdd100k_image(source)
        raw_mask = read_train_id_mask(mask_path)
        mask_size = (int(raw_mask.shape[1]), int(raw_mask.shape[0]))
        if image.size != mask_size:
            raise AssertionError(f"Image/mask dimensions differ for {image_id}: {image.size} vs {mask_size}")
        boxes, labels, annotation = _load_boxes(annotation_path, *image.size)
        if annotation.get("name") != image_id:
            raise AssertionError(f"Annotation name mismatch for {image_id}: {annotation.get('name')}")

        boxed = image.copy()
        draw = ImageDraw.Draw(boxed)
        for coords, class_id in zip(boxes, labels):
            category = DETECTION_CLASS_NAMES[class_id]
            color = palette[class_id % len(palette)]
            draw.rectangle(coords, outline=color, width=4)
            draw.text((coords[0] + 2, max(0, coords[1] - 16)), category,
                      fill=color, stroke_width=2, stroke_fill=(0, 0, 0))
        mapped = map_train_id_mask(raw_mask)
        mask_rgb = Image.fromarray(_palette(mapped), mode="RGB")
        overlay = Image.blend(image, mask_rgb, 0.42)
        views = (("original", image), ("bounding_boxes", boxed), ("segmentation_overlay", overlay))
        names = []
        for column, (kind, view) in enumerate(views):
            filename = f"sample_{index:02d}_{image_id}_{kind}.jpg"
            view.save(sample_dir / filename, quality=95)
            names.append(filename)
            x = column * 640
            y = (index - 1) * 385
            contact_draw.rectangle((x, y, x + 640, y + 24), fill=(25, 31, 42))
            contact_draw.text((x + 8, y + 5), f"{image_id} - {kind}", fill="white")
            resample = Image.Resampling.NEAREST if kind == "segmentation_overlay" else Image.Resampling.BILINEAR
            contact.paste(view.resize((640, 360), resample), (x, y + 25))
        metadata.append({
            "image_id": image_id,
            "annotation_name": annotation.get("name"),
            "same_id": annotation.get("name") == image_id,
            "image_size_wh": list(image.size),
            "mask_size_wh": list(mask_size),
            "same_size": image.size == mask_size,
            "boxes_drawn": len(boxes),
            "class_counts": json.loads(row["class_counts"]),
            "files": names,
        })

    contact_file = "alignment_contact_sheet.jpg"
    contact.save(sample_dir / contact_file, quality=92)
    (sample_dir / "samples.json").write_text(json.dumps({
        "method": "Three exact-ID detection/segmentation training triplets; source image orientation normalized before drawing; annotation name and image/mask dimensions asserted.",
        "examples": metadata,
        "contact_sheet": contact_file,
    }, indent=2) + "\n", encoding="utf-8")
    (split_dir / "README.md").write_text(
        "# BDD100K split ID manifests\n\n"
        "Each `*_ids.txt` file has one exact image stem per line, sorted lexicographically. Detection uses seed 42 "
        "and a 20% validation slice of the 3,430 matched 10K training images; the separate 830 test IDs stay held out. "
        "Segmentation validation remains the official 1,000-ID validation split. Its working train manifest removes any "
        "ID in either task's validation/test set; `segmentation_official_train_ids.txt` preserves the full official "
        "7,000-ID source list, and `segmentation_excluded_combined_evaluation_ids.txt` records the 686 removed IDs. "
        "There is no semantic test-mask split, so `segmentation_test_ids.txt` is empty.\n\n"
        "`manifest_summary.json` gives within-task, cross-task train/evaluation and validation/test intersections, plus "
        "combined split counts. Same-ID sharing between detection and segmentation training is allowed; no combined "
        "training ID appears in either evaluation set.\n", encoding="utf-8")
    (sample_dir / "README.md").write_text(
        "# BDD100K visual alignment examples\n\n"
        "Each example has an orientation-normalized original, a detection-box rendering, and a five-class "
        "segmentation overlay. Box coordinates are native image pixels; the mask blend is 42% opacity and "
        "void/ignore appears yellow. All example IDs match the detection JSON `name` and mask stem, and "
        "the source image and mask dimensions were checked. `samples.json` records IDs, dimensions, box "
        "counts, class counts and filenames. `alignment_contact_sheet.jpg` previews all three triplets.\n",
        encoding="utf-8")
    print(json.dumps({"manifests": manifests, "visual_examples": metadata,
                      "contact_sheet": str(sample_dir / contact_file)}, indent=2))


if __name__ == "__main__":
    main()
