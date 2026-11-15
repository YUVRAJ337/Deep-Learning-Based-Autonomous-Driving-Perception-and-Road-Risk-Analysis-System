"""Focused BDD100K taxonomy, validation, split and loader tests."""
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from data.bdd100k_dataset import (
    BDD100KDetectionDataset,
    BDD100KSegmentationDataset,
    DEFAULT_PROCESSED_ROOT,
    DETECTION_CLASS_NAMES,
    SEGMENTATION_CLASS_NAMES,
    SEGMENTATION_IGNORE_INDEX,
    TRAIN_ID_NAMES,
    assert_no_split_leakage,
    audit_multitask_split_ids,
    build_bdd100k_dataloaders,
    map_train_id_mask,
    normalize_bdd100k_image,
    read_train_id_mask,
)
from data.prepare_bdd100k import split_training_ids


class BDD100KMappingTests(unittest.TestCase):
    def test_detection_class_count_and_order(self):
        self.assertEqual(len(DETECTION_CLASS_NAMES), 10)
        self.assertEqual(DETECTION_CLASS_NAMES[0], "bike")
        self.assertEqual(DETECTION_CLASS_NAMES[-1], "truck")

    def test_all_official_train_ids_and_void_mapping(self):
        self.assertEqual(len(TRAIN_ID_NAMES), 19)
        raw = np.array([list(range(19)) + [255, 19, 254]], dtype=np.uint8)
        got = map_train_id_mask(raw)
        expected = np.array([[1, 2] + [0] * 9 + [4, 4] + [3] * 6 + [255, 255, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(got, expected)
        self.assertEqual(SEGMENTATION_CLASS_NAMES, ("background", "road", "sidewalk", "vehicles", "vulnerable road users"))
        self.assertEqual(SEGMENTATION_IGNORE_INDEX, 255)

    def test_rejects_rgb_masks(self):
        with self.assertRaises(ValueError):
            map_train_id_mask(np.zeros((2, 2, 3), dtype=np.uint8))

    def test_palette_indices_are_not_converted_to_palette_luminance(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "palette.png"
            image = Image.fromarray(np.array([[0, 1, 13, 255]], dtype=np.uint8), mode="P")
            image.putpalette([255, 255, 255] * 256)
            image.save(p)
            np.testing.assert_array_equal(read_train_id_mask(p), np.array([[0, 1, 13, 255]], dtype=np.uint8))

    def test_portrait_bdd_frame_rotates_to_landscape_mask_orientation(self):
        normalized = normalize_bdd100k_image(Image.new("RGB", (720, 1280)))
        self.assertEqual(normalized.size, (1280, 720))


class SplitTests(unittest.TestCase):
    def test_seeded_split_is_reproducible_and_complete(self):
        source = [f"id-{i}" for i in range(19)]
        a = split_training_ids(source, seed=42, val_fraction=0.2)
        b = split_training_ids(source, seed=42, val_fraction=0.2)
        self.assertEqual(a, b)
        self.assertEqual(set(a[0]) | set(a[1]), set(source))
        self.assertFalse(set(a[0]) & set(a[1]))

    def test_split_leakage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "leak"):
            assert_no_split_leakage({"train": ["a", "b"], "val": ["b"], "test": ["c"]})
        assert_no_split_leakage({"train": ["a"], "val": ["b"], "test": ["c"]})
        with self.assertRaisesRegex(ValueError, "Combined BDD100K splits leak"):
            audit_multitask_split_ids(
                {"train": ["shared-val"]}, {"val": ["shared-val"]}
            )
        allowed = audit_multitask_split_ids(
            {"train": ["shared-train"]}, {"train": ["shared-train"]}
        )
        self.assertTrue(allowed["leakage_free"])
        self.assertEqual(allowed["same_split_shared_ids"]["train"], 1)


class BDD100KDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "images").mkdir()
        (self.root / "labels").mkdir()
        (self.root / "masks").mkdir()
        Image.new("RGB", (8, 4), (100, 120, 140)).save(self.root / "images" / "one.jpg")
        self.annotation = {
            "name": "one",
            "frames": [{"objects": [
                {"category": "car", "box2d": {"x1": 1, "y1": 1, "x2": 7, "y2": 3}},
                {"category": "person", "box2d": {"x1": 2, "y1": 0, "x2": 4, "y2": 4}},
                {"category": "area/drivable", "box2d": {"x1": 0, "y1": 0, "x2": 2, "y2": 2}},
                {"category": "car", "box2d": {"x1": 2, "y1": 2, "x2": 2, "y2": 3}},
                {"category": "car", "box2d": {"x1": float("nan"), "y1": 0, "x2": 4, "y2": 3}},
                {"category": "road", "poly2d": []},
            ]}],
        }
        (self.root / "labels" / "one.json").write_text(json.dumps(self.annotation))
        raw_mask = np.array([
            [0, 0, 1, 1, 2, 2, 11, 11],
            [13, 13, 14, 14, 18, 18, 255, 255],
            [3, 3, 4, 4, 5, 5, 6, 6],
            [7, 7, 8, 8, 9, 9, 10, 10],
        ], dtype=np.uint8)
        Image.fromarray(raw_mask, mode="L").save(self.root / "masks" / "one.png")
        self._write_manifest(self.root / "det.csv", ["image_id", "image_split", "label_split", "image_path", "annotation_path"], [{
            "image_id": "one", "image_split": "train", "label_split": "train",
            "image_path": "images/one.jpg", "annotation_path": "labels/one.json",
        }])
        self._write_manifest(self.root / "seg.csv", ["image_id", "image_path", "train_id_mask_path"], [{
            "image_id": "one", "image_path": "images/one.jpg", "train_id_mask_path": "masks/one.png",
        }])

    def _write_manifest(self, path, fields, rows):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)

    def tearDown(self):
        self.tmp.cleanup()

    def test_detection_boxes_scale_skip_invalid_and_unknown_labels(self):
        ds = BDD100KDetectionDataset("train", self.root / "det.csv", self.root, image_size=(8, 16))
        image, target = ds[0]
        self.assertEqual(tuple(image.shape), (3, 8, 16))
        self.assertEqual(tuple(target["boxes"].shape), (2, 4))
        self.assertEqual(target["labels"].tolist(), [2, 4])
        self.assertAlmostEqual(float(target["boxes"][0, 0]), 2.0)
        self.assertAlmostEqual(float(target["boxes"][0, 2]), 14.0)

    def test_empty_detection_sample_returns_well_shaped_targets(self):
        self.annotation["frames"] = [{"objects": []}]
        (self.root / "labels" / "one.json").write_text(json.dumps(self.annotation))
        ds = BDD100KDetectionDataset("train", self.root / "det.csv", self.root, image_size=None)
        _, target = ds[0]
        self.assertEqual(tuple(target["boxes"].shape), (0, 4))
        self.assertEqual(tuple(target["labels"].shape), (0,))

    def test_detection_annotation_id_mismatch_is_rejected(self):
        self.annotation["name"] = "not-one"
        (self.root / "labels" / "one.json").write_text(json.dumps(self.annotation))
        ds = BDD100KDetectionDataset("train", self.root / "det.csv", self.root, image_size=None)
        with self.assertRaisesRegex(ValueError, "does not match"):
            ds[0]

    def test_missing_file_is_reported_at_sample_load(self):
        (self.root / "images" / "one.jpg").unlink()
        ds = BDD100KDetectionDataset("train", self.root / "det.csv", self.root, image_size=None)
        with self.assertRaises(FileNotFoundError):
            ds[0]

    def test_segmentation_loader_maps_pixels_preserves_ignore_and_rejects_invalid_output(self):
        ds = BDD100KSegmentationDataset("train", self.root / "seg.csv", self.root, image_size=(4, 8))
        image, mask = ds[0]
        self.assertEqual(tuple(image.shape), (3, 4, 8))
        self.assertEqual(tuple(mask.shape), (4, 8))
        self.assertEqual(mask[0, 0].item(), 1)
        self.assertEqual(mask[0, 2].item(), 2)
        self.assertEqual(mask[0, 6].item(), 4)
        self.assertIn(SEGMENTATION_IGNORE_INDEX, mask.tolist()[1])

        def invalid_transform(**sample):
            bad_mask = sample["mask"].copy()
            bad_mask[0, 0] = 42
            return {"image": sample["image"], "mask": bad_mask}

        invalid_ds = BDD100KSegmentationDataset(
            "train", self.root / "seg.csv", self.root, image_size=None, transform=invalid_transform,
        )
        with self.assertRaisesRegex(ValueError, "invalid class IDs: \\[42\\]"):
            invalid_ds[0]

    def test_segmentation_loader_rejects_unaligned_mask(self):
        Image.new("L", (4, 4), 0).save(self.root / "masks" / "one.png")
        ds = BDD100KSegmentationDataset("train", self.root / "seg.csv", self.root, image_size=None)
        with self.assertRaisesRegex(ValueError, "spatial mismatch"):
            ds[0]


class PreparedDataIntegrationTests(unittest.TestCase):
    @unittest.skipUnless((DEFAULT_PROCESSED_ROOT / "summary.json").is_file(), "prepared BDD100K manifests not generated yet")
    def test_actual_manifests_have_disjoint_complete_splits(self):
        def read_ids(name):
            with (DEFAULT_PROCESSED_ROOT / name).open(newline="", encoding="utf-8") as f:
                return [row["image_id"] for row in csv.DictReader(f)]
        detection = {s: read_ids(f"detection_{s}.csv") for s in ("train", "val", "test")}
        assert_no_split_leakage(detection)
        self.assertEqual([len(detection[s]) for s in ("train", "val", "test")], [2744, 686, 830])
        segmentation = {s: read_ids(f"segmentation_{s}.csv") for s in ("train", "val")}
        assert_no_split_leakage(segmentation)
        self.assertEqual([len(segmentation[s]) for s in ("train", "val")], [6314, 1000])
        official_train = read_ids("segmentation_official_train.csv")
        self.assertEqual(len(official_train), 7000)
        excluded = set(official_train) - set(segmentation["train"])
        self.assertEqual(len(excluded), 686)
        self.assertEqual(excluded, set(detection["val"]))
        self.assertEqual(len(set(detection["test"]) & (set(detection["train"]) | set(detection["val"]))), 0)
        audit = audit_multitask_split_ids(
            detection, {**segmentation, "test": []}, reject_leakage=True
        )
        self.assertTrue(audit["leakage_free"])
        self.assertEqual(audit["same_split_shared_ids"]["train"], 2744)
        self.assertEqual(audit["combined_splits"]["train_vs_validation"], 0)
        self.assertEqual(audit["combined_splits"]["train_vs_test"], 0)
        self.assertEqual(audit["combined_splits"]["validation_vs_test"], 0)
        loaders = build_bdd100k_dataloaders(batch_size=2, image_size=None, num_workers=0)
        self.assertEqual(len(loaders["segmentation"]["train"].dataset), 6314)

    @unittest.skipUnless((DEFAULT_PROCESSED_ROOT / "summary.json").is_file(), "prepared BDD100K manifests not generated yet")
    def test_actual_manifest_source_files_exist(self):
        for path in DEFAULT_PROCESSED_ROOT.glob("*.csv"):
            if path.name not in {"missing_files.csv", "detection_unmatched_images.csv", "corrupt_files.csv", "unusable_segmentation.csv", "class_distribution.csv"}:
                with path.open(newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        for key in ("image_path", "annotation_path", "train_id_mask_path"):
                            if row.get(key):
                                p = Path(row[key])
                                p = p if p.is_absolute() else DEFAULT_PROCESSED_ROOT.parents[2] / p
                                self.assertTrue(p.is_file(), f"missing source file: {p}")


if __name__ == "__main__":
    unittest.main()
