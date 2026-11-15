# BDD100K dataset preparation and verification

**Project:** `/Users/yuvrajsingh/Downloads/Project/Deep-Learning-Based-Autonomous-Driving-Perception-and-Road-Risk-Analysis-System/`  
**Audit date:** 2026-11-14  
**Scope:** prepare the selected 10K images, BDD100K 2018 detection labels and semantic train-ID masks. No model training was run. The reference `../autonomous-driving/` repository was not changed.

## Result at a glance

- Original ZIPs remain in `data/raw/bdd100k/`; extracted source trees were read in place from sibling folders, not duplicated.
- Detection manifests: **2,744 train / 686 validation / 830 held-out test**. All 830 test IDs remain unchanged.
- Working segmentation manifests: **6,314 train / 1,000 official validation / 0 test**. The original official 7,000-ID train list is preserved separately; its 686 IDs shared with detection validation are excluded from working segmentation training.
- Combined unique-ID splits: **6,314 train / 1,686 validation / 830 test**. Same-split train sharing of 2,744 IDs is allowed; all train/evaluation and validation/test intersections are zero.
- The preparation audit checked all **10,000 images** and all **8,000 official segmentation image/mask pairs**. Missing files: **0**; corrupt images/masks or matched detection JSONs: **0**; unusable segmentation pairs: **0**.
- **5,740** 10K images have no matching 2018 detection JSON; this is the selected-package coverage gap, not file loss. **21** malformed or degenerate detection boxes were skipped; unknown categories: **0**.

## Sources, provenance and integrity

The official BDD100K download documentation lists the 10K image package and MD5 `08f26aecceda982568063d3d5873378e`, and points to the ETH Zürich data host. The local image archive MD5 is `9c5ee9df0097cc1ab799f58feb776611`, so the exact image-package release identity cannot be confirmed from that checksum. The earlier download record names the legacy Berkeley URL `http://128.32.162.150/bdd100k/bdd100k_images_10k.zip`; unlike the labels and mask ZIPs, the image archive has no macOS `kMDItemWhereFroms` attribute. The labels and segmentation ZIPs do have source attributes naming the same legacy Berkeley host.

A fresh `zipfile.testzip()` pass reported no bad members for any archive. Separately, every non-directory ZIP member was mapped to its extracted counterpart; extracted byte length and CRC32 were compared with the ZIP directory values. The result was **126,000/126,000 matches** across **3,136,138,559 uncompressed bytes**, with no missing extracted members or size/CRC differences:

| Archive | Members | Extracted matches | Local MD5 |
|---|---:|---:|---|
| `bdd100k_images_10k.zip` | 10,000 | 10,000 | `9c5ee9df0097cc1ab799f58feb776611` |
| `bdd100k_labels.zip` | 100,000 | 100,000 | `2d0165fd89b11025206e21a25ae1089d` |
| `bdd100k_seg_maps.zip` | 16,000 | 16,000 | `7607c97f0fc366fad39f0a450ab7c4fc` |

These CRC and size results support intact extraction and internally consistent archives; CRC32 is not a cryptographic proof that the image ZIP is the exact official release. The differing official MD5 remains unresolved, **not evidence by itself of corruption**. The machine-readable member audit is `data/processed/bdd100k/archive_integrity.json`.

## Source and output structure

```text
<main-project>/
├── data/raw/bdd100k/                 # three original ZIPs; gitignored
├── data/processed/bdd100k/           # manifests, audits and summary; gitignored
│   ├── detection_{train,val,test}.csv
│   ├── segmentation_{train,val}.csv       # leakage-free working splits
│   ├── segmentation_official_train.csv    # full 7,000-ID source split
│   └── segmentation_train_excluded_evaluation.csv
│   ├── class_distribution.csv
│   ├── missing_files.csv
│   ├── corrupt_files.csv
│   ├── unusable_segmentation.csv
│   ├── summary.json
│   ├── archive_integrity.json
│   └── runtime_validation.json
├── split_manifest/                   # ID lists; gitignored
├── dataset_samples/                  # derived visual examples; gitignored
├── dataset_config.yaml               # small project configuration
├── dataset_report.md                 # this report
└── test_results.txt                  # test log; gitignored

Read-only extracted inputs: ../10k, ../100k, ../bdd100k_seg_maps
```

## Detection splits and classes

IDs are matched on exact filename stem. A deterministic seed-42, 20% validation split was made from only the **3,430 matched 10K train images**. The **830 test IDs** remain held apart; each has `image_split=test` and `label_split=test`. The official segmentation validation IDs are untouched. The official 7,000 segmentation-train IDs are retained in `segmentation_official_train.csv`, while 686 IDs that belong to detection validation are excluded from the working segmentation-train manifest. Thus working counts are 6,314/1,000/0.

| Detection class ID | Class | Train boxes | Validation boxes | Test boxes |
|---:|---|---:|---:|---:|
| 0 | bike | 366 | 107 | 116 |
| 1 | bus | 581 | 141 | 171 |
| 2 | car | 29,603 | 7,523 | 9,228 |
| 3 | motor | 161 | 50 | 39 |
| 4 | person | 4,624 | 1,137 | 1,228 |
| 5 | rider | 241 | 63 | 72 |
| 6 | traffic light | 6,767 | 1,812 | 2,159 |
| 7 | traffic sign | 9,937 | 2,599 | 3,027 |
| 8 | train | 21 | 1 | 2 |
| 9 | truck | 1,484 | 427 | 474 |
| **Total boxes** |  | **53,785** | **13,860** | **16,516** |

No unknown box category or empty labeled image was found. There are 5,740 unmatched images (3,570 train, 1,000 validation, 1,170 test) without a matching 2018 JSON; the preparation CSV records each one.

## Semantic train IDs and mapping

The raw masks use the official BDD100K `train_id` encoding: `0 road, 1 sidewalk, 2 building, 3 wall, 4 fence, 5 pole, 6 traffic light, 7 traffic sign, 8 vegetation, 9 terrain, 10 sky, 11 person, 12 rider, 13 car, 14 truck, 15 bus, 16 train, 17 motorcycle, 18 bicycle`; `255` is void/ignore.

| Official train IDs | Source labels | Project class ID |
|---|---|---:|
| 0 | road | 1 road |
| 1 | sidewalk | 2 sidewalk |
| 2–10 | built/other: building through sky | 0 background |
| 11–12 | person, rider | 4 vulnerable road users |
| 13–18 | car, truck, bus, train, motorcycle, bicycle | 3 vehicles |
| 255 or any unrecognized value | void/unknown | 255 ignore |

Unknown values are ignored, never treated as background. The exhaustive mask scan found only IDs 0–18 and 255, with zero unexpected values.

| Mapped mask class | Working train pixels | Validation pixels |
|---|---:|---:|
| Background | 2,798,837,238 | 476,506,812 |
| Road | 1,252,441,133 | 199,063,763 |
| Sidewalk | 115,943,431 | 18,895,436 |
| Vehicles | 564,629,111 | 99,072,672 |
| Vulnerable road users | 15,657,964 | 2,370,886 |
| Void / ignore (255) | 1,071,473,523 | 125,690,431 |

### Combined split leakage audit

The 686 IDs formerly shared between detection validation and segmentation training have been removed from the working segmentation-train split. Official segmentation validation remains all 1,000 IDs; the full 7,000-ID official segmentation-train source list and the 686 excluded IDs remain separately recorded. The 830 detection test IDs were verified absent from segmentation train/validation.

Train/evaluation intersections (all expected to be zero):

| ID split pair | Count | Status |
|---|---:|---|
| Detection train × detection val | 0 | disjoint |
| Detection train × detection test | 0 | disjoint |
| Segmentation train × segmentation val | 0 | disjoint |
| Segmentation train × segmentation test | 0 | disjoint |
| Detection train × segmentation val | 0 | disjoint |
| Detection train × segmentation test | 0 | disjoint |
| Segmentation train × detection val | 0 | disjoint |
| Segmentation train × detection test | 0 | disjoint |

Validation/test intersections are also explicit:

| ID split pair | Count | Status |
|---|---:|---|
| Detection val × detection test | 0 | disjoint |
| Segmentation val × segmentation test | 0 | disjoint |
| Detection val × segmentation test | 0 | disjoint |
| Segmentation val × detection test | 0 | disjoint |

Combined train × validation, train × test and validation × test are each **0**. Detection train × segmentation train shares **2,744 IDs**, a legitimate same-split overlap. Segmentation test is empty because no semantic test masks were included. Exact pairwise counts and combined ID lists are in `split_manifest/manifest_summary.json` and the adjacent `*_ids.txt` files. `build_bdd100k_dataloaders()` now rejects cross-task train/evaluation leakage before constructing loaders.

## Alignment examples and loader profiling

The preparation scan matched each segmentation mask by image ID and confirmed dimensions after orientation normalization for all 8,000 train/validation pairs. Six portrait source JPEGs in the segmentation partitions were rotated counter-clockwise to match landscape masks. The bounded loader audit then read **64 seeded samples from each of five splits** (detection train/validation/test and segmentation train/validation), batch size 8. It checked finite images, box bounds/order/classes, annotation IDs, image/mask tensor dimensions, valid mapped mask IDs and unique sampled IDs. Coverage was 320 distinct task-split samples per worker setting; no full dataset tensors were held in memory.

The earlier exhaustive DataLoader iteration over all 12,260 task-split records did not finish within 300 seconds. Profiling found per-item `torch.unique` class-ID validation as a segmentation hot path; it was replaced with a NumPy membership check while retaining rejection of invalid transformed labels. The latest bounded run read 64 seeded samples from each of the five task splits in batches of 8 (320 task-split samples per worker setting), without retaining a full dataset. With `num_workers=0`, measured throughput was 126.95/120.29/119.96 samples/s for detection train/validation/test and 26.38/28.53 samples/s for segmentation train/validation; two-worker fresh-loader throughput was only 4.96–5.69 samples/s because startup dominated. Peak process RSS was 543.9 MB. These are sampled timings, not a full epoch or exhaustive loader pass. See `data/processed/bdd100k/runtime_validation.json` and `test_results.txt`.

Three sample IDs were rendered with an original, a bounding-box image and a mapped-mask overlay each; all annotation names and image/mask dimensions matched:

- `00067cfb-e535423e` — 33 boxes; 1280×720
- `067cecc1-d40ce82c` — 17 boxes; 1280×720
- `38fd0fe3-ef0726e9` — 19 boxes; 1280×720

Contact sheet: `dataset_samples/alignment_contact_sheet.jpg`. Separate files (relative to the project root):

- `dataset_samples/sample_01_00067cfb-e535423e_original.jpg`
- `dataset_samples/sample_01_00067cfb-e535423e_bounding_boxes.jpg`
- `dataset_samples/sample_01_00067cfb-e535423e_segmentation_overlay.jpg`
- `dataset_samples/sample_02_067cecc1-d40ce82c_original.jpg`
- `dataset_samples/sample_02_067cecc1-d40ce82c_bounding_boxes.jpg`
- `dataset_samples/sample_02_067cecc1-d40ce82c_segmentation_overlay.jpg`
- `dataset_samples/sample_03_38fd0fe3-ef0726e9_original.jpg`
- `dataset_samples/sample_03_38fd0fe3-ef0726e9_bounding_boxes.jpg`
- `dataset_samples/sample_03_38fd0fe3-ef0726e9_segmentation_overlay.jpg`

`dataset_samples/samples.json` records the IDs, counts, dimensions and filenames.

## Tests and exact commands

All 15 focused tests pass against the revised manifests. They cover reproducible splits, same-split cross-task sharing, rejection of cross-task train/evaluation leakage, actual 6,314/1,000 manifests, all 830 held-out detection test IDs, source-file existence, empty detection targets, invalid labels, annotation-ID mismatch, image-mask alignment and palette handling. Full output is in `test_results.txt`.

From the main project root:

```bash
python3 -m data.prepare_bdd100k --seed 42 --val-fraction 0.2
python3 tools/create_bdd100k_deliverables.py
python3 tools/bdd100k_runtime_audit.py --samples-per-split 64 --batch-size 8 --workers 0 2 --seed 2026
python3 -m unittest -v tests.test_bdd100k_dataset
```

Construct CPU-friendly loaders without starting training:

```python
from data.bdd100k_dataset import build_bdd100k_dataloaders
loaders = build_bdd100k_dataloaders(
    batch_size=4, image_size=(448, 800), num_workers=0, seed=42
)
images, targets = next(iter(loaders["detection"]["train"]))
seg_images, masks = next(iter(loaders["segmentation"]["train"]))
# Detection test remains separate: loaders["detection"]["test"]
```

The agreed ResNet-18 + FPN detector and ResNet-18 + U-Net segmenter were not changed. No training or checkpoint generation was started.

## Sources

- [BDD100K official download documentation](https://doc.bdd100k.com/download.html)
- [BDD100K official package host](https://dl.cv.ethz.ch/bdd100k/data/)
- [BDD100K label format](https://doc.bdd100k.com/format.html)
- [Official BDD100K train-ID definitions](https://github.com/bdd100k/bdd100k/blob/master/bdd100k/label/label.py)
- [Recorded legacy image archive URL](http://128.32.162.150/bdd100k/bdd100k_images_10k.zip)
