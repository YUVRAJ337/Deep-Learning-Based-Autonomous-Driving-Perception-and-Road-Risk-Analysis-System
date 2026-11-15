# Vision-Based Autonomous Driving Research Platform

> Working project name. The name can be changed later without changing the project scope.

An educational and research project for building a modular autonomous-driving perception and decision system. The first milestone is a small, complete, reproducible Deep Learning project. The longer roadmap expands it toward a simulator-based research platform with detection, bird's-eye view (BEV), temporal perception, decision-model comparisons, and an optional language-model reasoning layer.

**Project status:** Planning  
**Near-term target:** One-month college-project core  
**Execution environment:** Apple Silicon MacBook Air M4; PyTorch MPS where supported  
**Initial operating scope:** Offline driving-image evaluation and visualization; simulation integration is an extension  
**Safety scope:** Research and simulation only. This project is not intended to control a real vehicle.

## Why this project

Autonomous driving is a useful setting for learning how perception, representation, decisions, and control fit together. The project is intended to build those skills through implementation and measurement, rather than only calling model APIs.

The reference project at [manntalati/autonomous-driving](https://github.com/manntalati/autonomous-driving) is an architectural learning reference. Its README and architecture plan describe a PyTorch pipeline with a CNN backbone, detection, segmentation, ViT, learned BEV, temporal fusion, and an agent interface. We will adapt ideas to our own constraints, document our own experiments, and not treat reported reference metrics as reproduced results.

## Research direction

### Working research question

> How does visual perception quality and uncertainty affect safe high-level driving decisions under familiar and shifted conditions in simulation or recorded driving data?

This question may be narrowed after the dataset and available labels are confirmed.

### Initial hypotheses to investigate

1. A trained road/lane segmentation model can provide a useful visual scene representation for a driving task.
2. Performance can change under conditions not represented in training, such as night, rain, or unfamiliar scenes.
3. Measuring uncertainty or detecting poor perception can help a downstream policy choose a more conservative action.
4. A compact decision model may handle bounded routine choices with different accuracy/latency trade-offs than a rules baseline or a frontier LLM.

These are hypotheses, not promised outcomes.

## One-month core deliverable

Build a complete, presentable vertical slice that:

- trains a compact PyTorch road/lane segmentation model;
- uses a documented dataset split that avoids near-duplicate frame leakage;
- reports training curves and per-class IoU / mean IoU;
- visualizes image, labels, predictions, and representative failures;
- projects road/lane predictions into an approximate or calibrated BEV view, clearly documenting its assumptions;
- includes a reproducible run guide, model/config notes, and a short demo.

**Proposed lead candidate: BDD100K drivable-area and lane-marking labels.** The official label documentation includes drivable-area and lane-marking tasks, plus weather, scene, and time-of-day attributes that can support held-out-condition analysis. This still needs a Phase 0 sample audit: confirm exact downloadable subsets, convert labels, inspect masks, check terms, and estimate storage before committing. [BDD100K label format](https://github.com/bdd100k/bdd100k/blob/master/doc/source/format.rst)

Alternatives: **nuScenes mini** has calibrated multi-sensor data useful for a later metric BEV/3D stage, but the reference project generates road/lane masks from map polygons, adding label-pipeline work. **CARLA-generated data** offers controlled scenarios and simulator labels, but local CARLA should not be assumed available on macOS; a campus Linux/Windows machine with a suitable GPU may make it a later or alternate route. Dataset selection is not final until Phase 0 verifies task labels, terms, size, and compute.

## Architecture

### Core vertical slice

```text
Driving image + ground-truth mask
              |
              v
Dataset / transforms / scene-level split
              |
              v
PyTorch U-Net-style road/lane segmenter
              |
      +-------+--------+
      |                |
      v                v
Segmentation metrics   Predicted mask
                       |
                       v
            Geometric BEV projection
                       |
                       v
         Visual demo + run report
```

### Long-term modular system

```text
Simulator / recorded sensors
            |
            v
Perception: segmentation + object detection + tracking
            |
            v
Scene representation: structured objects + BEV + uncertainty
            |
            v
Decision: rules / XGBoost / trained neural policy / Laya-Jev
            |
            v
Safety supervisor -> controller -> simulator
            |
            v
Metrics, logs, replay, and failure analysis

Optional later path: frontier LLM for selected slow, high-level reasoning
```

The language model is not a low-level controller. A deterministic safety layer remains responsible for enforcing safe action bounds in simulation.

## Phase and execution roadmap

Weeks are relative to the project start. The first month targets Phases 0–3. Later phases are planned extensions, not requirements for the first demonstration.

### Phase 0 — Scope, environment, and reproducibility (Week 1)

- [ ] Confirm dataset, license/terms, labels, disk footprint, and split strategy.
- [ ] Set up Python, PyTorch, and MPS; record versions and device selection.
- [ ] Create repository layout, configuration convention, logging, and seed handling.
- [ ] Run a tiny end-to-end training smoke run before building the full model.

**Exit condition:** one command can load a batch, train a tiny model, and save a result.

### Phase 1 — Data pipeline and baselines (Week 1)

- [ ] Implement dataset loading, transforms, visualization, and scene-level splitting.
- [ ] Verify image-mask alignment by viewing samples and augmentations.
- [ ] Record a simple baseline (for example, majority class or classical thresholding).

**Exit condition:** labels are trusted, split is documented, and baseline metrics are saved.

### Phase 2 — Train road/lane segmentation (Week 2)

- [ ] Implement a compact U-Net-style network and training loop.
- [ ] Use a documented segmentation loss and optimizer/scheduler configuration.
- [ ] Save checkpoints, curves, per-class IoU, and sample visualizations.
- [ ] Compare a small controlled variant or pretrained encoder if feasible.

**Exit condition:** trained checkpoint, validation metrics, and qualitative error review exist.

### Phase 3 — BEV visualization and demo (Week 3)

- [ ] Implement a geometric perspective-to-BEV transform using calibration where available.
- [ ] Record assumptions and quantify alignment against available ground truth or reference points.
- [ ] Produce a simple image / mask / BEV visualization and a short demonstration video.

**Exit condition:** the complete perception-to-BEV path runs on held-out examples.

### Phase 4 — Robustness and report (Week 4)

- [ ] Evaluate on held-out scenes or conditions; do not split adjacent frames randomly.
- [ ] Compare baseline and trained model with per-class metrics and failure cases.
- [ ] Measure inference time/FPS on the actual M4 setup and record input resolution.
- [ ] Document limitations, reproducibility steps, and next research experiments.

**Exit condition:** report and presentation accurately describe what was run and observed.

### Phase 5 — Object detection (future)

- Add cars, pedestrians, cyclists, and other relevant classes.
- Compare a pretrained detector with a trainable/custom head where data and time permit.
- Evaluate mAP and per-class errors.

### Phase 6 — Stronger BEV representation (future)

- Compare the geometric BEV projection with a learned camera-to-BEV approach.
- Only attempt Lift-Splat-Shoot or a similar model when calibration, depth supervision, data, and compute are available.
- Evaluate BEV segmentation/detection, not only visual appearance.

### Phase 7 — Temporal perception (future)

- Add tracking or temporal fusion across consecutive frames.
- Compare single-frame and temporal results, including flicker and track continuity.

### Phase 8 — Decision-model study (future)

- Define a bounded action space, such as KEEP / SLOW / STOP / YIELD.
- Compare rules, XGBoost, and a trained neural policy on identical structured inputs.
- Test Laya/Jev only after verifying its input/output interface and documenting a fair benchmark.
- Measure action quality, safety outcomes, latency, and behavior under uncertain perception.

### Phase 9 — Closed-loop simulation and optional LLM layer (future)

- Integrate with CARLA or another available simulator, ideally on a supported Linux/Windows GPU host.
- Run fixed, repeatable scenarios and report route completion and infractions.
- Add a frontier LLM only for selected high-level cases; measure call frequency, latency, and cost.
- Keep controller and safety override separate from model-generated reasoning.

### Phase 10 — Research-grade evaluation and release (future)

- Run multiple seeds and repeated scenarios.
- Add ablations, held-out environment testing, calibration, and error analysis.
- Publish experiment configs, plots, model/data notes, demo, and a limitations section.

## Measures

| Component | Candidate measures |
|---|---|
| Road/lane segmentation | Per-class IoU, mIoU, precision/recall, confusion matrix |
| Object detection (future) | mAP, per-class AP |
| BEV (future) | BEV IoU or geometric alignment error |
| Temporal module (future) | Flicker, track continuity |
| Uncertainty (future) | Calibration error, risk-coverage curve |
| Driving (future) | Route completion, collisions/infractions, interventions |
| Runtime | Latency, FPS, input resolution, device, model size |

## Initial technology stack

- Python and PyTorch
- PyTorch MPS for supported Apple Silicon operations, with CPU fallback only where needed and documented
- NumPy and OpenCV for data handling and visualization
- Matplotlib for training/evaluation plots
- A small, versioned dataset/config setup
- Streamlit or a notebook/video-based showcase, chosen after the core inference path works
- CARLA only as a later stage if a supported host and suitable GPU access are available

Exact library versions and dataset are to be pinned in the project environment after Phase 0.

## Proposed repository layout

```text
autonomy-research/
├── README.md                 # project charter and roadmap
├── current.md                # living session-by-session state
├── pyproject.toml            # dependencies and tooling
├── configs/                  # experiment configurations
├── data/                     # scripts and dataset instructions; data itself stays untracked
├── models/                   # segmentation, then later detector/temporal/decision models
├── training/                 # training loops, losses, checkpoints metadata
├── evaluation/               # metrics, robustness, and ablations
├── simulation/               # later CARLA/other simulator interfaces
├── demo/                     # visualizer and recorded demo
├── docs/                     # diagrams, experiment notes, report materials
└── outputs/                  # local results; large generated data/checkpoints are not committed
```

Never commit dataset archives, API keys, or large checkpoints by default. Add a `.gitignore` and document how to obtain data.

## Definition of done for the one-month core

- [ ] A reproducible training command and saved configuration.
- [ ] A trained checkpoint and saved validation metrics.
- [ ] Scene-safe train/validation split documented.
- [ ] Ground truth and prediction visualizations.
- [ ] BEV view with calibration/approximation limitations stated.
- [ ] Runtime measured on the M4 and input settings recorded.
- [ ] Demo and report include failures and limitations, not just best examples.
- [ ] Future roadmap clearly separates completed work from planned work.

## Reference links

- [Reference project README](https://github.com/manntalati/autonomous-driving)
- [Reference project architecture plan](https://github.com/manntalati/autonomous-driving/blob/main/architecture_plan.md)
- [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/mps.html)
- [CARLA quick start and requirements](https://carla.readthedocs.io/en/0.9.16/start_quickstart/)
- [nuScenes dataset and usage terms](https://www.nuscenes.org/nuscenes)

## BDD100K prepared subset

The local BDD100K starter set is prepared for **10-class object detection** and **five-class semantic segmentation**. This section supersedes the earlier Phase 0 note that the dataset was still only a candidate. The chosen data are the 10K image package, 2018 2D detection labels, and semantic train-ID masks; no 100K image package or model training is used here. The segmentation reduction is intentionally coarse: it is not a drivable-area or lane-marking target.

### Official sources and terms

- [BDD100K download instructions](https://doc.bdd100k.com/download.html) — package descriptions, links and license/usage terms.
- [BDD100K data download directory](https://dl.cv.ethz.ch/bdd100k/data/) — official package host.
- [BDD100K label format](https://doc.bdd100k.com/format.html) — annotation and segmentation formats.
- [Official BDD100K label definitions](https://github.com/bdd100k/bdd100k/blob/master/bdd100k/label/label.py) — semantic train IDs.
- [Semantic-label usage note](https://doc.bdd100k.com/usage.html) — semantic masks encode the official `train_id` values.

The dataset is subject to BDD100K's license and terms. Check those terms before redistribution or any non-research use.

### Local structure

All paths below are relative to the main working repository unless marked as a sibling input. The reference repository `../autonomous-driving/` was not modified.

```text
<main-project>/
├── data/raw/bdd100k/                 # original ZIPs; gitignored
│   ├── bdd100k_images_10k.zip
│   ├── bdd100k_labels.zip
│   └── bdd100k_seg_maps.zip
├── data/processed/bdd100k/           # generated manifests/audits; gitignored
│   ├── detection_{train,val,test}.csv
│   ├── detection_unmatched_images.csv
│   ├── segmentation_{train,val}.csv           # working, leakage-free splits
│   ├── segmentation_official_train.csv         # preserved 7,000-ID source list
│   ├── segmentation_train_excluded_evaluation.csv
│   ├── class_distribution.csv
│   ├── summary.json
│   ├── missing_files.csv
│   ├── corrupt_files.csv
│   ├── unusable_segmentation.csv
│   └── visualizations/alignment_00067cfb-e535423e.jpg
├── data/bdd100k_dataset.py            # PyTorch datasets and ID mapping
├── data/prepare_bdd100k.py            # reproducible preparation/audit
├── configs/bdd100k.yaml               # class counts and split settings
├── dataset_config.yaml                # handoff config and source/integrity notes
├── dataset_report.md                  # final audit and class distributions
├── dataset_samples/                   # generated visuals; gitignored
├── split_manifest/                    # generated ID lists; gitignored
├── tools/bdd100k_runtime_audit.py     # bounded batched validation/profiling
├── tools/create_bdd100k_deliverables.py # ID lists and visual examples
└── tests/test_bdd100k_dataset.py      # dataset, mapping and split tests

Sibling extracted inputs (read by the script; not copied or re-extracted):
../10k/{train,val,test}/*.jpg
../100k/{train,val,test}/*.json
../bdd100k_seg_maps/labels/{train,val}/*_train_id.png
../bdd100k_seg_maps/color_labels/{train,val}/*_train_color.png
```

`.gitignore` excludes `data/raw/`, `data/processed/`, `dataset_samples/`, `split_manifest/`, and `test_results.txt`. The small `dataset_config.yaml` and `dataset_report.md` are project documentation; raw archives and generated image/ID data remain out of Git. Extracted source trees stay in their sibling folders.

### Image IDs and splits

Detection IDs are matched by exact filename stem. The detection train/validation split uses only the 3,430 matched 10K **train** images; the seeded 80/20 split (`seed=42`) is 2,744 train and 686 validation. The 830 matched 10K **test** images remain in a separate test manifest and all have 2018 test labels. Detection train/validation/test intersections are zero. Of the 3,430 matched training IDs, 2,976 annotations came from the 2018 train-label folder and 454 from its validation-label folder; label-folder origin does not override the image-level split.

| Task / working split | Train | Validation | Test |
|---|---:|---:|---:|
| Detection labeled images | 2,744 | 686 | 830 |
| Semantic segmentation images | 6,314 | 1,000 | 0 masks |

The segmentation source still has the full official **7,000 train / 1,000 validation** IDs. For combined evaluation, the working segmentation-train manifest excludes the 686 detection-validation IDs; the official 1,000-ID segmentation validation remains unchanged, and all 830 detection-test IDs remain held out. The complete 7,000-ID official training list is retained in `split_manifest/segmentation_official_train_ids.txt`; the 686 exclusions are in `split_manifest/segmentation_excluded_combined_evaluation_ids.txt`.

Across tasks, all training-versus-evaluation intersections are **0**; combined validation-vs-test overlap is **0**. The 2,744 IDs shared by detection train and segmentation train are allowed same-split sharing. Combined unique-ID counts are 6,314 train, 1,686 validation, and 830 test. `split_manifest/manifest_summary.json` contains every within-task and cross-task intersection count, and `build_bdd100k_dataloaders()` rejects unsafe stale manifests before returning loaders.

The 10K image folders contain 7,000 train, 1,000 validation and 2,000 test images. **5,740 10K images have no matching 2018 detection JSON**: 3,570 train, 1,000 validation and 1,170 test. They are listed separately in `detection_unmatched_images.csv`; this is the expected coverage gap between these selected packages, not a missing-file or corrupt-image count. The 2018 label archive contains 70,000 train, 10,000 validation and 20,000 test JSONs.

After excluding the 686 detection-validation IDs, the working segmentation manifests contain 6,314 train and 1,000 validation aligned image-mask pairs. All **8,000 official** segmentation image-mask pairs were still exhaustively checked during preparation. Ten source JPEGs in the complete 10K image set are stored as portrait 720×1280 frames; the loader rotates those counter-clockwise to the common 1280×720 orientation. Six are in segmentation train/validation, and all six image/mask IDs and dimensions were checked after normalization.

### Detection taxonomy and distribution

The 2018 JSON box categories observed in the matched images are exactly the ten supported classes below. IDs are zero-based in this project-specific order. Box coordinates are clamped to image bounds; 21 non-finite, malformed or degenerate boxes were excluded. There were no unknown box categories, empty labeled images, or annotation-name mismatches.

| ID | Class | Train boxes | Val boxes | Test boxes |
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

`class_distribution.csv` also records the number of images containing each category. Detection manifests contain `box_count`, `invalid_box_count`, the original 10K/2018 split provenance, and per-image class counts.

### Semantic train-ID mapping

The source masks use the official BDD100K semantic `train_id` values. The project mapping is deliberately many-to-one:

| Official train ID | Official label | Project class ID |
|---:|---|---:|
| 0 | road | 1 — road |
| 1 | sidewalk | 2 — sidewalk |
| 2–10 | building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky | 0 — background |
| 11–12 | person, rider | 4 — vulnerable road users |
| 13–18 | car, truck, bus, train, motorcycle, bicycle | 3 — vehicles |
| 255 | official void / ignore | 255 — ignore |
| any other value | unknown / invalid | 255 — ignore |

The full source-name list is in `data/bdd100k_dataset.py`; the exact lookup is in `configs/bdd100k.yaml` and `summary.json`. Unknown raw pixel values and official void are excluded from loss/metrics as 255, never relabeled as background. The observed train-ID masks contained only IDs 0–18 and 255; there were no unexpected values.

| Mapped pixel class | Working train pixels | Validation pixels |
|---|---:|---:|
| Background | 2,798,837,238 | 476,506,812 |
| Road | 1,252,441,133 | 199,063,763 |
| Sidewalk | 115,943,431 | 18,895,436 |
| Vehicles | 564,629,111 | 99,072,672 |
| Vulnerable road users | 15,657,964 | 2,370,886 |
| Void / ignore (255) | 1,071,473,523 | 125,690,431 |

Training-pixel counts are computed from the 6,314-ID working segmentation train split after holdout exclusion; the complete official 7,000-ID source totals remain in `summary.json`.

### Audit and visual verification

The preparation script decoded all 10,000 source images and the official train/validation mask files. There were **0 missing source files, 0 corrupt files, and 0 unusable image-mask pairs** after orientation normalization. `missing_files.csv`, `corrupt_files.csv`, and `unusable_segmentation.csv` provide machine-readable audit outputs; `summary.json` includes counts, category distributions, ID intersections, orientation corrections and archive checksums.

The combined four-panel visual sample shows image ID `00067cfb-e535423e`, its original image, 33 detection boxes, mapped mask and overlay at 1280×720. Additional separated original/box/overlay triplets for three exact IDs are in `dataset_samples/`; IDs, annotation names and dimensions are asserted in `samples.json`.

All three ZIPs passed a fresh `zipfile.testzip()` scan. A second pass mapped every one of the 126,000 non-directory archive entries to its extracted file and compared byte length plus CRC32: 10,000 images, 100,000 label JSONs and 16,000 segmentation PNGs; zero missing members and zero size/CRC differences across 3,136,138,559 uncompressed bytes. Label and mask macOS provenance attributes point to the legacy Berkeley host `http://128.32.162.150/bdd100k/`; the image ZIP has no `kMDItemWhereFroms` attribute, although the earlier download record gives the matching legacy image URL. The image archive's local MD5 `9c5ee9df0097cc1ab799f58feb776611` differs from the official documentation reference `08f26aecceda982568063d3d5873378e`. The extracted data matches this archive by size/CRC and the ZIP's own CRCs pass, so there is no evidence of extraction/archive corruption; the exact image-package release identity remains unresolved and the MD5 mismatch alone does not prove corruption.

The preparation audit decoded all 10,000 images and checked all 8,000 official segmentation pairs: zero missing/corrupt image or mask files, zero unusable pairs, zero unexpected train IDs. The 5,740 images without matching 2018 detection labels are an expected package coverage gap; 21 malformed/degenerate boxes were excluded. The 15 focused tests pass (see `test_results.txt`).

An exhaustive DataLoader pass over all 12,260 task-split records previously exceeded the 300-second limit; no full-dataset tensors were retained. cProfile identified per-mask `torch.unique` validation as a hot path and it was replaced with NumPy membership validation. The latest seeded bounded run checked 64 samples from each of five loaders (320 task-samples, batches of 8) and passed box/label and image-mask checks. With `num_workers=0`, measured throughput was 126.95/120.29/119.96 samples/s for detection train/validation/test and 26.38/28.53 samples/s for segmentation train/validation; two-worker fresh-loader startup reduced throughput to 4.96–5.69 samples/s. Peak RSS was 543.9 MB. This is sampled validation, not an exhaustive epoch. The audit and timings are in `data/processed/bdd100k/runtime_validation.json`.

### Prepare and load

Run preparation and tests from the main repository root:

```bash
cd /Users/yuvrajsingh/Downloads/Project/Deep-Learning-Based-Autonomous-Driving-Perception-and-Road-Risk-Analysis-System
python3 -m data.prepare_bdd100k --seed 42 --val-fraction 0.2
python3 tools/create_bdd100k_deliverables.py
python3 tools/bdd100k_runtime_audit.py --samples-per-split 64 --batch-size 8 --workers 0 2 --seed 2026
python3 -m unittest -v tests.test_bdd100k_dataset
```

The preparation command is repeatable with the same extracted inputs, seed and validation fraction; it recreates manifests, audits and the alignment visual under the ignored `data/processed/bdd100k/` directory. The deliverables script writes the requested split ID lists and three visual triplets. The runtime audit samples each task split in bounded batches; it is not a full-epoch scan. To construct loaders (this does not start training):

```python
from data.bdd100k_dataset import build_bdd100k_dataloaders

loaders = build_bdd100k_dataloaders(batch_size=4, image_size=(448, 800), num_workers=0, seed=42)
images, targets = next(iter(loaders["detection"]["train"]))
seg_images, masks = next(iter(loaders["segmentation"]["train"]))
# Detection test is separate: loaders["detection"]["test"]
```

No model training, checkpoint creation, architecture change, or reference-repository edit was performed.
