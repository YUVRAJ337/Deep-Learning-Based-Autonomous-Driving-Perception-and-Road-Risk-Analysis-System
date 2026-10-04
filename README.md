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
