# Current Project Status

> Update this file at the end of each work session. Keep completed facts separate from plans and assumptions.

**Last updated:** 2026-10-05  
**Overall status:** Planning  
**Current phase:** Phase 0 — scope, environment, and data selection  
**Time available for first deliverable:** Approximately one month  
**Primary machine:** MacBook Air M4  
**Current demo target:** Simulation/recorded-data perception demo; no real-vehicle control

## Project intent

Build a modular autonomous-driving research platform to learn PyTorch and evaluate how visual perception quality and uncertainty affect driving decisions. The first milestone is a trained road/lane segmentation model with quantitative evaluation and a BEV visualization. The longer roadmap includes detection, learned BEV, temporal perception, decision-model comparisons (including Laya/Jev if compatible), and optional simulator/LLM integration.

## Completed

- [x] Read the shared ChatGPT project discussion for the desired system direction.
- [x] Read the reference repository README and its detailed architecture plan.
- [x] Agreed that the reference architecture is a long-term blueprint; the first month should deliver a complete vertical slice.
- [x] Confirmed the available local computer is a MacBook Air M4; use PyTorch MPS where supported.

## Decisions made

- First milestone is road/lane segmentation + metrics + BEV visualization.
- Use a small, documented dataset and scene-level split; verify labels before committing to architecture.
- Do not make local CARLA installation a one-month blocker; official packaged CARLA requirements target Windows/Ubuntu and recommend a dedicated GPU.
- Keep Laya/Jev, learned BEV, temporal fusion, and LLM integration in the future roadmap until the core works.
- Use the reference repo for design ideas, not as proof that its reported results have been independently reproduced.
- No dataset has been selected or downloaded yet.

## Current unknowns / blockers

- Final dataset and exact subset/label download. BDD100K is the proposed lead candidate; pending a sample audit.
- Available disk space and unified memory configuration on the M4 Air.
- Campus machine OS/GPU/access; this may enable CARLA on supported Linux/Windows hardware.
- Course-specific expectations for from-scratch training versus fine-tuning.
- Preferred demo format: Streamlit, notebook, or rendered video.

## Next actions

1. Confirm course requirements for a trained model and acceptable datasets.
2. Audit a small BDD100K drivable-area/lane-marking sample: image-mask alignment, class encoding, label quality, terms, and storage.
3. Check Python/PyTorch/MPS environment and free disk space on the Mac; check campus machine OS/GPU and access.
4. Decide BDD100K versus a CARLA-generated dataset based on the sample audit and campus compute.
5. Run a minimal data-loading/training smoke run, then begin Phase 1 after viewing sample images and labels.

## Session log

### 2026-10-05 — Project charter started

- Drafted the project vision, one-month core, architecture, metrics, phase roadmap, and proposed repository layout in `README.md`.
- Compared dataset candidates; BDD100K is now the proposed lead for direct drivable-area/lane labels, with nuScenes and campus CARLA as alternatives for later calibrated BEV or controlled simulation work.
- Created this status file as the session-to-session record.
- No code, dataset, model training, or simulator setup has started.
- Next: verify dataset/labels and machine constraints before implementation.

## Session update template

For each work session, append a dated entry with:

- **Goal:** What we intended to do.
- **Done:** Files, experiments, or decisions completed.
- **Evidence:** Commands run, metrics, screenshots, or links to outputs.
- **Issues:** Errors, limitations, or unresolved questions.
- **Next:** The next 1–3 concrete actions.
