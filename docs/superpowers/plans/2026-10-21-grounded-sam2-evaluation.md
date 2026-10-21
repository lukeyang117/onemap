# Grounded-SAM2 Three-Class Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible Grounding DINO + SAM2 + multi-view 3D small-object pipeline and compare its three-class precision, recall, F1, coverage, and confusion counts against existing OneMap baselines.

**Architecture:** Keep detection, box-prompted SAM2 inference, RGB-D projection, association, and scoring in a new evaluation module, with deterministic pure helpers separately testable. Use the existing map evaluator for metric semantics and the existing four-room prepared dataset; create per-variant cached 2D outputs, fused maps, JSON, and visual comparisons.

**Tech Stack:** Python 3.10, PyTorch 2.2.2/CUDA, Transformers 4.57.3 Grounding DINO, SAM 2.1, NumPy, OpenCV, Pillow, pytest, Habitat prepared RGB-D/GT files.

## Global Constraints

- Main scenes are Rooms 9, 10, 11, and 17, with the existing eight poses per room.
- Camera height is 0.30 m and forward offset is 0.30 m, horizontal view.
- Semantic priority remains large object > small object > terrain (floor/stairs).
- Primary experiment uses Habitat design poses; do not call them FAST-LIO estimates.
- All variants score the same valid GT cells; uncovered cells are errors.
- Report per-room and pooled precision, recall, F1, counts, coverage, and confusion matrix for all three classes.
- Never use GT to generate boxes, masks, class labels, or poses.

---

## File Structure

- Create `reproduction/map_segmentation/grounded_small_objects.py`: detector and SAM2 adapters plus box/tile/mask geometry helpers.
- Create `reproduction/map_segmentation/run_grounded_sam2_eval.py`: inference caching, RGB-D projection, multi-view object association, map rasterization, scoring, and visual output.
- Create `reproduction/map_segmentation/tests/test_grounded_small_objects.py`: deterministic unit tests for geometry, association, and priority.
- Create `reproduction/map_segmentation/GROUNDED_SAM2_EVALUATION.md`: pipeline, setup, run commands, and caveats.
- Create `reproduction/map_segmentation/fastlio_results/grounded-sam2-three-class-20261021.json`: model revisions and per-room/pooled results.
- Do not alter the existing primary evaluator unless an integration seam is needed; if so, cover that seam with a test.

## Task 1: Dependency and model smoke test

**Files:**
- Modify environment only: `onemap` Conda environment.

- [ ] Install compatible Transformers and Accelerate in the `onemap` environment.
- [ ] Load `IDEA-Research/grounding-dino-tiny` and `facebook/sam2.1-hiera-tiny` from their documented APIs, reporting exact revision, device, and memory.
- [ ] Run a one-image prompt smoke test with only a real view image; confirm boxes, phrase scores, and SAM2 masks are non-empty or explicitly record no detections.

## Task 2: Geometry and association helpers (TDD)

**Files:**
- Create `reproduction/map_segmentation/grounded_small_objects.py`
- Test `reproduction/map_segmentation/tests/test_grounded_small_objects.py`

- [ ] Test clipping and inverse-mapping boxes from overlapping tiles to original pixel coordinates.
- [ ] Test projecting a binary mask through a depth image and pose into finite XYZ points.
- [ ] Test associating two views with overlapping 3D points and keeping spatially distinct objects separate.
- [ ] Test small-object rasterization does not overwrite large labels and does overwrite terrain when supported.
- [ ] Implement the minimal pure helpers until the tests pass.

## Task 3: Detection and SAM2 inference

**Files:**
- Modify `reproduction/map_segmentation/grounded_small_objects.py`

- [ ] Add a Grounding DINO Transformers adapter that accepts prompt phrases, tiles the image with overlap, maps boxes back to the original frame, performs NMS, and returns class phrase and detector score.
- [ ] Add a SAM2.1 box-prompt adapter returning the highest-quality mask and mask-quality score per detection.
- [ ] Save lossless per-view annotations and cache metadata; fail explicitly if model setup or inference fails.
- [ ] Run detection-only, then detector+SAM2 on one real view from each room and inspect candidate count, class text, mask area, and overlays.

## Task 4: Multi-view mapping and three-class scoring

**Files:**
- Create `reproduction/map_segmentation/run_grounded_sam2_eval.py`
- Modify `reproduction/map_segmentation/grounded_small_objects.py` only for tested missing helpers.

- [ ] Cache and load each room's eight predictions with their exact view IDs.
- [ ] Project per-view SAM masks through depth and the matching Habitat pose to 3D.
- [ ] Associate instances using 3D overlap and centroid gating; accumulate distinct-view support and detector/mask confidence.
- [ ] Evaluate variants: OneMap baseline, existing ConvNeXt-L recovery, DINO+SAM2 per-view projection, and DINO+SAM2 with cross-view 3D association.
- [ ] Build every prediction map with large > small > terrain priority; score all valid GT cells.
- [ ] Save per-room and pooled counts, metrics, confusion matrices, coverage, model revisions, commands, and pose source.
- [ ] Save side-by-side per-room visual outputs: RGB with boxes/masks, GT map, prediction map, and error map.

## Task 5: Reproducibility and review

**Files:**
- Create `reproduction/map_segmentation/GROUNDED_SAM2_EVALUATION.md`
- Create `reproduction/map_segmentation/fastlio_results/grounded-sam2-three-class-20261021.json`

- [ ] Run unit tests and compile checks.
- [ ] Independently recompute all pooled/per-room class metrics from saved confusion matrices and assert equality with the report.
- [ ] Check the saved visualizations and one cached mask per room.
- [ ] Document any checkpoint/network/hardware limitations and keep the baseline results if the new model is blocked.
- [ ] Stage and commit only the new pipeline, tests, documentation, and result artifacts.
