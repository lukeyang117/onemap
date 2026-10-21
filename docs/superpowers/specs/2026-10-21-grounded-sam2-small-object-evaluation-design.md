# Grounded-SAM2 small-object mapping evaluation

## Goal

Improve and measure small-object semantic mapping precision and recall while
reporting terrain, large-object, and small-object metrics under the same
eight-view Habitat protocol.

## Current evidence

The current four-room map evaluation has 65 GT small-object cells, 32 true
positive cells, 249 false positives, and 33 false negatives. This gives
11.39% small-object precision and 49.23% recall. Room 9 contains no small
object GT cells, so its small recall is undefined; its false predictions still
count against pooled precision. The current candidate crop threshold sweep
showed that increasing the crop small-pixel threshold from 0.20 to 0.30 in
Room 10 lowered both precision and recall, so threshold-only tuning is not a
sufficient fix.

## Proposed pipeline

1. Keep the OneMap ConvNeXt-L frontend as the baseline source for terrain and
   large-object labels.
2. Run `IDEA-Research/grounding-dino-tiny` through Transformers on each
   robot-height RGB view using small-object language prompts. Use overlapping
   image tiles so small distant objects are not reduced to a few pixels in
   the detector input.
3. Prompt `facebook/sam2.1-hiera-tiny` with each detector box to refine a
   per-view object mask. SAM is only the mask refiner; class identity and
   objectness come from the detector/text match.
4. Project mask pixels through the matching depth image and camera pose into
   3D. Associate observations across views using 3D point/voxel overlap and
   centroid distance, retain per-view support and detector scores, and reject
   inconsistent clusters.
5. Rasterize accepted 3D small-object evidence into semantic map cells while
   preserving the priority `large > small > terrain`. Existing terrain and
   large labels remain from the same OneMap semantic projection.
6. Score all valid GT cells using the existing three-class labels. Report
   confusion counts, pooled and per-room precision/recall/F1 for terrain,
   large, and small. Missing predictions remain errors; do not score only
   covered predictions.

## Approach choice

- **Chosen: Hugging Face Transformers implementations of Grounding DINO Tiny
  and SAM 2.1 Tiny.** This avoids adding separate compiled detector and mask
  repositories to the existing Habitat environment, and provides direct APIs
  for zero-shot detection and box-prompt masks. Both downloaded model revisions
  will be recorded to make the result reproducible.
- **Alternative: install the original GroundingDINO and SAM2 repositories.**
  This follows the authors' implementation more directly, but brings separate
  setup paths and version-sensitive native dependencies. It can be revisited
  if the Transformers backend fails numerical or mask-quality checks.
- **Alternative: use YOLO-World with SAM2.** This is a useful later speed
  baseline, but Grounding DINO is retained first because the proposed
  experiment specifically needs natural-language proposals for varied small
  obstacles and must match the agreed design.

## Experimental comparison

Run the original OneMap, current ConvNeXt-L crop recovery, detector+SAM2
without 3D consensus, and detector+SAM2 with 3D consensus on Rooms 9, 10, 11,
and 17, with the same eight views, poses, depth, GT, map resolution, and class
priority. Room 11's stair-specific plane prior is a separate ablation and is
not silently mixed into the main pooled result. All primary results use
Habitat design poses because no FAST-LIO pose files are present for these
four rooms.

## Implementation boundaries

- Add an isolated detector/SAM2 inference module and a command-line evaluator;
  avoid coupling new model dependencies into the existing OneMap runtime.
- Pin Transformers to `4.57.3`, retain the current PyTorch `2.2.2` runtime, and
  record the resolved Hugging Face model revisions in result JSON.
- Cache per-view boxes, masks, text labels, and confidence for repeatable
  fusion and visual audit.
- Pin model IDs/revisions and record software versions in result JSON.
- If a model checkpoint cannot be downloaded or loaded, stop with a clear
  error. Never substitute synthetic detections or GT-derived masks.
- The final result must include three-class per-room and pooled metrics, a
  confusion matrix, coverage, and visual comparisons of RGB, boxes/masks,
  GT, and predicted maps.

## Validation

- Unit-test box clipping/tile coordinate restoration, mask-to-3D projection,
  cross-view association, and class-priority rasterization.
- Run inference on a held-out view and inspect saved detector boxes and SAM2
  masks before full-room evaluation.
- Recompute the JSON metrics independently from saved prediction and GT grids.
- Compare every pipeline variant on the same valid GT cells and views.

## Literature grounding

- Grounding DINO provides open-set object detection from category names and
  referring expressions: https://arxiv.org/abs/2303.05499
- SAM 2 supplies promptable image/video masks and supports box prompts:
  https://arxiv.org/abs/2408.00714
- Open3DIS aggregates 2D masks across views into geometrically coherent 3D
  proposals, a relevant model for instance association:
  https://arxiv.org/abs/2312.10671

The experiments in this repository are a focused adaptation of these ideas,
not a reproduction of the published benchmarks or their reported scores.
