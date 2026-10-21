# Grounded-DINO + SAM2 small-object evaluation

This evaluator adds an open-vocabulary small-object branch to the existing
OneMap semantic map:

```text
RGB -> Grounding-DINO text prompts -> boxes -> SAM2 box masks
    -> depth + pose projection -> local terrain support estimate
    -> compact-obstacle gate -> 3-D cross-view association
    -> small-object evidence, with large > small > terrain priority
```

Grounding-DINO and SAM2 are used only for the small-object branch. Terrain and
large-object labels come from the same OneMap map used by the base comparison.
The local height reference is estimated from depth points whose current OneMap
prediction is terrain within a 0.30 m XY neighborhood. The prediction path
does not read `design.json` floor height, semantic GT, annotated mesh, or
FAST-LIO ground truth.

The implementation uses the Transformers checkpoints
`IDEA-Research/grounding-dino-tiny` and `facebook/sam2.1-hiera-tiny`, with
revisions recorded in the result JSON. Independent text prompts are used for
`banana`, `tin can`, and `cracker box`; ambiguous mixed labels are rejected.
Overlapping 448 px tiles are used for small-object detection. SAM2 masks are
projected with the 640 px RGB-D intrinsics and the eight Habitat camera poses.

## Reproduction

Use the project Conda environment (`onemap`) and run the pure OneMap base first:

```bash
python reproduction/map_segmentation/run_3d_instance_pose_eval.py \
  --base reproduction/map_segmentation/multi_room \
  --out /tmp/onemap-base-no-privilege \
  --rooms 9 10 11 17 \
  --no-sam --no-instance-fusion
```

Then run the cached or fresh Grounded-DINO + SAM2 evaluation:

```bash
python reproduction/map_segmentation/run_grounded_sam2_eval.py \
  --base reproduction/map_segmentation/multi_room \
  --out /tmp/onemap-grounded-sam2 \
  --base-pred-dir /tmp/onemap-base-no-privilege \
  --rooms 9 10 11 17 \
  --detector-threshold 0.24 \
  --text-threshold 0.18 \
  --tile-size 448 --tile-overlap 0.25 \
  --min-detection-score 0.32 \
  --min-support 2
```

`--min-support 1` gives the per-view proposal result; `--min-support 2`
requires the same 3-D proposal to be observed from two views. Both variants
are written to the summary so recall/precision trade-offs are visible.

## Strict four-room result

The primary protocol uses Rooms 9, 10, 11, and 17, eight views per room,
camera height 0.30 m, forward offset 0.30 m, and Habitat design poses. It
evaluates all valid semantic-elevation GT cells; an unmapped prediction is an
error. Terrain means floor and stairs; small means banana, tin can, and
cracker box; large is every other mapped class.

| Variant | Overall accuracy | Terrain P / R | Small P / R / F1 | Large P / R |
| --- | ---: | ---: | ---: | ---: |
| Pure OneMap base | 67.44% | 77.42% / 55.12% | 9.82% / 41.54% / 15.88% | 73.63% / 78.28% |
| DINO + SAM2 per-view | 67.34% | 77.52% / 54.79% | 10.77% / 49.23% / 17.68% | 73.63% / 78.28% |
| DINO + SAM2, 2-view consensus | 67.33% | 77.44% / 54.79% | 10.54% / 47.69% / 17.27% | 73.63% / 78.28% |

The strict result improves small-object recall by 7.69 percentage points in
the per-view variant, while precision remains close to the base because the
zero-shot detector still produces false positives. The large-object metrics
are unchanged because small evidence cannot overwrite a large label. The
cross-view requirement is conservative on this sparse eight-view sequence and
loses two additional true small-object cells; it is kept as an ablation rather
than the default.

Per-class formulas are:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

The complete per-room and pooled confusion matrices are in
`fastlio_results/grounded-sam2-three-class-20261022.json`. The generated
visuals contain RGB boxes/masks, GT map, prediction map, and error map for all
four rooms.

## Literature relation and limits

The detector/mask split follows Grounding DINO's open-set text-conditioned
detection and SAM 2's box-prompt segmentation. The 3-D association follows
the 2-D-guided proposal idea in Open3DIS, adapted to this eight-view RGB-D
OneMap evaluator. This is a focused robotics experiment, not a reproduction
of those papers' benchmark scores.

The data are four rooms from one HM3D building and use Habitat design poses;
no FAST-LIO pose file is present in this run. The small-object GT contains 65
valid cells, so precision/recall should be read together with TP/FP/FN counts.
