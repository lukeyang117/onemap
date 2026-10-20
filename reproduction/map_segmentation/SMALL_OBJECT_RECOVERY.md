# Small-object recovery

This patch adds a geometry-seeded recovery path for small obstacles that are
missed by full-frame ConvNeXt-L semantic projection.

The path is enabled with `--small-object-recovery` in
`run_3d_instance_pose_eval.py`:

1. Project each RGB-D view into the world frame and propose connected regions
   whose height is between `0.02 m` and `0.45 m` above the fitted floor plane.
2. Re-run ConvNeXt-L on a high-resolution crop around each proposal. The crop
   is accepted when at least `20%` of its pixel labels are small-object classes
   (`40, 41, 42`).
3. Fuse each accepted proposal with a categorical posterior using
   `bayesian_update`. The final semantic priority remains large object > small
   object > floor/stair.

The deployable diagnostic command is:

```bash
python reproduction/map_segmentation/run_3d_instance_pose_eval.py \
  --base reproduction/map_segmentation/multi_room \
  --out /tmp/recovery-three \
  --rooms 9 10 17 \
  --no-sam --no-instance-fusion \
  --small-object-recovery --small-support-views 1
```

`--no-instance-fusion` is intentional for the current implementation. The
existing instance overwrite path spreads a single crop decision over a large
projected region and creates more large-object and small-object false
positives than it recovers. The recovery path still fuses evidence at the
proposal level, but does not let one proposal overwrite unrelated map cells.

## Reproduced results

The numbers below use the same eight robot-height views for each room and
Habitat design poses. They are a semantic-map diagnostic; no FAST-LIO pose
file was available in this run.

| rooms | configuration | overall accuracy | small recall | small precision |
| --- | --- | ---: | ---: | ---: |
| 9, 10, 17 | baseline, no recovery | 67.4924% | 30.4348% | 8.2353% |
| 9, 10, 17 | recovery, proposal fusion | 67.5506% | 41.3043% | 10.7955% |
| 11 | recovery + online/multi-plane ground prior | 65.5087% | 73.6842% | 20.8955% |

For reference, Room 10 with MobileSAM and instance fusion after the rejection
fix reaches 68.5319% overall accuracy, 54.5455% small recall, and 4.8780%
small precision. It recovers more small cells but produces too many small
false positives, so the no-instance-fusion configuration remains the better
default for deployment.

The three-room recovery change is +10.87 percentage points in small-object
recall and +2.56 points in small-object precision, while overall accuracy is
nearly unchanged because small-object cells are a small fraction of the map.
The full per-room JSON and candidate counts are recorded in
`fastlio_results/small-object-recovery-summary.json`.

Unit tests for the geometry mask, crop likelihood, posterior update, and
semantic priority are in `tests/test_small_object_fusion.py`.
