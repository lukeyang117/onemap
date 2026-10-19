# ROS2 Humble FAST-LIO replay

The native Ubuntu installation used for this replay is Ubuntu 22.04 with ROS2
Humble, PCL 1.12 and Eigen 3.4. The ROS2 port of FAST-LIO and Livox ROS2
driver compile in `/tmp/fastlio_build_ws` with `colcon build`.

`write_fastlio_ros2_bag.py` converts the Habitat-derived LiDAR/IMU export to a
ROS2 SQLite/ CDR bag. The PointCloud2 layout matches FAST-LIO's aligned
Velodyne point type: `x,y,z` at offsets `0,4,8`, `intensity` at `16`, `time` at
`20`, and `ring` at `24` with a 32-byte point stride. This matters because a
12/24-byte layout is silently read as an empty or corrupt PCL cloud.

The bag opens with `ros2 bag info` and FAST-LIO receives all 32-ring scans and
IMU messages. The NVIDIA CUDA backend was unstable on the first synthetic
replay, so the node was rebuilt with `-DFASTLIO_ENABLE_CUDA=OFF`. With a denser
trajectory (`0.04 m` interpolation spacing), CPU FAST-LIO ran through 655
odometry messages without a crash. The simulator now records the explicit
Habitat-to-FAST-LIO axis matrix, a camera-origin LiDAR, and the IMU-to-LiDAR
translation `[0.3, 0, 0.3]` m. The estimated trajectory was rigidly aligned to
Habitat with a Kabsch fit and the eight mapping views were then replayed
through OneMap. A near-range test with `blind=0.1` and 0.2 m map filtering was
also run; it did not recover tracking in the two geometrically sparse rooms.
The exporter now stores the depth row as an explicit `ring` field, sorts
sector points by horizontal angle, and supports a four-sector/8-line profile.
The `--fastlio` motion profile adds a 1.5 s stationary initialization and
smoothstep acceleration between views.

The rolling-scan revision also reprojects each return using the interpolated
pose at its per-point time and writes that time into PointCloud2. A pure
FAST-LIO Room 9 replay still diverged (about 1.5 km over the 67 s sequence),
so this revision is a sensor-model diagnostic and is not used for OneMap
accuracy claims. The failure indicates that a depth snapshot plus synthetic
motion is still not equivalent to a real spinning LiDAR scan; all three-room
accuracy numbers remain withheld until FAST-LIO itself produces a bounded
trajectory.

`generate_raycast_lidar.py` now produces collision-mesh raycast scans with
per-ray azimuth time, explicit rings, interpolated rolling poses, and matching
IMU export. The first full Room 9 pure replay still diverged to about 3.9 km
over the sequence; this failure is retained as the current diagnostic and no
ICP or simulator pose was used to replace it.

Room 10, observed-only scoring:

| map pose source | semantic accuracy | floor/stairs | small objects | large objects | height MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Habitat ground truth | 61.25% | baseline | 0/22 | baseline | 111.18 cm |
| FAST-LIO CPU estimate, old axes | 0.28% | 0/2139 | 0/22 | 10/1449 | 106.07 cm |
| FAST-LIO CPU estimate, corrected axes/extrinsic | **14.46%** | 432/2139 | 1/22 | 89/1449 | 133.23 cm |

Across the three rooms with the corrected pipeline, semantic accuracy is Room
9 **0.00%**, Room 10 **14.46%**, and Room 17 **0.00%**. Room 9 and Room 17
produce no valid predicted elevation cells; their logs report repeated `No
Effective Points`, so their height MAE is undefined rather than treated as
zero. The control maps are 28.56%, 61.25%, and 67.43%, respectively. The
Room 10 visualization is
`reproduction/map_segmentation/fastlio_results/room-10/semantic-elevation-comparison-axis.png`.

The scan-to-scan fallback (`scanmatch_fallback.py`) is available as a
diagnostic watchdog for frames where FAST-LIO reports no effective points. Its
corrected accumulation direction and body-axis conversion produce Room 9
**11.86%**, Room 10 **35.65%**, and Room 17 **9.99%**. Room 10 floor/ground
agreement reaches 49.14%. Small-object agreement remains 0/22 in Room 10 and
0/24 in Room 17, so the next gain must come from semantic fusion and object
coverage rather than another pose-axis change. See
`fastlio_results/summary-axis-corrected.json` for the per-class counts.

The final raycast correction uses the transpose for FAST-LIO-to-Habitat ray
directions. With pure FAST-LIO Odometry and MobileSAM instance refinement, the
production-style three-room result is:

| room | semantic accuracy | small-object recall |
| --- | ---: | ---: |
| Room 9 | 63.02% | no small-object GT |
| Room 10 | 68.25% | 31.82% |
| Room 17 | 45.60% | 16.67% |
| pooled | **62.01%** | — |

No ICP or simulator pose replaces FAST-LIO in this result. The final report is
`fastlio_results/summary-fastlio-mobilesam.json`.

Adding a two-view support gate for small SAM instances reduces single-view
false positives. The updated pooled accuracy is **63.18%**: Room 9 65.37%,
Room 10 68.37%, and Room 17 46.62%. The updated report is
`fastlio_results/summary-fastlio-mobilesam-consistent.json`.

Example:

```bash
source /opt/ros/humble/setup.bash
source /tmp/fastlio_build_ws/install/setup.bash
python reproduction/map_segmentation/write_fastlio_ros2_bag.py \
  --input /tmp/habitat-lio-room10 --output /tmp/habitat-lio-room10-ros2
ros2 bag play /tmp/habitat-lio-room10-ros2
```

## Randomized robustness evaluation (2026-10-18)

The earlier 62–63% table above is retained as the historical `rayinst`
experiment. The numbers below are a fresh rerun from the current FAST-LIO pose
exports and are the values to use for robustness comparisons; the two result
sets should not be mixed.

The complete evaluation now uses the FAST-LIO pose JSON, ConvNeXt-L dense
front-end classes, MobileSAM instance refinement, and the large > small >
floor fusion rule. Ground-truth poses are not used by the mapper; GT is read
only by the final semantic-elevation scorer. The mapper accepts a subset of
the original view indices, so a dropped view cannot silently be paired with a
different cached feature image.

The clean full-map accuracies from the current reproducible raycast pose
exports are Room 9 **56.74%** (coverage 84.68%, observed-region accuracy
67.00%), Room 10 **52.30%** (93.82%, 55.74%), and Room 17 **46.62%** (80.33%,
58.03%); pooled accuracy is **53.01%**. The first number counts every valid GT
cell, including cells that the eight views did not cover. Coverage and
observed-region accuracy are reported separately so pose/coverage failures are
not confused with class errors.

Randomized perturbations use one deterministic trial per condition in this
smoke test: 5 cm translation plus 2° rotation noise, one randomly removed
view, 10% depth dropout plus 1 cm depth noise, and the same RGB-D corruption
with RGB gain jitter. Room 10 baseline scores are 52.30% clean, 50.86% under
pose jitter, 52.74% with one view removed, 52.27% under depth corruption, and
52.47% under combined RGB-D corruption. Room 9 scores are 56.74%, 60.32%, and
57.47% for clean, pose jitter, and depth corruption; Room 17 scores are
46.62%, 46.74%, and 47.00% for the same three conditions. A single trial is a
diagnostic, not a confidence interval; use `--trials 3` or more before making
statistical claims.

For the adaptive Room 10 chain, three repeated trials give 56.07% ± 0.00%
clean (the same deterministic input), 53.78% ± 1.30% under pose jitter with
worst case 52.08%, and 55.97% ± 0.18% under depth corruption with worst case
55.76%. Mean coverage in those runs is 89.25%, 90.81%, and 90.36%,
respectively.

The new self-consistency gate compares each view's projected coarse semantic
labels with the other views, without GT. It only activates when the
consistency distribution has a high median and a clear low tail, avoiding
coverage loss in globally inconsistent rooms. It keeps views 0, 1, 2, 5, 6,
7 in Room 10 and raises clean accuracy to **56.07%**; Room 9 and Room 17 keep
all eight views at 56.74% and 46.62%. In Room 10 the adaptive-gate robustness
scores are 56.07% clean, 55.24% pose jitter, 55.51% one-view drop, 56.09%
depth corruption, and 56.26% combined RGB-D corruption.

The per-view ablation found a real failure mode: Room 10 accuracy is 55.79%
without view 4, 48.86% without view 7, and 49.31% without view 2. This is why
the gate is relative and conservative rather than a fixed “always drop two
views” rule. Repeated ablations show that fixed view removal harms Room 9 and
Room 17.

Re-run the tests with:

```bash
python reproduction/map_segmentation/run_3d_instance_pose_eval.py \
  --base /path/to/rayinst-all --out /tmp/one-map-result --rooms 10 \
  --adaptive-consistency
python reproduction/map_segmentation/robustness_instance_eval.py \
  --base /path/to/rayinst-all --pose-file /tmp/rayroom10-fastlio8.json \
  --room 10 --out /tmp/robust --trials 3 --adaptive-consistency
```

Machine-readable results and figures are in
`fastlio_results/robustness-summary.json`,
`fastlio_results/robustness-clean-comparison.png`, and
`fastlio_results/robustness-room10-conditions.png`.

## Module ablation and current improvement path (2026-10-19)

The pipeline was ablated on the same Room 10 FAST-LIO pose sequence. The
complete chain scores 52.30%. Removing MobileSAM changes it to 52.41%, while
allowing single-view small-instance writes changes it to 52.52% but leaves
small-object precision at only 5.03%. Disabling the 3-D instance overwrite is
better: 54.90% overall and 22.73% small-object recall. Adding the relative
cross-view consistency gate gives 59.20%. Raising the large-class evidence
fraction to 0.9 gives 60.36%. A 5 cm floor prior gives 63.30% in Room 10, but
it is unsafe as a fixed rule: Room 9 falls to 37.36% because its FAST-LIO
height alignment is different.

The current candidate version therefore enables the floor prior only when the
projected frontend labels contain enough floor evidence, and only uses the
large fraction 0.9 when the consistency gate actually removes a low-tail
view. With this conditional policy, the current three-room results are Room 9
54.53%, Room 10 63.30%, Room 17 47.56%, pooled **56.93%**. Room 9 and Room 17
retain the original large evidence fraction and all eight views.
The 5 cm threshold currently uses the Habitat room floor datum for this
simulation; on the robot it must be replaced by an online ground-plane height
estimate before calling this fully deployment-ready.

Two oracle diagnostics locate the remaining ceiling. With GT poses but the
current ConvNeXt-L semantic labels, the no-instance map reaches 67.52% pooled.
With GT poses and per-view GT semantic labels, it reaches 85.23% pooled. Thus
80% is geometrically possible, but the current 2-D semantic front end and
FAST-LIO pose drift must both improve; changing serialization or adding more
SAM overwrite is not sufficient. The full table is in
`fastlio_results/module-ablation-summary.json`, with a chart in
`fastlio_results/module-ablation-room10.png`.

The recommended combination is available as one switch:

```bash
python reproduction/map_segmentation/run_3d_instance_pose_eval.py \
  --base /path/to/rayinst-all --out /tmp/robust-map \
  --rooms 10 --robust-fusion
```

## Online ground-plane estimation (2026-10-19)

`run_3d_instance_pose_eval.py` now has `--online-ground-plane`. It gathers
depth points whose per-view frontend label is floor, fits an up-facing plane by
RANSAC, refits it with SVD, and applies the floor prior using signed distance
to that plane. It does not read `floor_y` while estimating the plane. The
diagnostic output records candidate count, inlier ratio, normal, median
residual, and P95 residual.

Room 10 produced a stable plane with normal y=0.9997, 16,579 inliers, and P95
residual 2.6 cm. Varying the signed-distance tolerance from 3 to 30 cm gave
61.36%, 62.49%, 62.80%, 63.10%, **63.52%**, 63.49%, and 63.05%; 15 cm was the
best tested value. The floor prior must preserve small-object labels or it can
raise floor recall while reducing small-object recall to zero.

Room 9 fitted a near-horizontal plane but its frontend floor evidence was
below the automatic safety ratio, so the prior was disabled and the score
remained 54.53%. The tracked Room 17 raycast pose export in the current
checkout has zero overlap with its valid GT grid; the previous transient
aligned evaluation directory is unavailable, so no Room 17 online-plane
accuracy is reported. Results are in
`fastlio_results/online-ground-plane-summary.json` and
`fastlio_results/online-ground-plane-ablation.png`.

## Stair scene with added small objects (Room 11)

The local HM3D semantic inventory identifies Room 11 as containing class 15
(`stairs`). I prepared that room with the same nine banana/chef-can/cheezit
assets and eight 30 cm forward-facing views, then rebuilt the saved OneMap and
frontend feature cache.

The single-plane baseline reaches 66.87% full-map accuracy. The multi-plane
RANSAC model finds six horizontal candidates, including planes around 1.199 m,
1.256 m, 0.917 m, and 0.578 m; the dominant plane has 30,180 inliers and 2.9
mm P95 residual. Multi-plane fusion raises accuracy to **69.11%** and observed
region accuracy to 79.46%. Preserving small-object evidence gives 65.51%
overall but raises small-object recall from 0% to **73.68%**.

The fine stair-surface diagnostic contains 407 GT cells. Correct stair cells
increase from 2/407 with one plane to 28/407 with multiple planes. This is a
partial improvement: many stair cells are not covered or are not classified as
stairs by the RGB semantic frontend. The three-class scorer maps stairs into
`floor_stairs`, so the stair-specific count is reported separately. See
`fastlio_results/stair-room11-summary.json` and
`fastlio_results/stair-room11-ablation.png`.
