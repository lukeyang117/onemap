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

Example:

```bash
source /opt/ros/humble/setup.bash
source /tmp/fastlio_build_ws/install/setup.bash
python reproduction/map_segmentation/write_fastlio_ros2_bag.py \
  --input /tmp/habitat-lio-room10 --output /tmp/habitat-lio-room10-ros2
ros2 bag play /tmp/habitat-lio-room10-ros2
```
