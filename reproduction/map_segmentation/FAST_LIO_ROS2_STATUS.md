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
odometry messages without a crash. The estimated trajectory was rigidly
aligned to Habitat with a Kabsch fit and the eight mapping views were then
replayed through OneMap.

Room 10, observed-only scoring:

| map pose source | semantic accuracy | floor/stairs | small objects | large objects | height MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Habitat ground truth | 61.25% | baseline | 0/22 | baseline | 111.18 cm |
| FAST-LIO CPU estimate | **0.28%** | 0/2139 | 0/22 | 10/1449 | 106.07 cm |

This is a measured FAST-LIO result: 10 of 3610 observed GT cells agree and
3549 are unmapped. The comparison image is
`/tmp/fastlio-room10-eval/semantic_elevation/semantic-elevation-comparison.png`.
The node now runs end-to-end, but the depth-derived observation model still
does not preserve the eight camera view orientations well enough for OneMap
fusion. The Habitat-pose map remains the control condition.

Example:

```bash
source /opt/ros/humble/setup.bash
source /tmp/fastlio_build_ws/install/setup.bash
python reproduction/map_segmentation/write_fastlio_ros2_bag.py \
  --input /tmp/habitat-lio-room10 --output /tmp/habitat-lio-room10-ros2
ros2 bag play /tmp/habitat-lio-room10-ros2
```
