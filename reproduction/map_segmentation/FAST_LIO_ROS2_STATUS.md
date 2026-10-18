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

Example:

```bash
source /opt/ros/humble/setup.bash
source /tmp/fastlio_build_ws/install/setup.bash
python reproduction/map_segmentation/write_fastlio_ros2_bag.py \
  --input /tmp/habitat-lio-room10 --output /tmp/habitat-lio-room10-ros2
ros2 bag play /tmp/habitat-lio-room10-ros2
```
