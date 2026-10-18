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
IMU messages. The replay does not yet yield an evaluation trajectory: the
depth-derived point clouds are not a physical spinning LiDAR (many returns are
correlated camera pixels and the simulated motion has discontinuities), and
the unmodified FAST-LIO optimizer diverges and exits with a segmentation fault
after roughly 20–40 frames. Therefore no FAST-LIO pose is used in the OneMap
accuracy table. The earlier pose-file result remains a ground-truth replay,
not an estimated-SLAM result.

Example:

```bash
source /opt/ros/humble/setup.bash
source /tmp/fastlio_build_ws/install/setup.bash
python reproduction/map_segmentation/write_fastlio_ros2_bag.py \
  --input /tmp/habitat-lio-room10 --output /tmp/habitat-lio-room10-ros2
ros2 bag play /tmp/habitat-lio-room10-ros2
```
