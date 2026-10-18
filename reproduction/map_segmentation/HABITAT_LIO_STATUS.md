# Habitat LiDAR/IMU and FAST-LIO status

The exporter `generate_habitat_lio_dataset.py` creates a continuous trajectory through the same eight room poses, derives sparse LiDAR points from Habitat depth, and derives synchronized IMU acceleration/angular velocity from simulator ground truth. It writes `lidar/*.npz`, `imu.npz`, `ground-truth-poses.json`, and `manifest.json`.

`write_fastlio_rosbag.py` converts those files into a ROS1 bag without requiring ROS to write it:

```bash
python reproduction/map_segmentation/write_fastlio_rosbag.py \
  --input /tmp/habitat-lio-room10 \
  --output /tmp/habitat-lio-room10.bag
```

The bag contains `/fastlio/lidar` as `sensor_msgs/PointCloud2` with `x/y/z/time` float fields and `/fastlio/imu` as `sensor_msgs/Imu`. The point-time field is a synthetic within-scan time ramp; configure the FAST-LIO sensor model and extrinsics accordingly.

Example:

```bash
PYTHONPATH=reproduction/map_segmentation \
python reproduction/map_segmentation/generate_habitat_lio_dataset.py \
  --room 10 --output /tmp/habitat-lio-room10
```

This environment cannot execute FAST-LIO itself: the Habitat package has no native LiDAR/IMU sensor types, and the host has no ROS/catkin, PCL/Eigen, or Docker FAST-LIO runtime. The official FAST-LIO package requires ROS, PCL/Eigen, synchronized LiDAR/IMU timestamps, and LiDAR-IMU extrinsics. The generated files and bags are therefore a sensor-export stage, not an estimated SLAM trajectory. No FAST-LIO accuracy number is reported until its ROS node or a compatible standalone port consumes these messages.
