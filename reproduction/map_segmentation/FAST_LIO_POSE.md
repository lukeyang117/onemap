# FAST-LIO pose input for OneMap

The Habitat experiments currently use exact simulator poses. This adapter lets a FAST-LIO/FAST-LIO2 ROS node replace those poses without changing the OneMap RGB-D or semantic fusion code.

## Data flow

```text
LiDAR + IMU → FAST-LIO/FAST-LIO2 → body pose in LIO world
                                      ↓
                         world alignment + body-camera extrinsic
                                      ↓
                         camera pose passed to OneMap.update
```

FAST-LIO is a tightly coupled LiDAR-inertial odometry system. FAST-LIO2 supports direct registration of raw LiDAR points and incremental ikd-Tree mapping. See the [official repository](https://github.com/hku-mars/FAST_LIO) and [FAST-LIO paper](https://arxiv.org/abs/2010.08196).

## Pose file

Export one record per mapping RGB-D frame:

```json
{
  "world_alignment": [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]],
  "poses": [
    {"index": 0, "position": [0,0,0], "rotation_xyzw": [0,0,0,1]}
  ]
}
```

`position` and `rotation_xyzw` are the FAST-LIO body pose. `world_alignment` maps the FAST-LIO world frame into Habitat/world coordinates. The adapter applies `design.json` camera height and forward offset, then converts to the OneMap camera-to-episodic convention.

## Run

```bash
python reproduction/map_segmentation/build_maps.py \
  --experiment reproduction/map_segmentation/multi_room/room-10 \
  --cases augmented \
  --pose-file /path/to/fastlio-room-10.json
```

Omitting `--pose-file` retains the exact Habitat-pose baseline. The serialized map records `pose_source` and `pose_file` so later scores cannot confuse ground-truth pose and estimated pose.

## Important limitation

The current container has Habitat RGB-D data but no real LiDAR/IMU rosbag and no ROS FAST-LIO runtime. Therefore this change adds the tested pose adapter and OneMap integration point; it does not fabricate FAST-LIO output. A real FAST-LIO comparison requires synchronized LiDAR, IMU, camera timestamps, LiDAR-camera extrinsics, and a world-frame alignment. FAST-LIO pose error can reduce semantic-map consistency in simulation because the existing Habitat pose is exact; the purpose of this mode is to measure that localization effect and to support real-robot data.
