# FAST-LIO pose replay record

Date: 2026-10-18

The adapter was exercised end to end on Room 10 using a FAST-LIO-format JSON trajectory generated from the eight Habitat body poses. This verifies the file parser, world alignment, body-to-camera extrinsic, OneMap update, serialization, and elevation scoring. It is an interface replay, not a FAST-LIO accuracy result: no LiDAR/IMU rosbag or ROS FAST-LIO node is available in this container.

Command:

```bash
PYTHONPATH=reproduction/map_segmentation \
python reproduction/map_segmentation/build_maps.py \
  --experiment /tmp/onemap-fastlio-room10 \
  --cases augmented \
  --pose-file /tmp/onemap-fastlio-room10/fastlio-poses.json

PYTHONPATH=reproduction/map_segmentation \
python reproduction/map_segmentation/score_elevation_map.py \
  --experiment /tmp/onemap-fastlio-room10 --observed-only
```

Observed output:

| check | result |
|---|---:|
| mapping updates | 8 |
| observed map cells | 5248 |
| serialization reload | exact |
| semantic agreement, Room 10 | 61.2465% |
| small-object recall | 0/22 = 0% |
| height MAE | 111.1757 cm |

The semantic and elevation scores match the Habitat-pose Room 10 baseline. The serialized replay map hash is `d41ca25a6c7d07f5db7f9942c8fa7bf87efa68a5860b9212339a2ed23e04aeb4`; the hash is expected to differ from the older baseline artifact because the new map records `pose_source` metadata and uses the adapter path. Tensor values agree within the normal floating-point replay tolerance (feature maximum absolute difference `5.36e-7`).
