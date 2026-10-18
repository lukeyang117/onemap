"""Write the Habitat LiDAR/IMU export as a ROS 2 SQLite bag.

The bag is intentionally made with ``rosbags`` so the exporter does not need a
running ROS node.  FAST-LIO ROS2 can consume the resulting CDR messages with
``ros2 bag play``.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore


def ros_time(t, Time):
    sec = int(t)
    return Time(sec, int(round((t - sec) * 1e9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--scan-lines", type=int, default=32)
    args = ap.parse_args()
    ts = get_typestore(Stores.ROS2_HUMBLE)
    T = ts.types
    Header = T["std_msgs/msg/Header"]
    Time = T["builtin_interfaces/msg/Time"]
    PF = T["sensor_msgs/msg/PointField"]
    PC = T["sensor_msgs/msg/PointCloud2"]
    Imu = T["sensor_msgs/msg/Imu"]
    Vec = T["geometry_msgs/msg/Vector3"]
    Quat = T["geometry_msgs/msg/Quaternion"]
    files = sorted((args.input / "lidar").glob("*.npz"))
    imu = np.load(args.input / "imu.npz")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Writer(args.output, version=8) as writer:
        lc = writer.add_connection("/fastlio/lidar", "sensor_msgs/msg/PointCloud2", typestore=ts)
        ic = writer.add_connection("/fastlio/imu", "sensor_msgs/msg/Imu", typestore=ts)
        for k, f in enumerate(files):
            d = np.load(f)
            t = float(d["timestamp"])
            pts = np.asarray(d["points_body"], np.float32)
            n = len(pts)
            # Keep time zero so FAST-LIO derives per-ring timing from yaw; the
            # camera raster is arranged as 32 contiguous scan lines.
            rel = np.zeros(n, dtype=np.float32)
            # PCL_ADD_POINT4D includes a 4-byte alignment slot, so the
            # registered Velodyne struct is x/y/z at 0/4/8, intensity at 16,
            # time at 20, ring at 24 and 32 bytes per point.
            raw = np.zeros((n, 32), dtype=np.uint8)
            raw[:, 0:4] = np.ascontiguousarray(pts[:, 0]).view(np.uint8).reshape(n, 4)
            raw[:, 4:8] = np.ascontiguousarray(pts[:, 1]).view(np.uint8).reshape(n, 4)
            raw[:, 8:12] = np.ascontiguousarray(pts[:, 2]).view(np.uint8).reshape(n, 4)
            raw[:, 16:20] = np.ones(n, np.float32).view(np.uint8).reshape(n, 4)
            raw[:, 20:24] = rel.view(np.uint8).reshape(n, 4)
            raw[:, 24:26] = ((np.arange(n, dtype=np.uint16) // (n // args.scan_lines)) % args.scan_lines).view(np.uint8).reshape(n, 2)
            msg = PC(
                Header(ros_time(t, Time), "lidar"), 1, n,
                [PF("x", 0, 7, 1), PF("y", 4, 7, 1), PF("z", 8, 7, 1),
                 PF("intensity", 16, 7, 1), PF("time", 20, 7, 1), PF("ring", 24, 4, 1)],
                False, 32, 32 * n, raw.reshape(-1), True,
            )
            writer.write(lc, int(round(t * 1e9)), ts.serialize_cdr(msg, msg.__msgtype__))
            a = np.asarray(imu["linear_acceleration"][k])
            w = np.asarray(imu["angular_velocity"][k])
            im = Imu(Header(ros_time(t, Time), "imu"), Quat(0., 0., 0., 1.), np.zeros(9), Vec(*w), np.zeros(9), Vec(*a), np.zeros(9))
            writer.write(ic, int(round(t * 1e9)), ts.serialize_cdr(im, im.__msgtype__))
    manifest = {
        "input": str(args.input), "output": str(args.output), "frames": len(files),
        "ros_version": "ros2_humble", "storage": "sqlite3", "serialization": "cdr",
        "topics": {"lidar": "/fastlio/lidar", "imu": "/fastlio/imu"},
        "point_fields": ["x", "y", "z", "intensity", "time", "ring"], "scan_lines": args.scan_lines, "timestamp_unit": "second",
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
