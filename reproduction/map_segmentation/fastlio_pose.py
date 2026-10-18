"""FAST-LIO pose adapter for the Habitat/OneMap coordinate convention.

FAST-LIO publishes a LiDAR or body pose in its own world frame. OneMap needs a
camera-to-episodic transform. This module applies an optional world alignment,
the body-to-camera extrinsic, and the existing Habitat conversion in
``scene.pose_matrix``. It deliberately does not claim to run FAST-LIO itself;
the ROS node or rosbag replay produces the pose JSON consumed here.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from scene import pose_matrix


def _matrix(value):
    m = np.asarray(value, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"expected a 4x4 transform, got {m.shape}")
    return m


def _pose_matrix(position, quaternion):
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rotation.from_quat(np.asarray(quaternion, dtype=np.float64)).as_matrix()
    out[:3, 3] = np.asarray(position, dtype=np.float64)
    return out


def load_fastlio_poses(path, design):
    """Load FAST-LIO body poses and return OneMap-compatible pose dictionaries.

    JSON format::

        {"poses": [{"index": 0, "position": [x,y,z],
                    "rotation_xyzw": [qx,qy,qz,qw]}],
         "world_alignment": [[...], ...]}

    ``position`` and quaternion are body poses in the FAST-LIO world frame.
    ``world_alignment`` maps that world frame into Habitat world coordinates.
    If omitted, identity is used for simulation tests. The camera extrinsic is
    taken from ``design.camera`` (height and forward offset).
    """
    raw = json.loads(Path(path).read_text())
    records = raw.get("poses", raw if isinstance(raw, list) else None)
    if records is None:
        raise ValueError("FAST-LIO pose file must contain a 'poses' list")
    alignment = _matrix(raw.get("world_alignment", np.eye(4).tolist())) if isinstance(raw, dict) else np.eye(4)
    by_index = {}
    camera = design.get("camera", {})
    forward = float(camera.get("forward_offset_m", 0.0))
    height = float(camera.get("height_m", 0.0))
    for item in records:
        idx = int(item["index"])
        body = alignment @ _pose_matrix(item["position"], item["rotation_xyzw"])
        # Camera is mounted forward along the body optical forward (-Z).
        camera_offset = body[:3, :3] @ np.array([0.0, height, -forward])
        camera_world = body[:3, 3] + camera_offset
        q = Rotation.from_matrix(body[:3, :3]).as_quat()
        pose = {"position": body[:3, 3].tolist(), "rotation_xyzw": q.tolist(),
                "camera_world_position": camera_world.tolist(), "index": idx,
                "split": "mapping", "source": "FAST-LIO"}
        by_index[idx] = pose
    expected = [p["index"] for p in design["poses"] if p.get("split") == "mapping"]
    missing = sorted(set(expected) - set(by_index))
    if missing:
        raise ValueError(f"FAST-LIO pose file is missing mapping view indices: {missing}")
    return [by_index[i] for i in expected]


def onemap_matrices(path, design):
    """Return ``(index, T_camera_to_episodic)`` for diagnostics and mapping."""
    poses = load_fastlio_poses(path, design)
    origin = design["origin"]
    forward = float(design.get("camera", {}).get("forward_offset_m", 0.0))
    return [(p["index"], pose_matrix(p, origin, forward)) for p in poses]
