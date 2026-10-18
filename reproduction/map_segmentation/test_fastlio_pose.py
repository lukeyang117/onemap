import json
import numpy as np
from scipy.spatial.transform import Rotation
from fastlio_pose import load_fastlio_poses


def test_fastlio_to_camera_extrinsic(tmp_path):
    design = {"origin": [0, 0, 0], "camera": {"height_m": .3, "forward_offset_m": .3},
              "poses": [{"index": 0, "split": "mapping"}]}
    data = {"world_alignment": np.eye(4).tolist(), "poses": [{
        "index": 0, "position": [1, 2, 3], "rotation_xyzw": [0, 0, 0, 1]}]}
    path = tmp_path / "poses.json"; path.write_text(json.dumps(data))
    pose = load_fastlio_poses(path, design)[0]
    np.testing.assert_allclose(pose["position"], [1, 2, 3])
    np.testing.assert_allclose(pose["camera_world_position"], [1, 2.3, 2.7])


def test_fastlio_alignment_and_rotation(tmp_path):
    design = {"origin": [0, 0, 0], "camera": {"height_m": 0, "forward_offset_m": 0},
              "poses": [{"index": 4, "split": "mapping"}]}
    alignment = np.eye(4); alignment[:3, 3] = [10, 20, 30]
    q = Rotation.from_euler("z", 90, degrees=True).as_quat().tolist()
    path = tmp_path / "poses.json"; path.write_text(json.dumps({"world_alignment": alignment.tolist(), "poses": [{"index": 4, "position": [1, 2, 3], "rotation_xyzw": q}]}))
    pose = load_fastlio_poses(path, design)[0]
    np.testing.assert_allclose(pose["position"], [11, 22, 33])
