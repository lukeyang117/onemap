"""Run upstream OneMap fusion, then serialize the final map losslessly."""
import argparse
import gc
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np
import torch
import yaml

from scene import ROOT, CLASSES, PROMPTS, digest, dump, pose_matrix
from fastlio_pose import load_fastlio_poses
sys.path.insert(0, str(ROOT))
from mapping.feature_map import OneMap
from vision_models.clip_dense import ClipModel


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "reproduction/map_segmentation/experiment")
    parser.add_argument("--cases", nargs="+", choices=["baseline", "augmented"], default=["baseline", "augmented"])
    parser.add_argument("--pose-file", type=Path, default=None,
                        help="FAST-LIO JSON body poses; omit to use Habitat design poses")
    args = parser.parse_args()
    output = args.experiment.resolve()
    design = json.loads((output / "design.json").read_text())
    with (ROOT / "config/mon/mapping_conf_sim.yaml").open() as f:
        settings = yaml.safe_load(f)["MappingConf"]
    settings.update(design.get("mapping_calibration_overrides", {}))
    model = ClipModel(str(ROOT / "weights/clip.pth"), jetson=False).eval()
    text = model.get_text_features(PROMPTS).cpu()
    config = SimpleNamespace(**settings)
    poses = [p for p in design["poses"] if p["split"] == "mapping"]
    pose_source = "habitat_ground_truth"
    if args.pose_file is not None:
        poses = load_fastlio_poses(args.pose_file.resolve(), design)
        pose_source = "fastlio_body_pose_plus_camera_extrinsic"
    for name in args.cases:
        directory = output / name
        if (directory / "saved-map.pt").exists():
            raise FileExistsError(directory / "saved-map.pt")
        mapper = OneMap(768, config, map_device="cpu")
        mapper.set_camera_matrix(np.array([[320, 0, 320], [0, 320, 320], [0, 0, 1]]))
        started = time.monotonic()
        for index, pose in enumerate(poses):
            from PIL import Image
            rgb = np.asarray(Image.open(directory / f"view-{pose['index']:02d}-rgb.png"))
            depth = np.load(directory / f"view-{pose['index']:02d}-geometry.npz")["depth"].copy()
            with torch.inference_mode():
                features = model.get_image_features(rgb.transpose(2, 0, 1))[0]
                mapper.update(features, depth, pose_matrix(pose, design["origin"],
                    design.get("camera", {}).get("forward_offset_m", 0.0)))
            print(f"{name} upstream OneMap.update {index + 1}/{len(poses)}; elapsed={time.monotonic() - started:.1f}s", flush=True)
            del features
        active = mapper.confidence_map > 0
        assert active.any()
        assert torch.isfinite(mapper.feature_map[active]).all()
        assert not torch.count_nonzero(mapper.feature_map[~active])
        indices = active.nonzero().to(torch.int32)
        state = {"format": "OneMap observed float32 cells, lossless sparse serialization v1",
                 "features": mapper.feature_map[active].clone(), "indices": indices,
                 "confidence": mapper.confidence_map[active].clone(),
                 "obstacle_map": mapper.obstacle_map.clone(),
                 "checked_conf_map": mapper.checked_conf_map.clone(),
                 "navigable_map": torch.as_tensor(mapper.navigable_map),
                 "n_cells": mapper.n_cells, "size_m": mapper.size,
                 "cell_size_m": mapper.cell_size, "origin_habitat": design["origin"],
                 "mapping_config": settings, "classes": CLASSES, "prompts": PROMPTS,
                 "camera": design.get("camera", {"height_m": 0.88, "forward_offset_m": 0.0}),
                 "text_features": text, "mapping_pose_indices": [p["index"] for p in poses],
                 "feature_dim": 768, "image_weight_sha256": digest(ROOT / "weights/clip.pth"),
                 "backend": "unmodified mapping.feature_map.OneMap, CPU map and GPU projection",
                 "pose_source": pose_source,
                 "pose_file": str(args.pose_file.resolve()) if args.pose_file else None}
        path = directory / "saved-map.pt"
        torch.save(state, path)
        # Verify serialization now, then destroy the live map. All accuracy is
        # computed by score_saved.py in another process without the encoder.
        restored = torch.load(path, map_location="cpu", weights_only=True)
        for key in ("features", "indices", "confidence", "text_features", "obstacle_map"):
            assert torch.equal(state[key], restored[key]), key
        report = {"map_updates": len(poses), "observed_cells": int(active.sum()),
                  "cell_size_m": mapper.cell_size, "saved_bytes": path.stat().st_size,
                  "saved_map_sha256": digest(path), "reload_bitwise_equal": True,
                  "SAM_calls": 0, "detector_calls": 0, "pose_source": pose_source,
                  "elapsed_seconds": time.monotonic() - started}
        dump(directory / "build-verification.json", report)
        print(f"Saved/reloaded {name}: {report}", flush=True)
        del mapper, state, restored, active, indices
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
