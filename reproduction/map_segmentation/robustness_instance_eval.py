"""Randomized robustness evaluation for the complete FAST-LIO + OneMap chain.

The mapper is rerun from perturbed FAST-LIO poses and corrupted RGB-D inputs.
Ground truth is used only by the final scorer; it is never used to construct
or repair a trial pose file.
"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def perturb_pose(src, dst, rng, trans_sigma, rot_sigma_deg, drop_views=0, drop_indices=None):
    raw = json.loads(Path(src).read_text())
    records = list(raw["poses"])
    if drop_views:
        ids = np.asarray(drop_indices if drop_indices is not None else rng.choice(len(records), size=min(drop_views, max(0, len(records)-1)), replace=False), dtype=int)
        records = [x for i, x in enumerate(records) if i not in set(ids)]
    out = []
    for p in records:
        q = dict(p)
        q["position"] = (np.asarray(p["position"], dtype=float) + rng.normal(0, trans_sigma, 3)).tolist()
        dr = Rotation.from_rotvec(rng.normal(0, np.deg2rad(rot_sigma_deg), 3))
        q["rotation_xyzw"] = (dr * Rotation.from_quat(p["rotation_xyzw"])).as_quat().tolist()
        out.append(q)
    raw["poses"] = out
    if drop_views:
        raw["allow_missing_views"] = True
    raw["source"] = "FAST-LIO pose with randomized robustness perturbation"
    Path(dst).write_text(json.dumps(raw, indent=2))


def corrupt_rgbd(room_dir, rng, depth_dropout=0.0, depth_noise_cm=0.0, rgb_jitter=0.0):
    aug = Path(room_dir) / "augmented"
    for geom_path in sorted(aug.glob("view-*-geometry.npz")):
        z = np.load(geom_path)
        depth = np.asarray(z["depth"], dtype=np.float32).copy()
        instance = np.asarray(z["instance"])
        good = np.isfinite(depth) & (depth > 0)
        if depth_noise_cm:
            noise = rng.normal(0, depth_noise_cm / 100.0, depth.shape).astype(np.float32)
            depth[good] = np.maximum(0.05, depth[good] + noise[good])
        if depth_dropout:
            drop = (rng.random(depth.shape) < depth_dropout) & good
            depth[drop] = 0.0
        np.savez_compressed(geom_path, depth=depth, instance=instance)
    if rgb_jitter:
        for rgb_path in sorted(aug.glob("view-*-rgb.png")):
            im = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32)
            # Multiplicative brightness plus small additive sensor noise.
            gain = 1.0 + rng.normal(0, rgb_jitter)
            im = np.clip(im * gain + rng.normal(0, 3.0 * rgb_jitter, im.shape), 0, 255).astype(np.uint8)
            Image.fromarray(im).save(rgb_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="parent containing room-XX directories")
    ap.add_argument("--pose-file", type=Path, required=True)
    ap.add_argument("--room", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--conditions", nargs="+", default=None, help="optional subset of condition names")
    ap.add_argument("--drop-index", type=int, default=None, help="deterministic mapping view index to remove")
    ap.add_argument("--adaptive-consistency", action="store_true", help="enable mapper's self-consistency view gate")
    args = ap.parse_args()
    rng = np.random.default_rng(20261018 + args.room)
    # The four conditions separate pose sensitivity, view coverage, and RGB-D
    # sensor corruption instead of hiding all effects in one number.
    levels = [
        {"name": "clean", "trans_sigma": 0.0, "rot_sigma": 0.0, "drop_views": 0, "depth_dropout": 0.0, "depth_noise_cm": 0.0, "rgb_jitter": 0.0},
        {"name": "pose-jitter", "trans_sigma": 0.05, "rot_sigma": 2.0, "drop_views": 0, "depth_dropout": 0.0, "depth_noise_cm": 0.0, "rgb_jitter": 0.0},
        {"name": "one-view-drop", "trans_sigma": 0.02, "rot_sigma": 1.0, "drop_views": 1, "depth_dropout": 0.0, "depth_noise_cm": 0.0, "rgb_jitter": 0.0},
        {"name": "depth-corrupt", "trans_sigma": 0.0, "rot_sigma": 0.0, "drop_views": 0, "depth_dropout": 0.10, "depth_noise_cm": 1.0, "rgb_jitter": 0.0},
        {"name": "rgbd-corrupt", "trans_sigma": 0.02, "rot_sigma": 1.0, "drop_views": 0, "depth_dropout": 0.10, "depth_noise_cm": 1.0, "rgb_jitter": 0.08},
    ]
    if args.conditions is not None:
        levels = [x for x in levels if x["name"] in set(args.conditions)]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg in levels:
        for trial_id in range(args.trials):
            name = f"room-{args.room:02d}-{cfg['name']}-{trial_id}"
            trial = args.out / name
            if trial.exists():
                shutil.rmtree(trial)
            trial.mkdir(parents=True)
            room_dst = trial / f"room-{args.room:02d}"
            shutil.copytree(args.base / f"room-{args.room:02d}", room_dst)
            trial_rng = np.random.default_rng(20261018 + args.room * 1000 + trial_id + 17 * list(x["name"] for x in levels).index(cfg["name"]))
            drop_indices = [args.drop_index] if args.drop_index is not None and cfg["drop_views"] else None
            perturb_pose(args.pose_file, room_dst / "fastlio-poses.json", trial_rng, cfg["trans_sigma"], cfg["rot_sigma"], cfg["drop_views"], drop_indices)
            if cfg["depth_dropout"] or cfg["depth_noise_cm"] or cfg["rgb_jitter"]:
                corrupt_rgbd(room_dst, trial_rng, cfg["depth_dropout"], cfg["depth_noise_cm"], cfg["rgb_jitter"])
            result_dir = trial / "result"
            cmd = [sys.executable, "reproduction/map_segmentation/run_3d_instance_pose_eval.py", "--base", str(trial), "--out", str(result_dir), "--rooms", str(args.room)]
            if args.adaptive_consistency:
                cmd.append("--adaptive-consistency")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            report = json.loads((result_dir / f"room-{args.room:02d}.json").read_text())
            small = next(x for x in report["classes"] if x["class"] == "small")
            rows.append({"room": args.room, "condition": cfg["name"], "trial": trial_id, "accuracy_percent": report["accuracy"], "observed_accuracy_percent": report["observed_accuracy"], "coverage_percent": report["coverage_percent"], "height_MAE_cm": report["height_MAE_cm"], "small_recall_percent": small["recall"], "small_precision_percent": small["precision"], "large_recall_percent": next(x for x in report["classes"] if x["class"] == "large")["recall"]})
    out_file = args.out / f"room-{args.room:02d}.json"
    out_file.write_text(json.dumps(rows, indent=2) + "\n")
    for cfg in levels:
        a = [x for x in rows if x["condition"] == cfg["name"]]
        vals = np.asarray([x["accuracy_percent"] for x in a], float)
        cov = np.asarray([x["coverage_percent"] for x in a], float)
        print(args.room, cfg["name"], "mean", float(vals.mean()), "std", float(vals.std()), "worst", float(vals.min()), "coverage", float(cov.mean()))


if __name__ == "__main__":
    main()
