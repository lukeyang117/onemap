"""Evaluate a Grounding-DINO + SAM2 small-object mapper on four rooms.

The existing OneMap map supplies terrain/large-object evidence.  Grounding
DINO proposes small objects, SAM2 refines each box, and RGB-D projection plus
multi-view association supplies the 3-D consensus before small labels are
written.  GT is read only by the scoring stage.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from grounded_small_objects import (
    MODEL_REVISIONS,
    GroundingDinoDetector,
    Sam2BoxSegmenter,
    associate_observations,
    compact_obstacle_geometry,
    coarse_label,
    estimate_local_support_height,
    small_label_from_phrase,
    project_mask_to_world,
    rasterize_small_evidence,
)
from scene import ROOT, dump
from score_saved import grid_indices
from score_voxel_map import depth_points


SMALL_PROMPTS = ["banana", "tin can", "cracker box"]
CLASS_NAMES = {1: "terrain", 2: "small", 3: "large"}


def three_class_score(gt: np.ndarray, prediction: np.ndarray, valid: np.ndarray) -> dict:
    valid = np.asarray(valid, dtype=bool) & np.isin(gt, [1, 2, 3])
    cm = np.zeros((4, 4), dtype=np.int64)
    np.add.at(cm, (gt[valid], prediction[valid]), 1)
    rows = []
    for class_id, name in CLASS_NAMES.items():
        gt_count = int(cm[class_id].sum())
        predicted = int(cm[:, class_id].sum())
        tp = int(cm[class_id, class_id])
        recall = 100.0 * tp / gt_count if gt_count else None
        precision = 100.0 * tp / predicted if predicted else None
        f1 = None if recall is None or precision is None or recall + precision == 0 else 2 * recall * precision / (recall + precision)
        rows.append({"class": name, "gt": gt_count, "correct": tp, "predicted": predicted,
                     "fp": predicted - tp, "fn": gt_count - tp, "precision": precision,
                     "recall": recall, "f1": f1})
    total = int(valid.sum())
    covered = int(((prediction > 0) & valid).sum())
    return {
        "accuracy": 100.0 * float(np.trace(cm)) / total if total else 0.0,
        "correct": int(np.trace(cm)),
        "GT": total,
        "classes": rows,
        "cm": cm.tolist(),
        "coverage_percent": 100.0 * covered / total if total else 0.0,
    }


def _map_grid(state: dict) -> np.ndarray:
    grid = np.zeros((int(state["n_cells"]), int(state["n_cells"])), dtype=np.uint8)
    fine = (state["features"] @ state["text_features"].T).argmax(1).numpy()
    ids = state["indices"].numpy()
    coarse = np.asarray([coarse_label(item) for item in fine], dtype=np.uint8)
    grid[ids[:, 0], ids[:, 1]] = coarse
    return grid


def _base_elevation_prediction(room_dir: Path, poses: list[dict], state: dict) -> np.ndarray:
    """Rebuild the baseline semantic elevation labels from saved OneMap cells.

    This mirrors score_elevation_map.py: every observed depth point queries the
    losslessly saved OneMap feature at its XY cell, then semantic priority is
    reduced over all views. No semantic GT is read here.
    """
    grid = _map_grid(state)
    n = int(state["n_cells"])
    prediction = np.zeros((n, n), dtype=np.uint8)
    for pose in poses:
        view = int(pose["index"])
        depth = np.load(room_dir / "augmented" / f"view-{view:02d}-geometry.npz")["depth"].astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        if not valid.any():
            continue
        points = depth_points(depth, pose)[valid]
        rows, cols = grid_indices(points[:, 0], points[:, 2], state)
        inside = (rows >= 0) & (rows < n) & (cols >= 0) & (cols < n)
        rows, cols = rows[inside], cols[inside]
        labels = grid[rows, cols]
        np.maximum.at(prediction, (rows, cols), labels)
    return prediction


def _view_overlay(rgb: np.ndarray, detections: list[dict], masks: np.ndarray, path: Path) -> None:
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image, "RGBA")
    for index, item in enumerate(detections):
        box = np.asarray(item["box"], dtype=float)
        color = (40, 220, 80, 180) if item.get("accepted") else (220, 80, 50, 150)
        draw.rectangle(tuple(float(v) for v in box), outline=color, width=2)
        draw.text((float(box[0]), max(0.0, float(box[1]) - 14)),
                  f"{item.get('phrase', '')[:18]} {item.get('score', 0):.2f}", fill=color)
        if index < len(masks):
            mask = masks[index].astype(bool)
            overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            overlay[mask] = [40, 220, 80, 55] if item.get("accepted") else [220, 80, 50, 35]
            image = Image.alpha_composite(image.convert("RGBA"), Image.fromarray(overlay, "RGBA"))
            draw = ImageDraw.Draw(image, "RGBA")
    image.convert("RGB").save(path)


def _cache_one_view(room_dir: Path, cache_dir: Path, pose: dict, detector, segmenter) -> dict:
    view = int(pose["index"])
    rgb = np.asarray(Image.open(room_dir / "augmented" / f"view-{view:02d}-rgb.png"))[:, :, :3]
    detections = detector.detect(rgb, SMALL_PROMPTS)
    masks = segmenter.segment(rgb, [item["box"] for item in detections]) if detections else []
    metadata = []
    mask_array = np.zeros((len(masks), *rgb.shape[:2]), dtype=np.uint8)
    for index, (item, mask_record) in enumerate(zip(detections, masks)):
        mask = np.asarray(mask_record["mask"], dtype=bool)
        mask_array[index] = mask
        label = small_label_from_phrase(item["phrase"])
        record = {
            "box": np.asarray(item["box"], dtype=float).tolist(),
            "score": float(item["score"]),
            "phrase": str(item["phrase"]),
            "mask_score": float(mask_record["mask_score"]),
            "label": label,
            "accepted": False,
            "filter_reason": "not_yet_checked_against_predicted_local_support",
        }
        metadata.append(record)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / f"view-{view:02d}.npz", masks=mask_array)
    dump(cache_dir / f"view-{view:02d}.json", {
        "view": view, "model": {"detector": detector.model_id, "detector_revision": detector.revision,
                                  "sam2": segmenter.model_id, "sam2_revision": segmenter.revision},
        "detections": metadata,
    })
    return {"metadata": metadata, "masks": mask_array,
            "model": {"detector": detector.model_id, "detector_revision": detector.revision,
                      "sam2": segmenter.model_id, "sam2_revision": segmenter.revision}}


def _load_or_cache(room_dir: Path, room_out: Path, poses: list[dict], detector, segmenter, force: bool) -> list[dict]:
    cache_dir = room_out / "detections"
    outputs = []
    for pose in poses:
        view = int(pose["index"])
        json_path = cache_dir / f"view-{view:02d}.json"
        npz_path = cache_dir / f"view-{view:02d}.npz"
        if force or not json_path.exists() or not npz_path.exists():
            outputs.append(_cache_one_view(room_dir, cache_dir, pose, detector, segmenter))
        else:
            saved = json.loads(json_path.read_text())
            outputs.append({"metadata": saved["detections"], "model": saved.get("model", {}),
                            "masks": np.load(npz_path)["masks"]})
    return outputs


def _predicted_terrain_points(room_dir: Path, poses: list[dict], state: dict,
                              base_prediction: np.ndarray) -> np.ndarray:
    """Collect RGB-D points whose saved OneMap cell predicts terrain.

    The terrain map is the current deployable prediction; no GT labels,
    design-file floor height, or mesh geometry enter this support estimate.
    """
    n = int(state["n_cells"])
    points_by_view = []
    for pose in poses:
        view = int(pose["index"])
        depth = np.load(room_dir / "augmented" / f"view-{view:02d}-geometry.npz")["depth"].astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        if not valid.any():
            continue
        points = depth_points(depth, pose)[valid]
        rows, cols = grid_indices(points[:, 0], points[:, 2], state)
        inside = (rows >= 0) & (rows < n) & (cols >= 0) & (cols < n)
        terrain = np.zeros(len(points), dtype=bool)
        terrain[inside] = base_prediction[rows[inside], cols[inside]] == 1
        if terrain.any():
            points_by_view.append(points[terrain])
    return np.concatenate(points_by_view, axis=0) if points_by_view else np.empty((0, 3), dtype=np.float32)


def _observations(room_dir: Path, poses: list[dict], cached: list[dict], state: dict,
                  predicted_terrain: np.ndarray, min_score: float = 0.0) -> list[dict]:
    observations = []
    for pose, data in zip(poses, cached):
        view = int(pose["index"])
        depth = np.load(room_dir / "augmented" / f"view-{view:02d}-geometry.npz")["depth"].astype(np.float32)
        for index, item in enumerate(data["metadata"]):
            label = small_label_from_phrase(item["phrase"])
            if label is None or float(item["score"]) < min_score:
                item["accepted"] = False
                item["filter_reason"] = "unknown_or_low_confidence_phrase"
                continue
            points = project_mask_to_world(data["masks"][index].astype(bool), depth, pose)
            support_height, support_count = estimate_local_support_height(
                points, predicted_terrain, radius=0.30, min_support_points=3
            )
            if support_height is None:
                geometry_ok, reason = False, "no_local_predicted_terrain_support"
            else:
                geometry_ok, reason = compact_obstacle_geometry(points, support_height)
            item["accepted"] = bool(geometry_ok)
            item["filter_reason"] = reason
            item["local_support_height_m"] = support_height
            item["local_support_points"] = int(support_count)
            if not geometry_ok:
                continue
            rows, cols = grid_indices(points[:, 0], points[:, 2], state)
            inside = (rows >= 0) & (rows < int(state["n_cells"])) & (cols >= 0) & (cols < int(state["n_cells"]))
            points = points[inside]
            cells = np.unique(np.stack([rows[inside], cols[inside]], axis=1), axis=0) if inside.any() else np.empty((0, 2), dtype=np.int64)
            if len(points) and len(cells):
                observations.append({"view": view, "points": points, "cells": cells,
                                     "score": float(item["score"]), "mask_score": float(item["mask_score"]),
                                     "label": int(label), "phrase": item["phrase"],
                                     "box": item["box"]})
    return observations


def _prediction_for_variant(base: np.ndarray, observations: list[dict], variant: str, min_support: int) -> tuple[np.ndarray, dict]:
    prediction = base.copy()
    if variant == "per_view":
        accepted = observations
        groups = [[item] for item in observations]
    else:
        groups = associate_observations(observations, voxel_size=0.05, centroid_gate=0.18, surface_gate=0.08)
        accepted = []
        for group in groups:
            if len({int(item["view"]) for item in group}) >= min_support:
                accepted.extend(group)
    cells = [item["cells"] for item in accepted if len(item["cells"])]
    if cells:
        indices = np.unique(np.concatenate(cells, axis=0), axis=0)
        prediction = rasterize_small_evidence(prediction, indices)
    return prediction, {"observations": len(observations), "groups": len(groups), "accepted_observations": len(accepted),
                        "accepted_cells": int(len(np.unique(np.concatenate(cells, axis=0), axis=0))) if cells else 0}


def _map_visual(gt: np.ndarray, prediction: np.ndarray, valid: np.ndarray, path: Path, title: str) -> None:
    palette = np.asarray([[40, 40, 40], [210, 190, 120], [235, 145, 30], [60, 130, 210]], dtype=np.uint8)
    coordinates = np.argwhere(valid)
    if len(coordinates):
        low = np.maximum(coordinates.min(axis=0) - 3, 0)
        high = np.minimum(coordinates.max(axis=0) + 4, np.asarray(gt.shape))
        r0, c0 = (int(value) for value in low)
        r1, c1 = (int(value) for value in high)
    else:
        r0, c0, r1, c1 = 0, 0, gt.shape[0], gt.shape[1]
    gt_crop = gt[r0:r1, c0:c1]
    pred_crop = prediction[r0:r1, c0:c1]
    valid_crop = valid[r0:r1, c0:c1]
    gt_rgb = palette[np.clip(gt_crop, 0, 3)]
    pred_rgb = palette[np.clip(pred_crop, 0, 3)]
    gt_rgb[~valid_crop] = [255, 255, 255]
    pred_rgb[~valid_crop] = [255, 255, 255]
    error = np.zeros((*gt_crop.shape, 3), dtype=np.uint8) + 255
    error[valid_crop & (gt_crop == pred_crop)] = [70, 170, 80]
    error[valid_crop & (gt_crop != pred_crop)] = [210, 70, 60]
    width, height = gt_crop.shape[1], gt_crop.shape[0]
    canvas = Image.new("RGB", (width * 3, height + 58), "white")
    canvas.paste(Image.fromarray(gt_rgb), (0, 58))
    canvas.paste(Image.fromarray(pred_rgb), (width, 58))
    canvas.paste(Image.fromarray(error), (width * 2, 58))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"{title} | GT | Prediction | Error", fill="black")
    for index, (label, color) in enumerate((
        ("terrain", tuple(palette[1])), ("small", tuple(palette[2])), ("large", tuple(palette[3])),
    )):
        x = 8 + index * 120
        draw.rectangle((x, 32, x + 12, 44), fill=color)
        draw.text((x + 18, 30), label, fill="black")
    canvas.save(path)


def run_room(room: int, base: Path, out: Path, detector, segmenter, args) -> dict:
    room_dir = base / f"room-{room:02d}"
    design = json.loads((room_dir / "design.json").read_text())
    state = torch.load(room_dir / "augmented" / "saved-map.pt", map_location="cpu", weights_only=True)
    poses = design["poses"]
    room_out = out / f"room-{room:02d}"
    cached = _load_or_cache(room_dir, room_out, poses, detector, segmenter, args.force_cache)
    if args.base_pred_dir is None:
        base_prediction = _base_elevation_prediction(room_dir, poses, state)
        base_source = "reloaded OneMap feature grid projected through eight RGB-D views"
    else:
        prediction_file = args.base_pred_dir / f"room-{room:02d}.npz"
        if not prediction_file.exists():
            raise FileNotFoundError(prediction_file)
        base_prediction = np.load(prediction_file)["pred"].astype(np.uint8)
        base_source = str(prediction_file)
    predicted_terrain = _predicted_terrain_points(room_dir, poses, state, base_prediction)
    observations = _observations(room_dir, poses, cached, state, predicted_terrain, args.min_detection_score)
    for pose, data in zip(poses, cached):
        view = int(pose["index"])
        dump(room_out / "detections" / f"view-{view:02d}.json",
             {"view": view, "model": data.get("model", {}), "detections": data["metadata"]})
    filter_counts = Counter()
    candidate_count = 0
    for data in cached:
        for item in data["metadata"]:
            candidate_count += 1
            filter_counts[item.get("filter_reason", "unknown")] += 1
    maps = np.load(room_dir / "semantic_elevation" / "maps.npz")
    gt, valid = maps["GT_semantic"], maps["valid_GT"]
    baseline = three_class_score(gt, base_prediction, valid)
    baseline.update({"room": room, "variant": "base_map", "pose_source": "Habitat design pose",
                     "base_prediction_source": base_source, "semantic_priority": "large > small > terrain"})
    variants = {"base_map": baseline}
    _map_visual(gt, base_prediction, valid, room_out / "base-map-three-class-map.png", f"Room {room} base map")
    for variant in ("per_view", "cross_view"):
        prediction, audit = _prediction_for_variant(base_prediction, observations, variant, args.min_support)
        report = three_class_score(gt, prediction, valid)
        report.update({"room": room, "variant": variant, "audit": audit, "small_support_views": args.min_support,
                       "pose_source": "Habitat design pose", "semantic_priority": "large > small > terrain"})
        variants[variant] = report
        _map_visual(gt, prediction, valid, room_out / f"{variant}-three-class-map.png", f"Room {room} {variant}")
    first_rgb = np.asarray(Image.open(room_dir / "augmented" / "view-00-rgb.png"))[:, :, :3]
    first_meta = cached[0]["metadata"]
    _view_overlay(first_rgb, first_meta, cached[0]["masks"], room_out / "view-00-grounded-sam2.png")
    dump(room_out / "report.json", {"room": room, "variants": variants, "observations": len(observations),
                                     "candidate_count": candidate_count,
                                     "candidate_filter_counts": dict(filter_counts),
                                     "predicted_terrain_support_points": int(len(predicted_terrain)),
                                     "model_revisions": {"detector": detector.revision if detector else MODEL_REVISIONS["IDEA-Research/grounding-dino-tiny"],
                                                          "sam2": segmenter.revision if segmenter else MODEL_REVISIONS["facebook/sam2.1-hiera-tiny"]}})
    return {"room": room, "variants": variants, "candidate_count": candidate_count,
            "candidate_filter_counts": dict(filter_counts),
            "predicted_terrain_support_points": int(len(predicted_terrain))}


def pooled(reports: list[dict], variant: str) -> dict:
    cm = np.zeros((4, 4), dtype=np.int64)
    for report in reports:
        cm += np.asarray(report["variants"][variant]["cm"], dtype=np.int64)
    # Reuse the exact count formula without reconstructing map arrays.
    rows = []
    for class_id, name in CLASS_NAMES.items():
        gt_count = int(cm[class_id].sum()); predicted = int(cm[:, class_id].sum()); tp = int(cm[class_id, class_id])
        recall = 100 * tp / gt_count if gt_count else None; precision = 100 * tp / predicted if predicted else None
        f1 = None if recall is None or precision is None or recall + precision == 0 else 2 * recall * precision / (recall + precision)
        rows.append({"class": name, "gt": gt_count, "correct": tp, "predicted": predicted, "fp": predicted - tp,
                     "fn": gt_count - tp, "precision": precision, "recall": recall, "f1": f1})
    total = int(cm.sum())
    return {"variant": variant, "accuracy": 100 * float(np.trace(cm)) / total if total else 0.0,
            "correct": int(np.trace(cm)), "GT": total, "classes": rows, "cm": cm.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rooms", type=int, nargs="+", default=[9, 10, 11, 17])
    parser.add_argument("--detector-threshold", type=float, default=0.24)
    parser.add_argument("--text-threshold", type=float, default=0.18)
    parser.add_argument("--tile-size", type=int, default=448)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-detection-score", type=float, default=0.32,
                        help="post-cache detector confidence gate for mapping")
    parser.add_argument("--base-pred-dir", type=Path, default=None,
                        help="optional directory containing room-XX.npz with the base prediction in pred")
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cache_complete = all(
        all((args.out / f"room-{room:02d}" / "detections" / f"view-{view:02d}.{suffix}").exists()
            for view in range(8) for suffix in ("json", "npz"))
        for room in args.rooms
    )
    if args.force_cache or not cache_complete:
        detector = GroundingDinoDetector(threshold=args.detector_threshold, text_threshold=args.text_threshold,
                                          tile_size=args.tile_size, tile_overlap=args.tile_overlap)
        segmenter = Sam2BoxSegmenter()
    else:
        detector = segmenter = None
    reports = [run_room(room, args.base, args.out, detector, segmenter, args) for room in args.rooms]
    summary = {"protocol": {"rooms": args.rooms, "views_per_room": 8, "camera_height_m": 0.3,
                             "forward_offset_m": 0.3, "pose_source": "Habitat design pose",
                             "semantic_classes": {"terrain": "floor and stairs", "small": "banana/tin_can/cracker_box",
                                                  "large": "all other mapped classes"},
                             "model_revisions": MODEL_REVISIONS, "detector_threshold": args.detector_threshold,
                             "text_threshold": args.text_threshold, "tile_size": args.tile_size,
                             "tile_overlap": args.tile_overlap, "min_support": args.min_support,
                             "min_detection_score": args.min_detection_score,
                             "local_support_estimation": "median height of depth-projected points in OneMap-predicted terrain cells within 0.30 m XY; at least three support points required",
                             "prediction_uses_GT_floor_or_mesh": False,
                             "base_prediction_dir": str(args.base_pred_dir) if args.base_pred_dir else None},
              "metric_definition": {
                  "per_class_precision": "TP / (TP + FP)",
                  "per_class_recall": "TP / (TP + FN)",
                  "per_class_f1": "2 * precision * recall / (precision + recall)",
                  "valid_domain": "all valid GT semantic-elevation XY cells; missing prediction label 0 counts as error",
                  "class_ids": {"1": "terrain (floor + stairs)", "2": "small objects", "3": "large objects"}
              },
              "rooms": reports, "pooled": {variant: pooled(reports, variant) for variant in ("base_map", "per_view", "cross_view")},
              "elapsed_seconds": time.time() - started}
    dump(args.out / "grounded-sam2-three-class-summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
