"""Grounding-DINO/SAM2 helpers for small-object RGB-D mapping.

The geometry functions in this module are dependency-light and are used by
unit tests.  Model adapters import Transformers lazily so importing this file
does not download weights or require CUDA.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

MODEL_REVISIONS = {
    "IDEA-Research/grounding-dino-tiny": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
    "facebook/sam2.1-hiera-tiny": "de431c4043854a71d8101e17995dfe596bf101a5",
}


def restore_tile_box(
    box: Sequence[float],
    tile_origin: Sequence[int],
    tile_size: Sequence[int],
    image_size: Sequence[int],
):
    """Map an ``xyxy`` tile-local box into clipped full-image coordinates."""
    values = np.asarray(box, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        return None
    tile_x, tile_y = (int(tile_origin[0]), int(tile_origin[1]))
    tile_w, tile_h = (int(tile_size[0]), int(tile_size[1]))
    image_h, image_w = (int(image_size[0]), int(image_size[1]))
    if min(tile_w, tile_h, image_h, image_w) <= 0:
        raise ValueError("image and tile dimensions must be positive")
    x0, y0, x1, y1 = values
    x0, x1 = np.clip([x0, x1], 0.0, float(tile_w))
    y0, y1 = np.clip([y0, y1], 0.0, float(tile_h))
    result = np.array([x0 + tile_x, y0 + tile_y, x1 + tile_x, y1 + tile_y])
    result[[0, 2]] = np.clip(result[[0, 2]], 0.0, float(image_w))
    result[[1, 3]] = np.clip(result[[1, 3]], 0.0, float(image_h))
    if result[2] - result[0] < 1.0 or result[3] - result[1] < 1.0:
        return None
    return result


def _box_iou(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    x0 = np.maximum(one[0], many[:, 0])
    y0 = np.maximum(one[1], many[:, 1])
    x1 = np.minimum(one[2], many[:, 2])
    y1 = np.minimum(one[3], many[:, 3])
    inter = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
    area_one = max(0.0, one[2] - one[0]) * max(0.0, one[3] - one[1])
    area_many = np.maximum(0.0, many[:, 2] - many[:, 0]) * np.maximum(0.0, many[:, 3] - many[:, 1])
    return inter / np.maximum(area_one + area_many - inter, 1e-12)


def coarse_label(label: int) -> int:
    """Map MPCat40/added labels to terrain=1, small=2, large=3, unknown=0."""
    label = int(label)
    if label in (1, 15):
        return 1
    if 40 <= label <= 42:
        return 2
    if label in (16, 43, 255):
        return 0
    return 3


def small_label_from_phrase(phrase: str) -> int | None:
    """Map exactly one explicit class word; reject mixed/open-set phrases."""
    text = " ".join(str(phrase).lower().replace(".", " ").split())
    text = text.removeprefix("a ").removeprefix("an ")
    aliases = {
        "banana": 40,
        "tin": 41,
        "can": 41,
        "tin can": 41,
        "cracker": 42,
        "cracker box": 42,
        "box of crackers": 42,
    }
    return aliases.get(text)


def compact_obstacle_geometry(
    points: np.ndarray,
    support_height: float,
    max_horizontal_extent: float = 0.40,
    max_vertical_extent: float = 0.14,
    min_height: float = -0.05,
    max_height: float = 0.45,
) -> tuple[bool, str]:
    """Reject masks whose RGB-D support is not a compact low obstacle."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 5:
        return False, "too_few_depth_points"
    if not np.isfinite(points).all():
        return False, "nonfinite_points"
    extent = np.ptp(points, axis=0)
    if max(float(extent[0]), float(extent[2])) > float(max_horizontal_extent):
        return False, "too_wide"
    if float(extent[1]) > float(max_vertical_extent):
        return False, "too_tall"
    relative_height = points[:, 1] - float(support_height)
    p05, p95 = np.percentile(relative_height, [5, 95])
    if p05 < float(min_height):
        return False, "below_floor"
    if p95 > float(max_height):
        return False, "too_high"
    return True, "accepted"


def estimate_local_support_height(
    candidate_points: np.ndarray,
    terrain_points: np.ndarray,
    radius: float = 0.30,
    min_support_points: int = 3,
) -> tuple[float | None, int]:
    """Estimate local ground/stair height from nearby predicted-terrain points.

    This uses only projected sensor points and the current predicted terrain
    mask. It does not consume a scene floor datum or semantic GT.
    """
    candidate_points = np.asarray(candidate_points, dtype=np.float32)
    terrain_points = np.asarray(terrain_points, dtype=np.float32)
    if candidate_points.ndim != 2 or candidate_points.shape[1] != 3:
        raise ValueError("candidate_points must be Nx3")
    if terrain_points.ndim != 2 or terrain_points.shape[1] != 3:
        raise ValueError("terrain_points must be Mx3")
    if radius <= 0 or min_support_points < 1:
        raise ValueError("invalid local support settings")
    terrain_points = terrain_points[np.isfinite(terrain_points).all(axis=1)]
    candidate_points = candidate_points[np.isfinite(candidate_points).all(axis=1)]
    if len(terrain_points) < min_support_points or len(candidate_points) == 0:
        return None, 0
    from scipy.spatial import cKDTree

    center = np.median(candidate_points[:, [0, 2]], axis=0)
    tree = cKDTree(terrain_points[:, [0, 2]])
    indices = tree.query_ball_point(center, float(radius))
    if len(indices) < min_support_points:
        return None, len(indices)
    support = terrain_points[np.asarray(indices, dtype=np.int64)]
    return float(np.median(support[:, 1])), int(len(support))


def nms_detections(
    detections: Sequence[dict], iou_threshold: float = 0.5, class_agnostic: bool = False
) -> list[dict]:
    """Keep high-scoring boxes, optionally suppressing across text phrases."""
    kept = []
    groups = [list(detections)] if class_agnostic else [
        [item for item in detections if str(item.get("phrase", "small_object")) == phrase]
        for phrase in sorted({str(item.get("phrase", "small_object")) for item in detections})
    ]
    for group in groups:
        order = sorted(range(len(group)), key=lambda i: float(group[i]["score"]), reverse=True)
        while order:
            current = order.pop(0)
            kept.append(group[current])
            if not order:
                break
            boxes = np.asarray([group[index]["box"] for index in order], dtype=np.float64)
            overlaps = _box_iou(np.asarray(group[current]["box"], dtype=np.float64), boxes)
            order = [index for index, overlap in zip(order, overlaps) if overlap <= iou_threshold]
    return sorted(kept, key=lambda item: float(item["score"]), reverse=True)


def project_mask_to_world(
    mask: np.ndarray,
    depth: np.ndarray,
    pose: dict,
    intrinsics: Sequence[float] = (320.0, 320.0, 320.0, 320.0),
) -> np.ndarray:
    """Project valid mask pixels through depth into world XYZ points.

    The camera convention matches ``score_voxel_map.depth_points``: optical
    depth points are ``[(u-cx)d/fx, -(v-cy)d/fy, -d]``.
    """
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth, dtype=np.float32)
    if mask.shape != depth.shape or mask.ndim != 2:
        raise ValueError("mask and depth must be identically shaped HxW arrays")
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    if min(fx, fy) <= 0:
        raise ValueError("focal lengths must be positive")
    valid = mask & np.isfinite(depth) & (depth > 0)
    rows, cols = np.where(valid)
    if len(rows) == 0:
        return np.empty((0, 3), dtype=np.float32)
    distance = depth[rows, cols]
    camera = np.stack(
        [(cols - cx) * distance / fx, -(rows - cy) * distance / fy, -distance], axis=1
    )
    rotation = Rotation.from_quat(np.asarray(pose["rotation_xyzw"], dtype=np.float64)).as_matrix()
    origin = np.asarray(pose["camera_world_position"], dtype=np.float64)
    return (camera @ rotation.T + origin).astype(np.float32)


def _voxel_keys(points: np.ndarray, voxel_size: float) -> set[tuple[int, int, int]]:
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return set()
    return {tuple(int(value) for value in row) for row in np.floor(points / voxel_size).astype(np.int64)}


def associate_observations(
    observations: Sequence[dict], voxel_size: float = 0.05, centroid_gate: float = 0.18,
    surface_gate: float = 0.08,
) -> list[list[dict]]:
    """Cluster per-view observations using 3D voxel overlap and centroid distance."""
    if voxel_size <= 0 or centroid_gate <= 0 or surface_gate < 0:
        raise ValueError("voxel_size and centroid_gate must be positive; surface_gate must be non-negative")
    clusters: list[list[dict]] = []
    centroids: list[np.ndarray] = []
    keys: list[set[tuple[int, int, int]]] = []
    for observation in observations:
        points = np.asarray(observation["points"], dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            continue
        centroid = points.mean(axis=0)
        current_keys = _voxel_keys(points, voxel_size)
        best = None
        best_overlap = 0.0
        for index, (known_centroid, known_keys) in enumerate(zip(centroids, keys)):
            distance = float(np.linalg.norm(centroid - known_centroid))
            if distance > centroid_gate or not current_keys or not known_keys:
                continue
            overlap = len(current_keys & known_keys) / max(1, len(current_keys | known_keys))
            if overlap == 0.0:
                from scipy.spatial import cKDTree

                nearest = cKDTree(np.asarray(observation["points"], dtype=np.float32)).query(
                    np.asarray([point for item in clusters[index] for point in np.asarray(item["points"])]),
                    k=1,
                )[0]
                surface_gap = float(np.min(nearest)) if len(nearest) else np.inf
                overlap = 1e-6 if surface_gap <= surface_gate else 0.0
            if overlap > best_overlap:
                best = index
                best_overlap = overlap
        if best is None:
            clusters.append([observation])
            centroids.append(centroid)
            keys.append(current_keys)
        else:
            clusters[best].append(observation)
            centroids[best] = np.mean([np.asarray(item["points"]).mean(axis=0) for item in clusters[best]], axis=0)
            keys[best] |= current_keys
    return clusters


def rasterize_small_evidence(prediction: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Write small labels while preserving the large-object priority."""
    output = np.asarray(prediction, dtype=np.uint8).copy()
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return output
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError("indices must have shape Nx2")
    for row, col in indices:
        if 0 <= row < output.shape[0] and 0 <= col < output.shape[1] and output[row, col] != 3:
            output[row, col] = 2
    return output


class GroundingDinoDetector:
    """Lazy Transformers Grounding-DINO adapter with optional tiled inference."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        revision: str | None = None,
        device: str | None = None,
        threshold: float = 0.22,
        text_threshold: float = 0.18,
        tile_size: int = 448,
        tile_overlap: float = 0.25,
    ):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.model_id = model_id
        self.revision = revision or MODEL_REVISIONS.get(model_id)
        self.device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        self.threshold = float(threshold)
        self.text_threshold = float(text_threshold)
        self.tile_size = int(tile_size)
        self.tile_overlap = float(tile_overlap)
        if not 0 <= self.tile_overlap < 1 or self.tile_size <= 0:
            raise ValueError("invalid tile settings")
        self.processor = AutoProcessor.from_pretrained(model_id, revision=self.revision)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, revision=self.revision
        ).to(self.device).eval()

    @staticmethod
    def _starts(length: int, tile: int, overlap: float) -> list[int]:
        if tile >= length:
            return [0]
        stride = max(1, int(round(tile * (1.0 - overlap))))
        starts = list(range(0, max(1, length - tile + 1), stride))
        last = length - tile
        if starts[-1] != last:
            starts.append(last)
        return starts

    def detect(self, image: np.ndarray, prompts: Sequence[str]) -> list[dict]:
        import torch
        from PIL import Image

        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be HxWx3 RGB")
        if not prompts:
            return []
        height, width = image.shape[:2]
        tile = min(self.tile_size, height, width)
        output = []
        # Independent prompts reduce mixed phrase labels such as "tin cracker
        # box" which otherwise make category assignment ambiguous.
        for prompt in prompts:
            for top in self._starts(height, tile, self.tile_overlap):
                for left in self._starts(width, tile, self.tile_overlap):
                    crop = Image.fromarray(image[top : top + tile, left : left + tile])
                    inputs = self.processor(images=crop, text=[[str(prompt)]], return_tensors="pt")
                    inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
                    with torch.inference_mode():
                        result = self.model(**inputs)
                    detections = self.processor.post_process_grounded_object_detection(
                        result,
                        input_ids=inputs.get("input_ids"),
                        threshold=self.threshold,
                        text_threshold=self.text_threshold,
                        target_sizes=[(tile, tile)],
                    )[0]
                    for box, score, phrase in zip(detections["boxes"], detections["scores"], detections["text_labels"]):
                        restored = restore_tile_box(box.detach().cpu().numpy(), (left, top), (tile, tile), (height, width))
                        if restored is None:
                            continue
                        output.append({"box": restored, "score": float(score), "phrase": str(prompt),
                                       "raw_phrase": str(phrase).strip()})
        return nms_detections(output, iou_threshold=0.60, class_agnostic=True)


class Sam2BoxSegmenter:
    """SAM2.1 box-prompt mask adapter."""

    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-tiny",
        revision: str | None = None,
        device: str | None = None,
    ):
        from transformers import Sam2Model, Sam2Processor

        self.model_id = model_id
        self.revision = revision or MODEL_REVISIONS.get(model_id)
        self.device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        self.processor = Sam2Processor.from_pretrained(model_id, revision=self.revision)
        self.model = Sam2Model.from_pretrained(model_id, revision=self.revision).to(self.device).eval()

    def segment(self, image: np.ndarray, boxes: Sequence[Sequence[float]]) -> list[dict]:
        import torch
        from PIL import Image

        image = np.asarray(image)
        if len(boxes) == 0:
            return []
        # Transformers 4.57's fast image processor asks for this PyTorch API
        # before checking whether its ROCm-only compile path applies. PyTorch
        # 2.2.2 predates the helper; this false shim is safe for eager CUDA/CPU
        # inference and avoids changing the project's installed torch build.
        if not hasattr(torch.compiler, "is_compiling"):
            torch.compiler.is_compiling = lambda: False
        inputs = self.processor(
            images=Image.fromarray(image),
            input_boxes=[[list(map(float, box)) for box in boxes]],
            return_tensors="pt",
        )
        original_sizes = inputs["original_sizes"]
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), original_sizes)[0]
        scores = outputs.iou_scores[0].detach().cpu()
        results = []
        for box_index in range(len(boxes)):
            best = int(scores[box_index].argmax())
            results.append(
                {"mask": masks[box_index, best].numpy().astype(bool), "mask_score": float(scores[box_index, best])}
            )
        return results
