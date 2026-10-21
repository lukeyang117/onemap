import numpy as np

from grounded_small_objects import (
    associate_observations,
    compact_obstacle_geometry,
    coarse_label,
    estimate_local_support_height,
    nms_detections,
    project_mask_to_world,
    rasterize_small_evidence,
    restore_tile_box,
    small_label_from_phrase,
)


def test_three_class_mapping_keeps_wall_as_large_and_ceiling_unmapped():
    assert coarse_label(0) == 3
    assert coarse_label(16) == 0
    assert coarse_label(43) == 0


def test_generic_box_phrase_is_not_promoted_to_cracker_box():
    assert small_label_from_phrase("a box") is None
    assert small_label_from_phrase("a cracker box") == 42
    assert small_label_from_phrase("tin") == 41
    assert small_label_from_phrase("banana a toy") is None
    assert small_label_from_phrase("tin cracker box") is None


def test_compact_geometry_rejects_wall_sized_mask_and_accepts_low_obstacle():
    obstacle = np.array([
        [0.0, 0.02, 0.0], [0.10, 0.10, 0.08], [0.20, 0.15, 0.04],
        [0.05, 0.06, 0.02], [0.15, 0.14, 0.06],
    ])
    wall = np.array([
        [0.0, 0.0, 0.0], [0.8, 0.4, 0.1], [0.1, 0.2, 0.8],
        [0.4, 0.3, 0.4], [0.7, 0.1, 0.7],
    ])

    assert compact_obstacle_geometry(obstacle, support_height=0.0)[0]
    assert not compact_obstacle_geometry(wall, support_height=0.0)[0]


def test_local_support_height_comes_from_nearby_predicted_terrain_points():
    terrain = np.array([
        [0.00, 0.10, 0.00], [0.04, 0.11, 0.03], [0.08, 0.10, -0.02],
        [0.12, 0.12, 0.02], [2.00, 5.0, 2.00],
    ])
    obstacle = np.array([[0.04, 0.22, 0.01], [0.05, 0.24, 0.02], [0.06, 0.25, 0.00]])

    height, support_count = estimate_local_support_height(obstacle, terrain, radius=0.30, min_support_points=3)

    assert support_count == 4
    assert abs(height - 0.105) < 1e-6


def test_restore_tile_box_clips_back_to_full_image():
    box = restore_tile_box(
        [-8, 12, 40, 50], tile_origin=(100, 200), tile_size=(200, 160), image_size=(640, 480)
    )

    np.testing.assert_allclose(box, [100, 212, 140, 250])


def test_restore_tile_box_drops_degenerate_box_after_clipping():
    assert restore_tile_box(
        [90, 4, 110, 20], tile_origin=(0, 0), tile_size=(100, 60), image_size=(80, 60)
    ) is None


def test_detection_nms_is_class_agnostic_for_small_object_prompts():
    detections = [
        {"box": np.array([0, 0, 20, 20]), "score": 0.9, "phrase": "banana"},
        {"box": np.array([1, 1, 19, 19]), "score": 0.8, "phrase": "tin can"},
        {"box": np.array([40, 40, 60, 60]), "score": 0.7, "phrase": "cracker box"},
    ]

    kept = nms_detections(detections, iou_threshold=0.5, class_agnostic=True)

    assert [item["phrase"] for item in kept] == ["banana", "cracker box"]


def test_project_mask_to_world_uses_depth_and_camera_pose():
    mask = np.zeros((640, 640), dtype=bool)
    mask[320, 321] = True
    depth = np.zeros((640, 640), dtype=np.float32)
    depth[320, 321] = 2.0
    pose = {
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "camera_world_position": [1.0, 2.0, 3.0],
    }

    points = project_mask_to_world(mask, depth, pose)

    np.testing.assert_allclose(points, [[1.00625, 2.0, 1.0]], atol=1e-6)


def test_associate_observations_merges_same_surface_across_views_only():
    surface = np.array([[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [0.0, 0.04, 0.0]])
    nearby_but_distinct = surface + np.array([0.5, 0.0, 0.0])
    observations = [
        {"view": 0, "points": surface, "score": 0.8},
        {"view": 1, "points": surface + np.array([0.005, 0.0, 0.0]), "score": 0.9},
        {"view": 1, "points": nearby_but_distinct, "score": 0.7},
    ]

    clusters = associate_observations(observations, voxel_size=0.05, centroid_gate=0.12)

    assert sorted(sorted(item["view"] for item in group) for group in clusters) == [[0, 1], [1]]


def test_associate_observations_matches_nearby_nonoverlapping_surface_samples():
    front = np.array([[0.0, 0.0, 0.0], [0.02, 0.02, 0.0], [0.04, 0.0, 0.0]])
    side = front + np.array([0.06, 0.0, 0.0])

    clusters = associate_observations(
        [{"view": 0, "points": front}, {"view": 1, "points": side}],
        voxel_size=0.05,
        centroid_gate=0.12,
        surface_gate=0.08,
    )

    assert len(clusters) == 1
    assert {item["view"] for item in clusters[0]} == {0, 1}


def test_small_evidence_never_overwrites_large_priority():
    prediction = np.array([[1, 3, 0]], dtype=np.uint8)

    fused = rasterize_small_evidence(prediction, np.array([[0, 0], [0, 1], [0, 2]]))

    np.testing.assert_array_equal(fused, [[2, 3, 2]])
    np.testing.assert_array_equal(prediction, [[1, 3, 0]])
