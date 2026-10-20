import numpy as np

from small_object_fusion import (
    bayesian_update,
    categorical_likelihood,
    geometric_small_candidate_mask,
    apply_semantic_priority,
)


def test_geometric_candidate_requires_small_positive_height():
    world_y = np.array([[0.00, 0.01, 0.08, 0.60]], dtype=np.float32)
    valid = np.ones_like(world_y, dtype=bool)

    mask = geometric_small_candidate_mask(
        world_y,
        valid,
        floor_height=0.0,
        min_height=0.02,
        max_height=0.45,
    )

    np.testing.assert_array_equal(mask, [[False, False, True, False]])


def test_bayesian_updates_accumulate_small_object_evidence():
    posterior = np.array([0.80, 0.10, 0.10], dtype=np.float64)
    observation = np.array([0.05, 0.90, 0.05], dtype=np.float64)

    posterior = bayesian_update(posterior, observation)
    posterior = bayesian_update(posterior, observation)

    assert posterior[1] > posterior[0]
    np.testing.assert_allclose(posterior.sum(), 1.0)


def test_crop_pixel_likelihood_uses_small_mask_fraction():
    labels = np.array([1, 1, 40, 41, 42, 3], dtype=np.uint8)

    likelihood = categorical_likelihood(labels)

    np.testing.assert_allclose(likelihood, [2 / 6, 3 / 6, 1 / 6])


def test_priority_keeps_large_above_small_but_does_not_promote_weak_large():
    large = np.array([[0.95, 0.40]], dtype=np.float32)
    small = np.array([[0.80, 0.80]], dtype=np.float32)
    floor = np.array([[0.90, 0.90]], dtype=np.float32)

    pred = apply_semantic_priority(
        large,
        small,
        floor,
        large_threshold=0.90,
        small_threshold=0.60,
        floor_threshold=0.50,
    )

    np.testing.assert_array_equal(pred, [[3, 2]])
