"""Small-object evidence helpers shared by the deployable map evaluator.

The helpers are intentionally NumPy-only so geometry and fusion behavior can be
tested without loading Habitat, CLIP, SAM, or CUDA.
"""

import numpy as np


def _probability_vector(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("probability vectors must be finite one-dimensional arrays")
    values = np.maximum(values, 0.0)
    total = values.sum()
    if total <= 0.0:
        return np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    return values / total


def bayesian_update(prior, likelihood, weight=1.0):
    """Fuse one calibrated class likelihood into an instance posterior."""
    if weight < 0:
        raise ValueError("weight must be non-negative")
    prior = _probability_vector(prior)
    likelihood = _probability_vector(likelihood)
    if prior.shape != likelihood.shape:
        raise ValueError("prior and likelihood must have the same shape")
    posterior = prior * np.power(likelihood, float(weight))
    return _probability_vector(posterior)


def geometric_small_candidate_mask(
    world_y,
    valid,
    floor_height,
    min_height=0.02,
    max_height=0.45,
):
    """Return possible low obstacles from projected height above a local plane.

    This is a proposal mask only.  A positive height interval catches small
    objects that the semantic frontend missed, while the later crop classifier
    and instance posterior decide whether a proposal is actually small.
    """
    world_y = np.asarray(world_y, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if world_y.shape != valid.shape:
        raise ValueError("world_y and valid must have the same shape")
    if min_height < 0 or max_height <= min_height:
        raise ValueError("invalid height interval")
    height = world_y - float(floor_height)
    return valid & np.isfinite(height) & (height >= min_height) & (height <= max_height)


def apply_semantic_priority(
    large_evidence,
    small_evidence,
    floor_evidence,
    large_threshold=0.90,
    small_threshold=0.60,
    floor_threshold=0.50,
):
    """Convert independent evidence layers to labels: large > small > floor."""
    large = np.asarray(large_evidence, dtype=np.float32)
    small = np.asarray(small_evidence, dtype=np.float32)
    floor = np.asarray(floor_evidence, dtype=np.float32)
    if not (large.shape == small.shape == floor.shape):
        raise ValueError("evidence layers must have identical shapes")
    pred = np.zeros(large.shape, dtype=np.uint8)
    pred[floor >= float(floor_threshold)] = 1
    pred[small >= float(small_threshold)] = 2
    pred[large >= float(large_threshold)] = 3
    return pred


def semantic_likelihood(class_scores, small_ids=(40, 41, 42)):
    """Collapse open-vocabulary class scores into floor/small/large likelihoods."""
    scores = np.asarray(class_scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("class_scores must be one-dimensional")
    scores = scores - np.max(scores)
    probs = np.exp(scores)
    probs /= probs.sum() if probs.sum() else 1.0
    small_ids = tuple(int(i) for i in small_ids if 0 <= int(i) < len(probs))
    small_mass = probs[list(small_ids)].sum() if small_ids else 0.0
    floor_mass = probs[1] if len(probs) > 1 else 0.0
    large_mass = max(0.0, 1.0 - small_mass - floor_mass)
    return np.array([floor_mass, small_mass, large_mass], dtype=np.float64)


def categorical_likelihood(labels, small_ids=(40, 41, 42), floor_ids=(1, 15)):
    """Build a likelihood from per-pixel crop labels, preserving mask mass."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    floor = np.isin(labels, tuple(floor_ids)).sum()
    small = np.isin(labels, tuple(small_ids)).sum()
    large = labels.size - floor - small
    return np.array([floor, small, large], dtype=np.float64) / labels.size
