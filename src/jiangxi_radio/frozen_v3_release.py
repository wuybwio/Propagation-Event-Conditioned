"""Leakage-safe utilities for the frozen CMF-PENO-RF v3 paper release."""

from __future__ import annotations

import math

import numpy as np


def _aligned(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    values = tuple(np.asarray(value) for value in arrays)
    if not values or any(value.shape != values[0].shape for value in values[1:]):
        raise ValueError("arrays must have identical shapes")
    return values


def validation_bias_shift(mean: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    mean, target, mask = _aligned(mean, target, mask)
    valid = mask.astype(bool)
    if not np.any(valid):
        raise ValueError("validation mask contains no valid voxels")
    shift = float(np.mean(target[valid] - mean[valid]))
    if not np.isfinite(shift):
        raise ValueError("validation bias shift is not finite")
    return shift


def conformal_multiplier(
    mean: np.ndarray,
    variance: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    target_coverage: float,
) -> float:
    mean, variance, target, mask = _aligned(mean, variance, target, mask)
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target coverage must lie in (0, 1)")
    valid = mask.astype(bool)
    scores = np.abs(target[valid] - mean[valid]) / np.sqrt(
        np.maximum(variance[valid], 1e-6)
    )
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("conformal scores are empty or non-finite")
    level = min(1.0, math.ceil((len(scores) + 1) * target_coverage) / len(scores))
    return max(float(np.quantile(scores, level, method="higher")), 1e-6)


def interval_metrics(
    mean: np.ndarray,
    variance: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    multiplier: float,
) -> dict[str, float]:
    mean, variance, target, mask = _aligned(mean, variance, target, mask)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("interval multiplier must be finite and positive")
    valid = mask.astype(bool)
    if not np.any(valid):
        raise ValueError("mask contains no valid voxels")
    half_width = multiplier * np.sqrt(np.maximum(variance[valid], 1e-6))
    error = np.abs(target[valid] - mean[valid])
    return {
        "coverage": float(np.mean(error <= half_width)),
        "mean_width_db": float(np.mean(2.0 * half_width)),
    }


def normal_detection_probability(
    mean_dbm: np.ndarray,
    variance_db2: np.ndarray | float,
    threshold_dbm: float,
    shadow_std_db: float,
) -> np.ndarray:
    mean = np.asarray(mean_dbm, dtype=float)
    variance = np.broadcast_to(np.asarray(variance_db2, dtype=float), mean.shape)
    if shadow_std_db <= 0.0 or np.any(variance < 0.0):
        raise ValueError("shadow standard deviation and variance must be valid")
    denominator = np.sqrt(shadow_std_db**2 + variance)
    z = (mean - threshold_dbm) / (denominator * math.sqrt(2.0))
    flat = z.ravel()
    values = np.fromiter((math.erf(float(item)) for item in flat), float, len(flat))
    return (0.5 * (1.0 + values.reshape(z.shape))).astype(np.float32)


def union_detection_probability(probabilities: np.ndarray, axis: int = 0) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    return 1.0 - np.prod(1.0 - values, axis=axis)


def greedy_select_sites(
    candidate_probabilities: np.ndarray,
    budget: int,
    mask: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    probabilities = np.asarray(candidate_probabilities, dtype=float)
    if probabilities.ndim < 2:
        raise ValueError("candidate probabilities must be [sites, ...]")
    valid = np.broadcast_to(np.asarray(mask, dtype=bool), probabilities.shape[1:])
    if budget < 1 or budget > probabilities.shape[0] or not np.any(valid):
        raise ValueError("budget or planning mask is invalid")
    union = np.zeros(probabilities.shape[1:], dtype=float)
    selected: list[int] = []
    remaining = set(range(probabilities.shape[0]))
    for _ in range(budget):
        best_index = None
        best_score = -np.inf
        best_union = None
        for index in sorted(remaining):
            proposal = 1.0 - (1.0 - union) * (1.0 - probabilities[index])
            score = float(np.mean(proposal[valid]))
            if score > best_score + 1e-15:
                best_index, best_score, best_union = index, score, proposal
        assert best_index is not None and best_union is not None
        selected.append(best_index)
        remaining.remove(best_index)
        union = best_union
    return selected, union.astype(np.float32)


def planning_metrics(
    probability: np.ndarray,
    mask: np.ndarray,
    probability_threshold: float,
) -> dict[str, float]:
    probability, mask = _aligned(probability, mask)
    valid = mask.astype(bool)
    if not 0.0 <= probability_threshold <= 1.0 or not np.any(valid):
        raise ValueError("probability threshold or mask is invalid")
    values = probability[valid]
    return {
        "mean_detection_probability": float(np.mean(values)),
        "coverage_fraction": float(np.mean(values >= probability_threshold)),
        "blind_fraction": float(np.mean(values < probability_threshold)),
        "minimum_detection_probability": float(np.min(values)),
    }


def representative_case_index(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> int:
    prediction, target, mask = _aligned(prediction, target, mask)
    if prediction.ndim < 2:
        raise ValueError("case arrays must have a leading case dimension")
    errors = []
    for index in range(prediction.shape[0]):
        valid = mask[index].astype(bool)
        if not np.any(valid):
            errors.append(np.inf)
        else:
            errors.append(float(np.mean(np.abs(prediction[index][valid] - target[index][valid]))))
    finite = np.isfinite(errors)
    if not np.any(finite):
        raise ValueError("no case contains valid voxels")
    median = float(np.median(np.asarray(errors)[finite]))
    return int(np.argmin(np.abs(np.asarray(errors) - median)))
