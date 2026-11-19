from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detection import at_least_k_probability, batch_correlated_at_least_k_probability


@dataclass(frozen=True)
class RouteCoverageResult:
    point_probabilities: np.ndarray
    covered: np.ndarray
    coverage_ratio: float
    total_gap_km: float
    max_contiguous_gap_km: float


def _point_support_lengths(cumulative_distance_km: np.ndarray) -> np.ndarray:
    intervals = np.diff(cumulative_distance_km)
    weights = np.empty_like(cumulative_distance_km)
    weights[0] = intervals[0] / 2.0
    weights[-1] = intervals[-1] / 2.0
    if len(weights) > 2:
        weights[1:-1] = (intervals[:-1] + intervals[1:]) / 2.0
    return weights


def _maximum_run_weight(mask: np.ndarray, weights: np.ndarray) -> float:
    maximum = 0.0
    current = 0.0
    for selected, weight in zip(mask, weights):
        if selected:
            current += float(weight)
            maximum = max(maximum, current)
        else:
            current = 0.0
    return maximum


def evaluate_route_coverage(
    station_probabilities: np.ndarray,
    required_k: np.ndarray,
    cumulative_distance_km: np.ndarray,
    probability_threshold: float,
    *,
    shadow_correlation: float = 0.0,
    correlation_samples: int = 4096,
    correlation_seed: int = 0,
) -> RouteCoverageResult:
    probabilities = np.asarray(station_probabilities, dtype=float)
    redundancy = np.asarray(required_k, dtype=int)
    distances = np.asarray(cumulative_distance_km, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("station probabilities must have shape [route points, stations]")
    if len(probabilities) != len(redundancy) or len(probabilities) != len(distances):
        raise ValueError("inputs must contain the same number of route points")
    if len(distances) < 2 or np.any(np.diff(distances) <= 0):
        raise ValueError("route distances must be strictly increasing")
    if np.any(redundancy < 1):
        raise ValueError("required station count must be positive")
    if not 0 < probability_threshold < 1:
        raise ValueError("probability threshold must be between 0 and 1")

    if shadow_correlation == 0.0:
        point_probabilities = np.array(
            [
                at_least_k_probability(point_probabilities, int(k))
                for point_probabilities, k in zip(probabilities, redundancy)
            ],
            dtype=float,
        )
    else:
        point_probabilities = batch_correlated_at_least_k_probability(
            probabilities,
            redundancy,
            correlation=shadow_correlation,
            samples=correlation_samples,
            seed=correlation_seed,
        )
    covered = point_probabilities >= probability_threshold
    support_lengths = _point_support_lengths(distances)
    route_length = float(distances[-1] - distances[0])
    uncovered = ~covered
    total_gap = float(support_lengths[uncovered].sum())
    max_gap = _maximum_run_weight(uncovered, support_lengths)
    return RouteCoverageResult(
        point_probabilities=point_probabilities,
        covered=covered,
        coverage_ratio=1.0 - total_gap / route_length,
        total_gap_km=total_gap,
        max_contiguous_gap_km=max_gap,
    )
