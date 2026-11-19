"""Uncertainty-aware candidate measurement scoring utilities."""

from __future__ import annotations

import numpy as np


def active_measurement_score(
    uncertainty: np.ndarray,
    physical_disagreement: np.ndarray,
    route_weight: np.ndarray,
    distance_to_measurement_km: np.ndarray,
    *,
    uncertainty_weight: float = 1.0,
    disagreement_weight: float = 0.75,
    route_weight_scale: float = 0.5,
    distance_penalty: float = 0.05,
) -> np.ndarray:
    """Score points that are informative, route-relevant, and not redundant."""
    arrays = [np.asarray(value, dtype=float) for value in (
        uncertainty, physical_disagreement, route_weight, distance_to_measurement_km
    )]
    if any(value.ndim != 1 for value in arrays):
        raise ValueError("all score inputs must be one-dimensional")
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("all score inputs must have the same length")
    if distance_penalty < 0:
        raise ValueError("distance_penalty must be non-negative")
    return (
        uncertainty_weight * arrays[0]
        + disagreement_weight * arrays[1]
        + route_weight_scale * arrays[2]
        - distance_penalty * arrays[3]
    )
