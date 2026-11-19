"""Numpy metrics for masked probabilistic radio-map evaluation."""

from __future__ import annotations

import numpy as np


def masked_probability_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Compute MAE and Brier score over labeled elements only."""
    pred = np.asarray(prediction, dtype=float)
    truth = np.asarray(target, dtype=float)
    weights = np.asarray(mask, dtype=float)
    if pred.shape != truth.shape or pred.shape != weights.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    valid = weights > 0
    if not np.any(valid):
        return {"mae": float("nan"), "brier": float("nan"), "count": 0.0}
    error = pred[valid] - truth[valid]
    return {
        "mae": float(np.abs(error).mean()),
        "brier": float(np.square(error).mean()),
        "count": float(valid.sum()),
    }
