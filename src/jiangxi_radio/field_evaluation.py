"""Pure NumPy received-power field metrics for independent evaluation."""

from __future__ import annotations

import math

import numpy as np


def power_exceedance_probability(
    power_dbm: np.ndarray,
    threshold_dbm: float = -100.0,
    shadow_std_db: float = 6.0,
) -> np.ndarray:
    if not math.isfinite(threshold_dbm) or not math.isfinite(shadow_std_db):
        raise ValueError("threshold and shadow standard deviation must be finite")
    if shadow_std_db <= 0.0:
        raise ValueError("shadow standard deviation must be positive")
    power = np.asarray(power_dbm, dtype=float)
    z = (power - threshold_dbm) / (shadow_std_db * math.sqrt(2.0))
    values = np.fromiter((math.erf(float(item)) for item in z.ravel()), float, z.size)
    return 0.5 * (1.0 + values.reshape(z.shape))


def field_metrics(
    prediction_dbm: np.ndarray,
    reference_dbm: np.ndarray,
    mask: np.ndarray,
    threshold_dbm: float = -100.0,
    shadow_std_db: float = 6.0,
    probability_threshold: float = 0.9,
) -> dict[str, float | int | None]:
    prediction = np.asarray(prediction_dbm, dtype=float)
    reference = np.asarray(reference_dbm, dtype=float)
    if prediction.shape != reference.shape:
        raise ValueError("predicted and reference fields must have identical shapes")
    valid = np.broadcast_to(np.asarray(mask, dtype=bool), prediction.shape)
    if not np.any(valid):
        raise ValueError("evaluation mask contains no valid voxels")
    predicted_values = prediction[valid]
    reference_values = reference[valid]
    if not np.isfinite(predicted_values).all() or not np.isfinite(reference_values).all():
        raise ValueError("valid predicted and reference powers must be finite")
    predicted_probability = power_exceedance_probability(
        predicted_values, threshold_dbm, shadow_std_db
    )
    reference_probability = power_exceedance_probability(
        reference_values, threshold_dbm, shadow_std_db
    )
    predicted_covered = predicted_probability >= probability_threshold
    reference_covered = reference_probability >= probability_threshold
    tp = int(np.sum(predicted_covered & reference_covered))
    fp = int(np.sum(predicted_covered & ~reference_covered))
    fn = int(np.sum(~predicted_covered & reference_covered))
    tn = int(np.sum(~predicted_covered & ~reference_covered))

    def ratio(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    coverage_recall = ratio(tp, tp + fn)
    blind_recall = ratio(tn, tn + fp)
    return {
        "voxels": int(len(predicted_values)),
        "mae_db": float(np.mean(np.abs(predicted_values - reference_values))),
        "rmse_db": float(np.sqrt(np.mean((predicted_values - reference_values) ** 2))),
        "bias_db": float(np.mean(predicted_values - reference_values)),
        "probability_mse": float(np.mean(
            (predicted_probability - reference_probability) ** 2
        )),
        "coverage_iou": ratio(tp, tp + fp + fn),
        "predicted_coverage_fraction": ratio(tp + fp, len(predicted_values)),
        "teacher_coverage_fraction": ratio(tp + fn, len(predicted_values)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "coverage_precision": ratio(tp, tp + fp),
        "coverage_recall": coverage_recall,
        "blind_recall": blind_recall,
        "blind_iou": ratio(tn, tn + fp + fn),
        "false_coverage_rate": ratio(fp, tn + fp),
        "balanced_accuracy": (
            (coverage_recall + blind_recall) / 2.0
            if coverage_recall is not None and blind_recall is not None else None
        ),
    }
