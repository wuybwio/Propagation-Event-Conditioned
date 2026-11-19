"""Validation-only calibration utilities for heteroscedastic field predictions.

The calibration temperature is fitted from a held-out validation or calibration split and is
applied only to the predicted variance.  The predictive mean is never changed.
This keeps signal-error metrics comparable while making interval and combined
probability metrics use the calibrated uncertainty.
"""

from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np


def _masked_arrays(
    mean_dbm: np.ndarray,
    variance_db2: np.ndarray,
    target_dbm: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(mean_dbm, dtype=float)
    variance = np.asarray(variance_db2, dtype=float)
    target = np.asarray(target_dbm, dtype=float)
    valid = np.broadcast_to(np.asarray(mask, dtype=bool), mean.shape)
    if mean.shape != variance.shape or mean.shape != target.shape:
        raise ValueError("mean, variance and target must have identical shapes")
    if not np.any(valid):
        raise ValueError("calibration mask selects no voxels")
    mean = mean[valid]
    variance = variance[valid]
    target = target[valid]
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or not np.isfinite(target).all():
        raise ValueError("calibration arrays must be finite")
    if np.any(variance <= 0.0):
        raise ValueError("calibration requires positive variance")
    return mean, variance, target


def estimate_temperature(
    mean_dbm: np.ndarray,
    variance_db2: np.ndarray,
    target_dbm: np.ndarray,
    mask: np.ndarray,
    *,
    minimum: float = 0.05,
    maximum: float = 100.0,
) -> float:
    """Estimate a scalar variance temperature from validation residuals.

    For a scalar scale ``T`` in ``T * variance``, the Gaussian variance MLE is
    ``T = mean((target - mean)^2 / variance)`` over valid validation voxels.
    The returned value is clipped only to a conservative finite range to avoid
    numerical explosions on tiny validation splits.
    """
    mean, variance, target = _masked_arrays(mean_dbm, variance_db2, target_dbm, mask)
    temperature = float(np.mean(np.square(target - mean) / variance))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("estimated temperature must be finite and positive")
    return float(np.clip(temperature, minimum, maximum))


def apply_temperature(variance_db2: np.ndarray, temperature: float) -> np.ndarray:
    """Scale predictive variance while preserving shape and non-negativity."""
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    variance = np.asarray(variance_db2, dtype=np.float32)
    if not np.isfinite(variance).all() or np.any(variance < 0.0):
        raise ValueError("variance must be finite and non-negative")
    return (variance * float(temperature)).astype(np.float32, copy=False)


def fit_prediction_archive(path: str | Path) -> dict[str, object]:
    """Fit calibration from an evaluation prediction archive.

    The archive must contain held-out validation/calibration predictions and aligned ``target_dbm``
    and ``mask`` arrays.  This intentionally refuses archives that omit the
    split metadata; callers must not accidentally fit on a test archive.
    """
    archive_path = Path(path)
    with np.load(archive_path, allow_pickle=False) as data:
        required = {"mean_dbm", "variance_db2", "target_dbm", "mask", "case_indices"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"prediction archive is missing fields: {missing}")
        mean = np.asarray(data["mean_dbm"])
        variance = np.asarray(data["variance_db2"])
        target = np.asarray(data["target_dbm"])
        mask = np.asarray(data["mask"], dtype=bool)
        case_indices = np.asarray(data["case_indices"], dtype=np.int64)
        if "split" not in data.files:
            raise ValueError("prediction archive must contain an explicit split")
        fit_split = str(data["split"].item())
    if fit_split not in {"val", "calibration"}:
        raise ValueError("variance calibration must be fitted on a validation or calibration split")
    selected_mean, selected_variance, selected_target = _masked_arrays(mean, variance, target, mask)
    temperature = estimate_temperature(mean, variance, target, mask)
    return {
        "schema": "cmf_peno_variance_calibration_v1",
        "fit_split": fit_split,
        "source_predictions": str(archive_path.resolve()),
        "case_indices": [int(value) for value in case_indices.tolist()],
        "valid_voxels": int(selected_mean.size),
        "raw_mean_std_error_db": float(np.sqrt(np.mean(np.square(selected_target - selected_mean)))),
        "raw_mean_variance_db2": float(np.mean(selected_variance)),
        "temperature": float(temperature),
    }


def load_temperature(path: str | Path) -> float:
    """Load a validation-fitted calibration JSON and return its temperature."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "cmf_peno_variance_calibration_v1":
        raise ValueError("unsupported variance calibration schema")
    if payload.get("fit_split") not in {"val", "calibration"}:
        raise ValueError("variance calibration must be fitted on a validation or calibration split")
    temperature = float(payload.get("temperature", float("nan")))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("calibration temperature must be finite and positive")
    return temperature


def masked_probability_mse(
    model_probability: np.ndarray,
    target_probability: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Return MSE of two aligned detection-probability fields on valid voxels."""
    predicted = np.asarray(model_probability, dtype=float)
    target = np.asarray(target_probability, dtype=float)
    valid = np.broadcast_to(np.asarray(mask, dtype=bool), predicted.shape)
    if predicted.shape != target.shape:
        raise ValueError("probability fields must have identical shapes")
    if not np.any(valid):
        raise ValueError("probability mask selects no voxels")
    if not np.isfinite(predicted[valid]).all() or not np.isfinite(target[valid]).all():
        raise ValueError("probability fields must be finite")
    return float(np.mean(np.square(predicted[valid] - target[valid])))
