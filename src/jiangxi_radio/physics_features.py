"""Derived spatial features from per-station received-power predictions."""

from __future__ import annotations

import numpy as np


def received_power_features(
    received_power_dbm: np.ndarray,
    *,
    signal_min_dbm: float = -135.0,
    signal_max_dbm: float = -40.0,
    detection_threshold_dbm: float = -95.0,
) -> np.ndarray:
    """Return spatially varying, height-specific propagation features.

    The output columns are normalized strongest power, second strongest power,
    mean of the three strongest powers, fraction of stations above threshold,
    and the strongest-to-second-strongest power gap.
    """
    values = np.asarray(received_power_dbm, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("received_power_dbm must have shape [points, stations]")
    if not np.isfinite(values).all():
        raise ValueError("received_power_dbm must be finite")
    if (
        not np.isfinite(signal_min_dbm)
        or not np.isfinite(signal_max_dbm)
        or not np.isfinite(detection_threshold_dbm)
        or signal_max_dbm <= signal_min_dbm
    ):
        raise ValueError("signal range must be finite with max > min")
    strongest = np.max(values, axis=1)
    ordered = np.sort(values, axis=1)[:, ::-1]
    second = ordered[:, min(1, ordered.shape[1] - 1)]
    top3 = ordered[:, : min(3, ordered.shape[1])].mean(axis=1)
    strong_fraction = np.mean(values >= detection_threshold_dbm, axis=1)
    gap = np.clip(strongest - second, 0.0, signal_max_dbm - signal_min_dbm)
    scale = signal_max_dbm - signal_min_dbm
    return np.column_stack(
        (
            np.clip((strongest - signal_min_dbm) / scale, 0.0, 1.0),
            np.clip((second - signal_min_dbm) / scale, 0.0, 1.0),
            np.clip((top3 - signal_min_dbm) / scale, 0.0, 1.0),
            strong_fraction,
            gap / scale,
        )
    ).astype(np.float32)
