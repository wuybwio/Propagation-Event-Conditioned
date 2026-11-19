"""Build sparse LTE-supervised tensors for the physics-guided map model."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def poisson_binomial_at_least(probabilities: np.ndarray) -> np.ndarray:
    """Return P(N>=1), P(N>=2), P(N>=3) for independent Bernoulli trials."""
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if values.size == 0:
        return np.zeros(3, dtype=float)
    values = np.clip(values, 0.0, 1.0)
    pmf = np.zeros(values.size + 1, dtype=float)
    pmf[0] = 1.0
    for probability in values:
        pmf[1:] = pmf[1:] * (1.0 - probability) + pmf[:-1] * probability
        pmf[0] *= 1.0 - probability
    cumulative = np.cumsum(pmf)
    # For k greater than the number of trials, P(N>=k)=0.  Using the last
    # cumulative bin avoids indexing beyond the PMF for one- or two-trial
    # cells, which are common in sparse radio measurements.
    return np.asarray(
        [1.0 - cumulative[min(k - 1, values.size)] for k in (1, 2, 3)],
        dtype=float,
    )


def rsrp_to_detection_probability(
    rsrp_dbm: np.ndarray | float,
    *,
    threshold_dbm: float = -95.0,
    transition_db: float = 5.0,
) -> np.ndarray:
    """Construct a heuristic LTE RSRP score, not a calibrated monitoring detection probability."""
    values = np.asarray(rsrp_dbm, dtype=float)
    if transition_db <= 0:
        raise ValueError("transition_db must be positive")
    return 1.0 / (1.0 + np.exp(-(values - threshold_dbm) / transition_db))


def _grid_axes(labels: pd.DataFrame, cell_size_deg: float) -> tuple[np.ndarray, np.ndarray]:
    if cell_size_deg <= 0:
        raise ValueError("cell_size_deg must be positive")
    lon = np.asarray(labels["grid_lon"], dtype=float)
    lat = np.asarray(labels["grid_lat"], dtype=float)
    x = np.arange(lon.min(), lon.max() + cell_size_deg * 0.5, cell_size_deg)
    y = np.arange(lat.min(), lat.max() + cell_size_deg * 0.5, cell_size_deg)
    return np.round(x, 9), np.round(y, 9)


def build_observation_targets(
    labels: pd.DataFrame,
    lon_axis: np.ndarray,
    lat_axis: np.ndarray,
    *,
    rsrp_threshold_dbm: float = -95.0,
    transition_db: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build LTE-cell proxy tails; never treat them as monitoring-station truth.

    Match confidence weights supervision, not the probability of detection.
    These tails describe only the observed/matched LTE cells, not unobserved cells.
    """
    required = {"grid_lon", "grid_lat", "matched_eci", "rsrp_mean_dbm", "label_weight"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"missing label columns: {sorted(missing)}")
    height, width = len(lat_axis), len(lon_axis)
    target = np.zeros((3, height, width), dtype=np.float32)
    mask = np.zeros((3, height, width), dtype=np.float32)
    grouped = labels[labels["label_weight"] > 0].groupby(["grid_lon", "grid_lat"], sort=False)
    lon_index = {float(value): index for index, value in enumerate(lon_axis)}
    lat_index = {float(value): index for index, value in enumerate(lat_axis)}
    for (grid_lon, grid_lat), frame in grouped:
        x = lon_index.get(round(float(grid_lon), 9))
        y = lat_index.get(round(float(grid_lat), 9))
        if x is None or y is None:
            continue
        trials = frame.copy()
        trials = trials.loc[
            trials["matched_eci"].notna()
            & trials["matched_eci"].astype(str).str.strip().ne("")
            & np.isfinite(trials["rsrp_mean_dbm"].to_numpy(float))
        ].copy()
        if trials.empty:
            continue
        trials["eci_key"] = trials["matched_eci"].astype(str)
        trials["trial_probability"] = rsrp_to_detection_probability(
            trials["rsrp_mean_dbm"].to_numpy(dtype=float),
            threshold_dbm=rsrp_threshold_dbm,
            transition_db=transition_db,
        )
        confidence = float(np.clip(trials["label_weight"].to_numpy(float), 0, 1).mean())
        # Several LTE bands can map to one ECI; retain the strongest evidence.
        trials = trials.groupby("eci_key", as_index=False)["trial_probability"].max()
        target[:, y, x] = poisson_binomial_at_least(trials["trial_probability"].to_numpy()).astype(np.float32)
        mask[:, y, x] = confidence
    return target, mask


def build_observation_signal_targets(
    labels: pd.DataFrame,
    lon_axis: np.ndarray,
    lat_axis: np.ndarray,
    *,
    signal_min_dbm: float = -135.0,
    signal_max_dbm: float = -40.0,
    mode: str = "weighted_mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize weighted LTE RSRP observations as a normalized signal target.

    The target is a single channel in ``[0, 1]``.  A weighted mean is used when
    several matched cells fall in one spatial bin; ``sample_count`` and the
    engineering-match ``label_weight`` jointly determine the reliability.
    This is deliberately separate from :func:`build_observation_targets` so
    existing probability-only datasets remain backward compatible.
    """
    required = {"grid_lon", "grid_lat", "rsrp_mean_dbm", "label_weight", "sample_count"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"missing label columns: {sorted(missing)}")
    if mode not in {"weighted_mean", "strongest"}:
        raise ValueError("mode must be 'weighted_mean' or 'strongest'")
    if not np.isfinite(signal_min_dbm) or not np.isfinite(signal_max_dbm) or signal_max_dbm <= signal_min_dbm:
        raise ValueError("signal range must be finite with max > min")
    height, width = len(lat_axis), len(lon_axis)
    target = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.float32)
    grouped = labels[labels["label_weight"] > 0].groupby(["grid_lon", "grid_lat"], sort=False)
    lon_index = {float(value): index for index, value in enumerate(lon_axis)}
    lat_index = {float(value): index for index, value in enumerate(lat_axis)}
    for (grid_lon, grid_lat), frame in grouped:
        x = lon_index.get(round(float(grid_lon), 9))
        y = lat_index.get(round(float(grid_lat), 9))
        if x is None or y is None:
            continue
        values = frame["rsrp_mean_dbm"].to_numpy(dtype=float)
        weights = frame["sample_count"].to_numpy(dtype=float) * frame["label_weight"].to_numpy(dtype=float)
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not np.any(valid):
            continue
        if mode == "strongest":
            # Strongest observed LTE cell; this does not identify a monitoring receiver.
            mean_dbm = float(np.max(values[valid]))
        else:
            mean_dbm = float(np.average(values[valid], weights=weights[valid]))
        target[y, x] = np.clip(
            (mean_dbm - signal_min_dbm) / (signal_max_dbm - signal_min_dbm), 0.0, 1.0
        )
        mask[y, x] = 1.0
    return target, mask


def gaussian_field(
    points: np.ndarray,
    centers: np.ndarray,
    *,
    scale_km: float,
) -> np.ndarray:
    """Compute a normalized max Gaussian field around geographic centers."""
    if scale_km <= 0:
        raise ValueError("scale_km must be positive")
    if len(centers) == 0:
        return np.zeros(len(points), dtype=np.float32)
    # Equirectangular approximation is sufficient for the small Nanchang area.
    lat0 = np.radians(points[:, 1:2])
    dx = (points[:, None, 0] - centers[None, :, 0]) * 111.32 * np.cos(lat0)
    dy = (points[:, None, 1] - centers[None, :, 1]) * 111.32
    distance = np.sqrt(dx * dx + dy * dy)
    return np.exp(-0.5 * (distance / scale_km) ** 2).max(axis=1).astype(np.float32)
