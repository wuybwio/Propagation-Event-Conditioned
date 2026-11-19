from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


REQUIRED_SAMPLE_COLUMNS = frozenset({"longitude", "latitude", "height_m", "station_id"})


@dataclass(frozen=True)
class RadioMapBatch:
    """Aligned tensors used by the probabilistic 3-D radio-map model."""

    environment: np.ndarray
    physical_prior: np.ndarray
    station_features: np.ndarray
    heights_m: np.ndarray
    metadata: Mapping[str, Any]


def validate_radio_map_samples(samples: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(samples, pd.DataFrame):
        raise TypeError("samples must be a pandas DataFrame")
    missing = REQUIRED_SAMPLE_COLUMNS.difference(samples.columns)
    if missing:
        raise ValueError(f"missing radio-map sample columns: {sorted(missing)}")
    result = samples.copy()
    for column in ("longitude", "latitude", "height_m"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["longitude", "latitude", "height_m"]].isna().any().any():
        raise ValueError("radio-map coordinates and height must be finite numbers")
    if not np.isfinite(result[["longitude", "latitude", "height_m"]].to_numpy(dtype=float)).all():
        raise ValueError("radio-map coordinates and height must be finite numbers")
    if ((result["longitude"] < -180) | (result["longitude"] > 180)).any():
        raise ValueError("longitude must be within [-180, 180]")
    if ((result["latitude"] < -90) | (result["latitude"] > 90)).any():
        raise ValueError("latitude must be within [-90, 90]")
    if (result["height_m"] <= 0).any():
        raise ValueError("height_m must be positive")
    if result["station_id"].isna().any() or (result["station_id"].astype(str).str.strip() == "").any():
        raise ValueError("station_id must be non-empty")
    return result


def validate_radio_map_batch(batch: RadioMapBatch) -> RadioMapBatch:
    environment = np.asarray(batch.environment)
    physical_prior = np.asarray(batch.physical_prior)
    station_features = np.asarray(batch.station_features)
    heights_m = np.asarray(batch.heights_m)
    if environment.ndim != 4:
        raise ValueError("environment must have shape [batch, channels, height, width]")
    if physical_prior.ndim != 4:
        raise ValueError("physical_prior must have shape [batch, channels, height, width]")
    if station_features.ndim != 2:
        raise ValueError("station_features must have shape [batch, features]")
    if heights_m.ndim != 1:
        raise ValueError("heights_m must have shape [batch]")
    batch_size = environment.shape[0]
    if any(array.shape[0] != batch_size for array in (physical_prior, station_features, heights_m)):
        raise ValueError("all radio-map tensors must share the same batch dimension")
    if physical_prior.shape[2:] != environment.shape[2:]:
        raise ValueError("physical_prior and environment must share spatial dimensions")
    if np.any(~np.isfinite(environment)) or np.any(~np.isfinite(physical_prior)):
        raise ValueError("radio-map tensors must contain finite values")
    if np.any(~np.isfinite(station_features)) or np.any(~np.isfinite(heights_m)):
        raise ValueError("radio-map metadata tensors must contain finite values")
    if np.any(heights_m <= 0):
        raise ValueError("heights_m must be positive")
    return RadioMapBatch(environment, physical_prior, station_features, heights_m, dict(batch.metadata))
