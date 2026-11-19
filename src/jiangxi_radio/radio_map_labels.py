from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import ScenarioConfig
from .physics_baseline import compute_route_station_probabilities
from .radio_map_schema import validate_radio_map_samples


@dataclass(frozen=True)
class MultiHeightPhysicalPriors:
    """Physical prior maps arranged as [height, point, station]."""

    heights_m: np.ndarray
    probabilities: np.ndarray
    metadata: dict[str, Any]


def _validate_heights(heights_m: Sequence[float]) -> np.ndarray:
    heights = np.asarray(tuple(heights_m), dtype=float)
    if heights.ndim != 1 or len(heights) == 0:
        raise ValueError("heights must be a non-empty one-dimensional sequence")
    if not np.isfinite(heights).all() or np.any(heights <= 0):
        raise ValueError("height values must be finite and positive")
    return heights


def _validate_stations(stations: pd.DataFrame) -> pd.DataFrame:
    required = {"station_id", "longitude", "latitude", "grade"}
    missing = required.difference(stations.columns)
    if missing:
        raise ValueError(f"missing station columns: {sorted(missing)}")
    if stations.empty:
        raise ValueError("stations must not be empty")
    return stations.reset_index(drop=True)


def generate_multiheight_physical_priors(
    points: pd.DataFrame,
    stations: pd.DataFrame,
    config: ScenarioConfig,
    *,
    heights_m: Sequence[float] | None = None,
    additional_loss_db: float | np.ndarray = 0.0,
) -> MultiHeightPhysicalPriors:
    """Generate per-station physical priors for several flight heights."""

    samples = validate_radio_map_samples(
        points.assign(height_m=float(points["height_m"].iloc[0]) if "height_m" in points else 1.0,
                      station_id="map-sample")
        if "height_m" not in points.columns
        else points
    )
    station_frame = _validate_stations(stations)
    heights = _validate_heights(
        config.study.heights_m if heights_m is None else heights_m
    )
    coordinates = samples[["longitude", "latitude"]].to_numpy(dtype=float)
    probability_maps: list[np.ndarray] = []
    for height in heights:
        probabilities = compute_route_station_probabilities(
            coordinates,
            float(height),
            station_frame,
            config.radio,
            config.station_grades,
            additional_loss_db=additional_loss_db,
            environment=config.environment,
        )
        probability_maps.append(probabilities)
    return MultiHeightPhysicalPriors(
        heights_m=heights,
        probabilities=np.stack(probability_maps, axis=0),
        metadata={
            "label_evidence": "physics_scenario",
            "scenario_version": "scenario.toml",
            "height_count": int(len(heights)),
            "point_count": int(len(samples)),
            "station_count": int(len(station_frame)),
        },
    )


def priors_to_long_table(
    priors: MultiHeightPhysicalPriors,
    points: pd.DataFrame,
    stations: pd.DataFrame,
) -> pd.DataFrame:
    """Flatten multi-height priors into one auditable row per point/station."""

    samples = validate_radio_map_samples(
        points.assign(height_m=float(priors.heights_m[0]), station_id="map-sample")
        if "height_m" not in points.columns
        else points
    )
    station_frame = _validate_stations(stations)
    expected = (len(priors.heights_m), len(samples), len(station_frame))
    if priors.probabilities.shape != expected:
        raise ValueError(f"probability tensor must have shape {expected}")
    rows: list[dict[str, Any]] = []
    for height_index, height in enumerate(priors.heights_m):
        for point_index, point in samples.iterrows():
            for station_index, station in station_frame.iterrows():
                rows.append(
                    {
                        "longitude": float(point.longitude),
                        "latitude": float(point.latitude),
                        "height_m": float(height),
                        "station_id": str(station.station_id),
                        "detection_probability": float(priors.probabilities[height_index, point_index, station_index]),
                        "physical_prior_probability": float(priors.probabilities[height_index, point_index, station_index]),
                        "label_evidence": priors.metadata.get("label_evidence", "physics_scenario"),
                    }
                )
    return pd.DataFrame.from_records(rows)
