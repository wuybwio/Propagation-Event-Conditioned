"""Sparse, route-aware sampling for physical radio-map labels."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd


_METERS_PER_DEGREE_LATITUDE = 111_320.0


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"missing sample columns: {sorted(missing)}")


def _inclusive_axis(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("spacing must be positive")
    count = int(math.floor((stop - start) / step + 1e-9))
    values = start + np.arange(count + 1, dtype=float) * step
    if not np.isclose(values[-1], stop):
        values = np.append(values, stop)
    return np.round(values, decimals=10)


def sample_domain_grid(
    *,
    bounds: tuple[float, float, float, float],
    spacing_degrees: float,
    heights_m: Iterable[int],
) -> pd.DataFrame:
    """Sample a geographic bounding box with inclusive edges at each altitude."""
    min_lon, min_lat, max_lon, max_lat = bounds
    if min_lon > max_lon or min_lat > max_lat:
        raise ValueError("bounds must be ordered min_lon, min_lat, max_lon, max_lat")
    longitudes = _inclusive_axis(min_lon, max_lon, spacing_degrees)
    latitudes = _inclusive_axis(min_lat, max_lat, spacing_degrees)
    records = []
    for height_m in heights_m:
        for longitude in longitudes:
            for latitude in latitudes:
                records.append(
                    {
                        "longitude": longitude,
                        "latitude": latitude,
                        "height_m": int(height_m),
                        "sampling_source": "domain_grid",
                        "sampling_priority": 1.0,
                    }
                )
    return pd.DataFrame.from_records(records)


def _bearing_degrees(points: pd.DataFrame, index: int) -> float:
    if len(points) == 1:
        return 0.0
    neighbor = points.iloc[index + 1] if index < len(points) - 1 else points.iloc[index - 1]
    current = points.iloc[index]
    dx = (float(neighbor.longitude) - float(current.longitude)) * math.cos(
        math.radians(float(current.latitude))
    )
    dy = float(neighbor.latitude) - float(current.latitude)
    return math.degrees(math.atan2(dx, dy))


def _offset_location(
    longitude: float, latitude: float, bearing_deg: float, offset_m: float
) -> tuple[float, float]:
    # A positive offset means left of the direction of travel.
    lateral_bearing = math.radians(bearing_deg - 90.0)
    d_lat = offset_m * math.cos(lateral_bearing) / _METERS_PER_DEGREE_LATITUDE
    d_lon = offset_m * math.sin(lateral_bearing) / (
        _METERS_PER_DEGREE_LATITUDE * math.cos(math.radians(latitude))
    )
    return longitude + d_lon, latitude + d_lat


def _symmetric_offsets(offsets_m: Iterable[float]) -> list[float]:
    output: list[float] = []
    include_center = False
    for offset in offsets_m:
        magnitude = abs(float(offset))
        if magnitude == 0.0:
            include_center = True
        else:
            output.extend([magnitude, -magnitude])
    if include_center:
        output.append(0.0)
    return output


def sample_route_corridors(
    route_points: pd.DataFrame,
    *,
    heights_m: Iterable[int],
    lateral_offsets_m: Iterable[float] = (0.0, 500.0),
    route_priority: float = 2.0,
) -> pd.DataFrame:
    """Densify physical labels along a route and on both sides of its corridor."""
    _require_columns(route_points, {"longitude", "latitude", "route_id"})
    if route_points.empty:
        raise ValueError("route_points must not be empty")
    offsets = _symmetric_offsets(lateral_offsets_m)
    records = []
    for index in range(len(route_points)):
        point = route_points.iloc[index]
        bearing = _bearing_degrees(route_points, index)
        for offset_m in offsets:
            longitude, latitude = _offset_location(
                float(point.longitude), float(point.latitude), bearing, offset_m
            )
            for height_m in heights_m:
                records.append(
                    {
                        "longitude": longitude,
                        "latitude": latitude,
                        "height_m": int(height_m),
                        "route_id": point.route_id,
                        "lateral_offset_m": offset_m,
                        "sampling_source": "route_corridor",
                        "sampling_priority": float(route_priority),
                    }
                )
    return pd.DataFrame.from_records(records)


def _destination_point(
    longitude: float, latitude: float, bearing_deg: float, distance_m: float
) -> tuple[float, float]:
    angle = math.radians(bearing_deg)
    d_lat = distance_m * math.cos(angle) / _METERS_PER_DEGREE_LATITUDE
    d_lon = distance_m * math.sin(angle) / (
        _METERS_PER_DEGREE_LATITUDE * math.cos(math.radians(latitude))
    )
    return longitude + d_lon, latitude + d_lat


def sample_station_neighborhoods(
    stations: pd.DataFrame,
    *,
    radii_m: Iterable[float],
    bearings_deg: Iterable[float],
    heights_m: Iterable[int],
    sampling_priority: float = 1.5,
) -> pd.DataFrame:
    """Probe radial neighborhoods around existing or proposed monitoring stations."""
    _require_columns(stations, {"station_id", "longitude", "latitude"})
    records = []
    for station in stations.itertuples(index=False):
        for radius_m in radii_m:
            for bearing_deg in bearings_deg:
                longitude, latitude = _destination_point(
                    float(station.longitude),
                    float(station.latitude),
                    float(bearing_deg),
                    float(radius_m),
                )
                for height_m in heights_m:
                    records.append(
                        {
                            "longitude": longitude,
                            "latitude": latitude,
                            "height_m": int(height_m),
                            "station_id": str(station.station_id),
                            "radius_m": float(radius_m),
                            "bearing_deg": float(bearing_deg),
                            "sampling_source": "station_neighborhood",
                            "sampling_priority": float(sampling_priority),
                        }
                    )
    return pd.DataFrame.from_records(records)


def select_active_samples(
    candidates: pd.DataFrame,
    *,
    max_samples: int,
    dedupe_degrees: float = 0.01,
) -> pd.DataFrame:
    """Retain the most uncertain, spatially distinct points for another label pass."""
    _require_columns(candidates, {"longitude", "latitude", "height_m", "uncertainty"})
    if max_samples <= 0 or dedupe_degrees <= 0:
        raise ValueError("max_samples and dedupe_degrees must be positive")
    ranked = candidates.sort_values("uncertainty", ascending=False, kind="stable").copy()
    selected_indices: list[object] = []
    occupied: set[tuple[int, int, int]] = set()
    for index, row in ranked.iterrows():
        key = (
            int(math.floor(float(row.longitude) / dedupe_degrees)),
            int(math.floor(float(row.latitude) / dedupe_degrees)),
            int(row.height_m),
        )
        if key in occupied:
            continue
        occupied.add(key)
        selected_indices.append(index)
        if len(selected_indices) == max_samples:
            break
    selected = ranked.loc[selected_indices].reset_index(drop=True)
    selected["sampling_source"] = "active_uncertainty"
    selected["sampling_priority"] = selected["uncertainty"].astype(float)
    return selected
