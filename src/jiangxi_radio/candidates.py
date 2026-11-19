from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .routes import EARTH_RADIUS_KM, haversine_distance_km


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, selected in enumerate(mask):
        if selected and start is None:
            start = index
        elif not selected and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _interpolate_route_point(route: np.ndarray, distance_km: float) -> tuple[float, float]:
    distances = route[:, 2]
    right = int(np.searchsorted(distances, distance_km, side="right"))
    if right == 0:
        return float(route[0, 0]), float(route[0, 1])
    if right >= len(route):
        return float(route[-1, 0]), float(route[-1, 1])
    left = right - 1
    span = distances[right] - distances[left]
    fraction = 0.0 if span == 0 else (distance_km - distances[left]) / span
    longitude = route[left, 0] + fraction * (route[right, 0] - route[left, 0])
    latitude = route[left, 1] + fraction * (route[right, 1] - route[left, 1])
    return float(longitude), float(latitude)


def _local_bearing_degrees(route: np.ndarray, distance_km: float) -> float:
    distances = route[:, 2]
    right = int(np.searchsorted(distances, distance_km, side="right"))
    left = max(0, min(right - 1, len(route) - 2))
    right = left + 1
    lon1, lat1 = map(math.radians, route[left, :2])
    lon2, lat2 = map(math.radians, route[right, :2])
    delta_lon = lon2 - lon1
    y = math.sin(delta_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    return math.degrees(math.atan2(y, x))


def _destination_point(
    longitude: float,
    latitude: float,
    bearing_degrees: float,
    distance_km: float,
) -> tuple[float, float]:
    if distance_km == 0:
        return longitude, latitude
    angular_distance = distance_km / EARTH_RADIUS_KM
    bearing = math.radians(bearing_degrees)
    lon1 = math.radians(longitude)
    lat1 = math.radians(latitude)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lon2), math.degrees(lat2)


def _anchor_distances(start_km: float, end_km: float, spacing_km: float) -> np.ndarray:
    if end_km <= start_km:
        return np.array([start_km], dtype=float)
    segment_count = max(1, math.ceil((end_km - start_km) / spacing_km))
    return np.linspace(start_km, end_km, segment_count + 1)


def generate_gap_candidates(
    route_points: np.ndarray,
    uncovered: np.ndarray,
    route_id: str,
    height_m: int,
    longitudinal_spacing_km: float,
    lateral_offsets_km: tuple[float, ...] = (-2.0, 0.0, 2.0),
) -> pd.DataFrame:
    route = np.asarray(route_points, dtype=float)
    gap_mask = np.asarray(uncovered, dtype=bool)
    if route.ndim != 2 or route.shape[1] < 3:
        raise ValueError("route_points must have longitude, latitude, and cumulative distance")
    if len(route) != len(gap_mask):
        raise ValueError("route points and uncovered mask must have equal length")
    if len(route) < 2 or np.any(np.diff(route[:, 2]) <= 0):
        raise ValueError("route cumulative distance must be strictly increasing")
    if longitudinal_spacing_km <= 0:
        raise ValueError("longitudinal spacing must be positive")

    rows: list[dict[str, object]] = []
    for gap_number, (start_index, end_index) in enumerate(
        _contiguous_true_runs(gap_mask), start=1
    ):
        gap_id = f"{route_id}-gap-{gap_number:03d}"
        start_km = float(route[start_index, 2])
        end_km = float(route[end_index, 2])
        anchors = _anchor_distances(start_km, end_km, longitudinal_spacing_km)
        for anchor_number, anchor_km in enumerate(anchors, start=1):
            center_lon, center_lat = _interpolate_route_point(route, float(anchor_km))
            bearing = _local_bearing_degrees(route, float(anchor_km))
            for lateral_offset in lateral_offsets_km:
                side_bearing = bearing + (90.0 if lateral_offset >= 0 else -90.0)
                longitude, latitude = _destination_point(
                    center_lon,
                    center_lat,
                    side_bearing,
                    abs(float(lateral_offset)),
                )
                rows.append(
                    {
                        "candidate_id": (
                            f"{gap_id}-a{anchor_number:03d}-"
                            f"o{float(lateral_offset):+05.1f}"
                        ),
                        "route_id": route_id,
                        "gap_id": gap_id,
                        "height_m": int(height_m),
                        "route_distance_km": float(anchor_km),
                        "lateral_offset_km": float(lateral_offset),
                        "longitude": longitude,
                        "latitude": latitude,
                    }
                )
    columns = [
        "candidate_id",
        "route_id",
        "gap_id",
        "height_m",
        "route_distance_km",
        "lateral_offset_km",
        "longitude",
        "latitude",
    ]
    return pd.DataFrame.from_records(rows, columns=columns)


def filter_candidate_locations(
    candidates: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    existing_stations: pd.DataFrame,
    minimum_station_distance_km: float,
    minimum_candidate_distance_km: float = 0.0,
) -> pd.DataFrame:
    required_candidate_columns = {"candidate_id", "longitude", "latitude"}
    if missing := required_candidate_columns.difference(candidates.columns):
        raise ValueError(f"missing candidate columns: {sorted(missing)}")
    if minimum_station_distance_km < 0:
        raise ValueError("minimum station distance must be non-negative")
    if minimum_candidate_distance_km < 0:
        raise ValueError("minimum candidate distance must be non-negative")
    min_lon, min_lat, max_lon, max_lat = bounds
    if min_lon > max_lon or min_lat > max_lat:
        raise ValueError("invalid geographic bounds")

    result = candidates.copy()
    inside = (
        result["longitude"].between(min_lon, max_lon)
        & result["latitude"].between(min_lat, max_lat)
    )
    result = result.loc[inside].copy()

    if len(existing_stations) and len(result):
        required_station_columns = {"longitude", "latitude"}
        if missing := required_station_columns.difference(existing_stations.columns):
            raise ValueError(f"missing existing station columns: {sorted(missing)}")
        keep: list[bool] = []
        for candidate in result.itertuples(index=False):
            nearest = min(
                haversine_distance_km(
                    candidate.longitude,
                    candidate.latitude,
                    station.longitude,
                    station.latitude,
                )
                for station in existing_stations.itertuples(index=False)
            )
            keep.append(nearest >= minimum_station_distance_km)
        result = result.loc[keep]
    if minimum_candidate_distance_km > 0 and len(result) > 1:
        # Keep the first candidate in generation order, then suppress nearby
        # candidates.  Candidate generation is ordered along each uncovered
        # route gap, so this is deterministic and avoids dense clusters.
        selected: list[tuple[float, float]] = []
        keep: list[bool] = []
        for candidate in result.itertuples(index=False):
            far_enough = all(
                haversine_distance_km(
                    float(candidate.longitude),
                    float(candidate.latitude),
                    longitude,
                    latitude,
                ) >= minimum_candidate_distance_km
                for longitude, latitude in selected
            )
            keep.append(far_enough)
            if far_enough:
                selected.append((float(candidate.longitude), float(candidate.latitude)))
        result = result.loc[keep]
    return result.reset_index(drop=True)
