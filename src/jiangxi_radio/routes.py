from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


EARTH_RADIUS_KM = 6371.0088


def haversine_distance_km(
    longitude_a: float,
    latitude_a: float,
    longitude_b: float,
    latitude_b: float,
) -> float:
    lon1, lat1, lon2, lat2 = map(
        math.radians,
        (longitude_a, latitude_a, longitude_b, latitude_b),
    )
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1
    value = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2.0) ** 2
    )
    central_angle = 2.0 * math.atan2(math.sqrt(value), math.sqrt(1.0 - value))
    return EARTH_RADIUS_KM * central_angle


def _to_unit_vector(longitude: float, latitude: float) -> np.ndarray:
    lon = math.radians(longitude)
    lat = math.radians(latitude)
    return np.array(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)],
        dtype=float,
    )


def _from_unit_vector(vector: np.ndarray) -> tuple[float, float]:
    vector = vector / np.linalg.norm(vector)
    longitude = math.degrees(math.atan2(vector[1], vector[0]))
    latitude = math.degrees(math.atan2(vector[2], math.hypot(vector[0], vector[1])))
    return longitude, latitude


def _spherical_interpolate(
    start: tuple[float, float],
    end: tuple[float, float],
    fraction: float,
) -> tuple[float, float]:
    a = _to_unit_vector(*start)
    b = _to_unit_vector(*end)
    angle = math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0)))
    if angle < 1e-12:
        return start
    sin_angle = math.sin(angle)
    if abs(sin_angle) < 1e-12:
        vector = (1.0 - fraction) * a + fraction * b
    else:
        vector = (
            math.sin((1.0 - fraction) * angle) / sin_angle * a
            + math.sin(fraction * angle) / sin_angle * b
        )
    return _from_unit_vector(vector)


def sample_geodesic_route(
    waypoints_lonlat: Sequence[tuple[float, float]],
    spacing_km: float,
) -> np.ndarray:
    if len(waypoints_lonlat) < 2:
        raise ValueError("a route needs at least two waypoints")
    if spacing_km <= 0:
        raise ValueError("spacing_km must be positive")

    sampled: list[tuple[float, float, float]] = []
    cumulative_offset = 0.0
    for segment_index, (start, end) in enumerate(
        zip(waypoints_lonlat[:-1], waypoints_lonlat[1:])
    ):
        segment_distance = haversine_distance_km(*start, *end)
        if segment_distance == 0:
            continue
        segment_count = max(1, math.ceil(segment_distance / spacing_km))
        first_step = 0 if segment_index == 0 else 1
        for step in range(first_step, segment_count + 1):
            fraction = step / segment_count
            longitude, latitude = _spherical_interpolate(start, end, fraction)
            sampled.append(
                (longitude, latitude, cumulative_offset + fraction * segment_distance)
            )
        cumulative_offset += segment_distance

    if len(sampled) < 2:
        raise ValueError("route waypoints collapse to one location")
    return np.asarray(sampled, dtype=float)


def assign_endpoint_priorities(
    cumulative_distance_km: np.ndarray,
    high_risk_radius_km: float,
    critical_radius_km: float,
) -> np.ndarray:
    distances = np.asarray(cumulative_distance_km, dtype=float)
    if distances.ndim != 1 or len(distances) == 0:
        raise ValueError("cumulative distance must be a non-empty one-dimensional array")
    if high_risk_radius_km < 0 or critical_radius_km < high_risk_radius_km:
        raise ValueError("priority radii must be non-negative and ordered")
    distance_to_endpoint = np.minimum(distances - distances[0], distances[-1] - distances)
    result = np.full(len(distances), "ordinary", dtype=object)
    result[distance_to_endpoint <= critical_radius_km] = "critical"
    result[distance_to_endpoint <= high_risk_radius_km] = "high_risk"
    return result
