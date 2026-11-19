from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EnvironmentConfig, RadioConfig, StationGradeProfile
from .detection import detection_probability
from .propagation import (
    combine_powers_dbm,
    detection_margin_db,
    free_space_path_loss_db,
    received_power_dbm,
)
from .routes import EARTH_RADIUS_KM, haversine_distance_km


def _station_antenna_heights(
    stations: pd.DataFrame,
    station_grades: dict[int, StationGradeProfile],
) -> np.ndarray:
    """Return measured station antenna heights when available.

    Engineering exports may provide ``antenna_height_m``.  Grade profiles are
    retained as a validated fallback for the compact station inventory.
    """
    grades = stations["grade"].to_numpy(dtype=int)
    fallback = np.asarray(
        [station_grades[int(grade)].antenna_height_m for grade in grades],
        dtype=float,
    )
    if "antenna_height_m" not in stations.columns:
        return fallback
    measured = pd.to_numeric(stations["antenna_height_m"], errors="coerce").to_numpy(float)
    valid = np.isfinite(measured) & (measured > 0.0)
    return np.where(valid, measured, fallback)


def _antenna_attenuation_db(
    route: np.ndarray,
    flight_height_m: float,
    stations: pd.DataFrame,
    station_grades: dict[int, StationGradeProfile],
    station_ground_elevation_m: np.ndarray | float = 0.0,
    target_ground_elevation_m: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Return optional horizontal/vertical antenna-pattern attenuation.

    Station files without antenna columns remain isotropic.  Supported
    columns are ``azimuth_deg`` (or ``beam_azimuth_deg``),
    ``mechanical_tilt_deg``, ``horizontal_beamwidth`` and
    ``vertical_beamwidth``; missing columns use conservative isotropic
    defaults.
    """
    n_points, n_stations = len(route), len(stations)
    attenuation = np.zeros((n_points, n_stations), dtype=float)
    azimuth_col = next(
        (name for name in ("azimuth_deg", "beam_azimuth_deg") if name in stations.columns),
        None,
    )
    horizontal_col = next(
        (name for name in ("horizontal_beamwidth_deg", "horizontal_beamwidth") if name in stations.columns),
        None,
    )
    vertical_col = next(
        (name for name in ("vertical_beamwidth_deg", "vertical_beamwidth") if name in stations.columns),
        None,
    )
    tilt_col = "mechanical_tilt_deg" if "mechanical_tilt_deg" in stations.columns else None
    if azimuth_col is None and horizontal_col is None and vertical_col is None and tilt_col is None:
        return attenuation

    def numeric_column(name: str | None, default: float) -> np.ndarray:
        if name is None:
            return np.full(n_stations, default, dtype=float)
        return pd.to_numeric(stations[name], errors="coerce").fillna(default).to_numpy(float)

    azimuth = numeric_column(azimuth_col, 0.0)
    hpbw = np.maximum(numeric_column(horizontal_col, 360.0), 1.0)
    vpbw = np.maximum(numeric_column(vertical_col, 180.0), 1.0)
    tilt = numeric_column(tilt_col, 0.0)
    station_lon = stations["longitude"].to_numpy(float)[None, :]
    station_lat = stations["latitude"].to_numpy(float)[None, :]
    dlon = np.radians(route[:, 0, None] - station_lon) * np.cos(np.radians((route[:, 1, None] + station_lat) / 2.0))
    dlat = np.radians(route[:, 1, None] - station_lat)
    bearing = (np.degrees(np.arctan2(dlon, dlat)) + 360.0) % 360.0
    horizontal_offset = np.abs((bearing - azimuth[None, :] + 180.0) % 360.0 - 180.0)
    horizontal_loss = np.minimum(12.0 * (horizontal_offset / hpbw[None, :]) ** 2, 25.0)

    antenna_height = _station_antenna_heights(stations, station_grades)
    station_ground = np.asarray(station_ground_elevation_m, dtype=float)
    target_ground = np.asarray(target_ground_elevation_m, dtype=float)
    if station_ground.ndim == 0:
        station_ground = np.full(n_stations, float(station_ground))
    if target_ground.ndim == 0:
        target_ground = np.full(n_points, float(target_ground))
    station_ground = np.asarray(station_ground, dtype=float)
    target_ground = np.asarray(target_ground, dtype=float)
    if station_ground.shape != (n_stations,) or target_ground.shape != (n_points,):
        raise ValueError("ground elevation arrays must align with points and stations")
    horizontal_m = np.hypot(dlon * 6_371_000.0, dlat * 6_371_000.0)
    elevation = np.degrees(np.arctan2(
        target_ground[:, None] + flight_height_m - station_ground[None, :] - antenna_height[None, :],
        np.maximum(horizontal_m, 1.0),
    ))
    vertical_offset = np.abs(elevation + tilt[None, :])
    vertical_loss = np.minimum(12.0 * (vertical_offset / vpbw[None, :]) ** 2, 30.0)
    return horizontal_loss + vertical_loss


def compute_route_station_probabilities(
    route_points: np.ndarray,
    flight_height_m: float,
    stations: pd.DataFrame,
    radio: RadioConfig,
    station_grades: dict[int, StationGradeProfile],
    additional_loss_db: float | np.ndarray = 0.0,
    environment: EnvironmentConfig | None = None,
    station_ground_elevation_m: np.ndarray | float = 0.0,
    target_ground_elevation_m: np.ndarray | float = 0.0,
) -> np.ndarray:
    route = np.asarray(route_points, dtype=float)
    if route.ndim != 2 or route.shape[1] < 2:
        raise ValueError("route_points must contain longitude and latitude")
    if flight_height_m <= 0:
        raise ValueError("flight height must be positive")
    required = {"longitude", "latitude", "grade"}
    if missing := required.difference(stations.columns):
        raise ValueError(f"missing station columns: {sorted(missing)}")

    longitudes = stations["longitude"].to_numpy(dtype=float)
    latitudes = stations["latitude"].to_numpy(dtype=float)
    grades = stations["grade"].to_numpy(dtype=int)
    if any(int(grade) not in station_grades for grade in grades):
        missing_grade = next(int(grade) for grade in grades if int(grade) not in station_grades)
        raise ValueError(f"missing station grade profile: {missing_grade}")

    point_loss = np.asarray(additional_loss_db, dtype=float)
    if point_loss.ndim == 0:
        point_loss = np.full(len(route), float(point_loss), dtype=float)
    elif point_loss.shape not in {(len(route),), (len(route), len(stations))}:
        raise ValueError("additional_loss_db must be scalar, one value per point, or point-by-station")
    if np.any(~np.isfinite(point_loss)) or np.any(point_loss < 0):
        raise ValueError("additional_loss_db must be finite and non-negative")
    if environment is not None:
        point_loss = point_loss + environment.loss_db(flight_height_m)

    # Vectorized haversine and propagation calculation. The former nested
    # Python loops made a 500-point route with 168 stations unnecessarily slow.
    lon_route = np.radians(route[:, 0])[:, None]
    lat_route = np.radians(route[:, 1])[:, None]
    lon_station = np.radians(longitudes)[None, :]
    lat_station = np.radians(latitudes)[None, :]
    delta_lon = lon_station - lon_route
    delta_lat = lat_station - lat_route
    haversine_a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_route) * np.cos(lat_station) * np.sin(delta_lon / 2.0) ** 2
    )
    horizontal_distance_m = 1000.0 * 2.0 * EARTH_RADIUS_KM * np.arctan2(
        np.sqrt(haversine_a), np.sqrt(np.maximum(0.0, 1.0 - haversine_a))
    )
    probabilities = np.empty((len(route), len(stations)), dtype=float)
    antenna_loss = _antenna_attenuation_db(
        route, flight_height_m, stations, station_grades,
        station_ground_elevation_m=station_ground_elevation_m,
        target_ground_elevation_m=target_ground_elevation_m,
    )
    antenna_height = _station_antenna_heights(stations, station_grades)
    station_ground = np.asarray(station_ground_elevation_m, dtype=float)
    target_ground = np.asarray(target_ground_elevation_m, dtype=float)
    if station_ground.ndim == 0:
        station_ground = np.full(len(stations), float(station_ground))
    if target_ground.ndim == 0:
        target_ground = np.full(len(route), float(target_ground))
    for grade, profile in station_grades.items():
        station_mask = grades == int(grade)
        if not np.any(station_mask):
            continue
        vertical_distance_m = np.abs(
            target_ground[:, None] + float(flight_height_m)
            - station_ground[None, station_mask]
            - antenna_height[station_mask][None, :]
        )
        # ``vertical_distance_m`` is already shaped ``[n_points, n_stations]``.
        # Adding a leading axis here would broadcast to ``[1, n_points, n_stations]``
        # and either fail assignment or silently produce an invalid path-loss tensor.
        distance_m = np.maximum(
            1.0, np.hypot(horizontal_distance_m[:, station_mask], vertical_distance_m),
        )
        distance_km = distance_m / 1000.0
        path_loss = 32.45 + 20.0 * np.log10(radio.frequency_mhz) + 20.0 * np.log10(distance_km)
        received = (
            radio.emitter_power_dbm
            + radio.emitter_gain_dbi
            + profile.receiver_gain_dbi
            - path_loss
            - antenna_loss[:, station_mask]
            - (point_loss[:, None] if point_loss.ndim == 1 else point_loss[:, station_mask])
        )
        noise_interference_dbm = combine_powers_dbm(
            radio.noise_power_dbm, radio.interference_power_dbm
        )
        threshold = max(
            profile.sensitivity_dbm,
            noise_interference_dbm + profile.required_sinr_db,
        )
        probabilities[:, station_mask] = detection_probability(
            received - threshold,
            radio.shadow_std_db,
        )
    return probabilities


def compute_station_received_power_dbm(
    route_points: np.ndarray,
    flight_height_m: float,
    stations: pd.DataFrame,
    radio: RadioConfig,
    station_grades: dict[int, StationGradeProfile],
    additional_loss_db: float | np.ndarray = 0.0,
    environment: EnvironmentConfig | None = None,
    station_ground_elevation_m: np.ndarray | float = 0.0,
    target_ground_elevation_m: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Return received power for every point and station before detection.

    Shape is ``[n_points, n_stations]``.  This is the per-station physical
    coverage layer used for station-specific maps; the legacy route function
    converts the same received powers into detection probabilities.
    """
    route = np.asarray(route_points, dtype=float)
    if route.ndim != 2 or route.shape[1] < 2:
        raise ValueError("route_points must contain longitude and latitude")
    if flight_height_m <= 0:
        raise ValueError("flight height must be positive")
    required = {"longitude", "latitude", "grade"}
    if missing := required.difference(stations.columns):
        raise ValueError(f"missing station columns: {sorted(missing)}")
    point_loss = np.asarray(additional_loss_db, dtype=float)
    if point_loss.ndim == 0:
        point_loss = np.full(len(route), float(point_loss), dtype=float)
    elif point_loss.shape not in {(len(route),), (len(route), len(stations))}:
        raise ValueError("additional_loss_db must be scalar, one value per point, or point-by-station")
    if np.any(~np.isfinite(point_loss)) or np.any(point_loss < 0):
        raise ValueError("additional_loss_db must be finite and non-negative")
    if environment is not None:
        point_loss = point_loss + environment.loss_db(flight_height_m)

    longitudes = stations["longitude"].to_numpy(dtype=float)
    latitudes = stations["latitude"].to_numpy(dtype=float)
    grades = stations["grade"].to_numpy(dtype=int)
    lon_route = np.radians(route[:, 0])[:, None]
    lat_route = np.radians(route[:, 1])[:, None]
    lon_station = np.radians(longitudes)[None, :]
    lat_station = np.radians(latitudes)[None, :]
    delta_lon = lon_station - lon_route
    delta_lat = lat_station - lat_route
    haversine_a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_route) * np.cos(lat_station) * np.sin(delta_lon / 2.0) ** 2
    )
    horizontal_distance_m = 1000.0 * 2.0 * EARTH_RADIUS_KM * np.arctan2(
        np.sqrt(haversine_a), np.sqrt(np.maximum(0.0, 1.0 - haversine_a))
    )
    received = np.empty((len(route), len(stations)), dtype=float)
    antenna_loss = _antenna_attenuation_db(
        route, flight_height_m, stations, station_grades,
        station_ground_elevation_m=station_ground_elevation_m,
        target_ground_elevation_m=target_ground_elevation_m,
    )
    antenna_height = _station_antenna_heights(stations, station_grades)
    station_ground = np.asarray(station_ground_elevation_m, dtype=float)
    target_ground = np.asarray(target_ground_elevation_m, dtype=float)
    if station_ground.ndim == 0:
        station_ground = np.full(len(stations), float(station_ground))
    if target_ground.ndim == 0:
        target_ground = np.full(len(route), float(target_ground))
    for grade, profile in station_grades.items():
        station_mask = grades == int(grade)
        if not np.any(station_mask):
            continue
        vertical_distance_m = np.abs(
            target_ground[:, None] + float(flight_height_m)
            - station_ground[None, station_mask]
            - antenna_height[station_mask][None, :]
        )
        distance_km = np.maximum(
            1.0,
            np.hypot(horizontal_distance_m[:, station_mask], vertical_distance_m),
        ) / 1000.0
        path_loss = 32.45 + 20.0 * np.log10(radio.frequency_mhz) + 20.0 * np.log10(distance_km)
        received[:, station_mask] = (
            radio.emitter_power_dbm
            + radio.emitter_gain_dbi
            + profile.receiver_gain_dbi
            - path_loss
            - antenna_loss[:, station_mask]
            - (point_loss[:, None] if point_loss.ndim == 1 else point_loss[:, station_mask])
        )
    return received


def received_power_to_station_probabilities(
    received_power_dbm: np.ndarray,
    flight_height_m: float,
    stations: pd.DataFrame,
    radio: RadioConfig,
    station_grades: dict[int, StationGradeProfile],
) -> np.ndarray:
    """Convert a received-power matrix into station detection probabilities."""
    from .detection import detection_probability
    values = np.asarray(received_power_dbm, dtype=float)
    grades = stations["grade"].to_numpy(dtype=int)
    noise_interference_dbm = combine_powers_dbm(radio.noise_power_dbm, radio.interference_power_dbm)
    result = np.empty_like(values)
    for grade, profile in station_grades.items():
        mask = grades == int(grade)
        if not np.any(mask):
            continue
        threshold = max(profile.sensitivity_dbm, noise_interference_dbm + profile.required_sinr_db)
        result[:, mask] = detection_probability(
            values[:, mask] - threshold, radio.shadow_std_db
        )
    return result
