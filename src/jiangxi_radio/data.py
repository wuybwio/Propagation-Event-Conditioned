from __future__ import annotations

import numpy as np
import pandas as pd


_GRADE_ALIASES = {
    "2": 2,
    "II": 2,
    "二类": 2,
    "3": 3,
    "III": 3,
    "三类": 3,
    "4": 4,
    "IV": 4,
    "四类": 4,
}


def _normalize_grade(value: object) -> int:
    if isinstance(value, (int, np.integer)):
        key = str(int(value))
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        key = str(int(value))
    else:
        key = str(value).strip().upper()
    try:
        return _GRADE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"unsupported station grade: {value!r}") from exc


def normalize_stations(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"station_id", "longitude", "latitude", "grade"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing station columns: {sorted(missing)}")

    # Preserve optional engineering metadata (antenna height, azimuth, beamwidth,
    # tilt, power, frequency, etc.) so the physical propagation layer can use
    # measured station parameters when they are available.  The previous
    # implementation silently discarded these columns, forcing every station
    # to fall back to grade-level defaults and weakening 3-D realism.
    result = frame.copy()
    result["station_id"] = result["station_id"].astype(str).str.strip()
    result["longitude"] = pd.to_numeric(result["longitude"], errors="coerce")
    result["latitude"] = pd.to_numeric(result["latitude"], errors="coerce")
    result["grade"] = result["grade"].map(_normalize_grade)
    for column in (
        "antenna_height_m", "azimuth_deg", "beam_azimuth_deg",
        "horizontal_beamwidth", "horizontal_beamwidth_deg",
        "vertical_beamwidth", "vertical_beamwidth_deg",
        "mechanical_tilt_deg", "electrical_tilt_deg",
        "max_power_w", "frequency_mhz",
    ):
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column].astype(str).str.replace("°", "", regex=False),
                errors="coerce",
            )
    result = result.drop_duplicates().reset_index(drop=True)

    valid_coordinates = (
        np.isfinite(result["longitude"])
        & np.isfinite(result["latitude"])
        & result["longitude"].between(-180, 180)
        & result["latitude"].between(-90, 90)
    )
    if not bool(valid_coordinates.all()):
        raise ValueError("invalid station coordinate")
    if (result["station_id"] == "").any():
        raise ValueError("station_id must not be empty")
    if result["station_id"].duplicated().any():
        raise ValueError("station_id maps to conflicting station records")
    return result


def normalize_routes(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"route_name", "origin", "destination"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing route columns: {sorted(missing)}")

    result = frame.copy()
    for column in required:
        result[column] = result[column].astype(str).str.strip()
    disallowed = (
        result["route_name"].str.contains("PC-12", case=False, regex=False)
        & (result["origin"] == "南昌")
        & (result["destination"] == "景德镇")
    )
    result = result.loc[~disallowed].reset_index(drop=True)
    result.insert(0, "route_id", [f"route-{i:03d}" for i in range(1, len(result) + 1)])
    return result
