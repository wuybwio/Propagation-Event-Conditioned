"""Data contracts for the standalone P²-RMNet 3-D coverage viewer.

The viewer consumes model output, but does not perform model inference.  This
module therefore keeps validation and lightweight geographic serialization in
one place so that the HTML generator and static exports use identical inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


_REQUIRED_FIELDS = (
    "longitude",
    "latitude",
    "heights_m",
    "station_id",
    "station_longitude",
    "station_latitude",
    "detection_probability",
)


def _as_axis(value: Any, name: str) -> np.ndarray:
    axis = np.asarray(value, dtype=float)
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(axis).all():
        raise ValueError(f"{name} must contain finite values")
    return axis


def _as_coordinates(value: Any, name: str, count: int) -> np.ndarray:
    coordinates = np.asarray(value, dtype=float)
    if coordinates.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{name} must contain finite values")
    return coordinates


def load_coverage_npz(path: str | Path) -> dict[str, Any]:
    """Load and validate a ``station_3d_coverage.npz`` result.

    The required probability tensor has shape ``(height, row, col, station)``.
    Values outside the probability interval are clipped for compatibility with
    older physical baselines; NaN and infinite values are rejected because
    silently rendering them produces misleading empty coverage layers.
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"coverage NPZ does not exist: {source}")
    with np.load(source, allow_pickle=False) as archive:
        missing = [name for name in _REQUIRED_FIELDS if name not in archive.files]
        if missing:
            raise ValueError(f"coverage NPZ is missing required fields: {', '.join(missing)}")

        longitude = _as_axis(archive["longitude"], "longitude")
        latitude = _as_axis(archive["latitude"], "latitude")
        heights_m = _as_axis(archive["heights_m"], "heights_m")
        probability = np.asarray(archive["detection_probability"], dtype=float)
        expected_shape = (len(heights_m), len(latitude), len(longitude))
        if probability.ndim != 4:
            raise ValueError(
                "detection_probability must have shape (height, row, col, station)"
            )
        if probability.shape[:3] != expected_shape:
            raise ValueError(
                "detection_probability grid shape does not match "
                f"heights/latitude/longitude: expected prefix {expected_shape}, "
                f"got {probability.shape[:3]}"
            )
        if not np.isfinite(probability).all():
            raise ValueError("detection_probability must not contain NaN or infinite values")
        probability = np.clip(probability, 0.0, 1.0).astype(np.float32)

        station_count = probability.shape[3]
        if station_count <= 0:
            raise ValueError("detection_probability must contain at least one station")
        station_ids = np.asarray(archive["station_id"]).astype(str)
        if station_ids.ndim != 1 or len(station_ids) != station_count:
            raise ValueError(
                f"station_id must have shape ({station_count},) aligned with probability"
            )
        if any(not station_id.strip() for station_id in station_ids):
            raise ValueError("station_id must not contain empty values")
        if len(set(station_ids.tolist())) != station_count:
            raise ValueError("station_id values must be unique")
        station_longitude = _as_coordinates(
            archive["station_longitude"], "station_longitude", station_count
        )
        station_latitude = _as_coordinates(
            archive["station_latitude"], "station_latitude", station_count
        )

        result: dict[str, Any] = {
            "longitude": longitude,
            "latitude": latitude,
            "heights_m": heights_m,
            "station_id": station_ids,
            "station_longitude": station_longitude,
            "station_latitude": station_latitude,
            "detection_probability": probability,
            "raw_grid_shape": (len(latitude), len(longitude)),
            "display_grid_shape": (len(latitude), len(longitude)),
            "station_count": station_count,
            "source_path": str(source),
        }

        # Preserve optional uncertainty and received-power arrays for later
        # layers, while validating only arrays whose dimensions are known.
        for name in archive.files:
            if name in result or name in _REQUIRED_FIELDS:
                continue
            value = np.asarray(archive[name])
            if name.lower() in {"uncertainty", "std", "probability_std", "detection_probability_std"}:
                if value.shape not in {probability.shape, probability.shape[:3]}:
                    raise ValueError(f"{name} shape must align with the probability grid")
                if not np.isfinite(value).all():
                    raise ValueError(f"{name} must not contain NaN or infinite values")
            result[name] = value
        return result


def compute_at_least_k(probability: np.ndarray, k: int) -> np.ndarray:
    """Compute ``P(N >= k)`` across the station axis.

    Stations are treated as conditionally independent detection events, which
    is the same aggregation used by the planning metrics.  The returned array
    has shape ``(height, row, col)``.
    """

    values = np.asarray(probability, dtype=float)
    if values.ndim != 4:
        raise ValueError("probability must have shape (height, row, col, station)")
    if not np.isfinite(values).all():
        raise ValueError("probability must not contain NaN or infinite values")
    values = np.clip(values, 0.0, 1.0)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer station count")
    k = int(k)
    station_count = values.shape[-1]
    if not 1 <= k <= station_count:
        raise ValueError(f"k must be between 1 and station count ({station_count})")

    # ``exact[..., j]`` stores the probability of exactly j detections after
    # each station is folded into the dynamic program.  Only j < k is needed.
    exact = np.zeros(values.shape[:-1] + (k,), dtype=float)
    exact[..., 0] = 1.0
    for index in range(station_count):
        p = values[..., index]
        updated = np.zeros_like(exact)
        updated[..., 0] = exact[..., 0] * (1.0 - p)
        for count in range(1, k):
            updated[..., count] = (
                exact[..., count] * (1.0 - p) + exact[..., count - 1] * p
            )
        exact = updated
    return np.clip(1.0 - exact.sum(axis=-1), 0.0, 1.0).astype(np.float32)


def _axis_edges(axis: np.ndarray) -> np.ndarray:
    values = np.asarray(axis, dtype=float)
    if len(values) == 1:
        half_step = 0.0005
        return np.array([values[0] - half_step, values[0] + half_step])
    edges = np.empty(len(values) + 1, dtype=float)
    edges[1:-1] = (values[:-1] + values[1:]) / 2.0
    edges[0] = values[0] - (edges[1] - values[0])
    edges[-1] = values[-1] + (values[-1] - edges[-2])
    return edges


def grid_cells(
    longitude: np.ndarray,
    latitude: np.ndarray,
    values: np.ndarray,
    *,
    threshold: float = 0.5,
    source_longitude: np.ndarray | None = None,
    source_latitude: np.ndarray | None = None,
    source_col_indices: np.ndarray | None = None,
    source_row_indices: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Convert a 2-D probability grid into lightweight cell records.

    Records retain row/column indices and cell bounds, allowing a generator to
    emit GeoJSON or Cesium rectangles without embedding the full tensor.
    """

    lon = _as_axis(longitude, "longitude")
    lat = _as_axis(latitude, "latitude")
    grid = np.asarray(values, dtype=float)
    if grid.shape != (len(lat), len(lon)):
        raise ValueError(
            f"values must have shape ({len(lat)}, {len(lon)}), got {grid.shape}"
        )
    if not np.isfinite(grid).all():
        raise ValueError("values must not contain NaN or infinite values")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    grid = np.clip(grid, 0.0, 1.0)
    source_lon = lon if source_longitude is None else _as_axis(source_longitude, "source_longitude")
    source_lat = lat if source_latitude is None else _as_axis(source_latitude, "source_latitude")
    col_indices = np.arange(len(lon), dtype=int) if source_col_indices is None else np.asarray(source_col_indices, dtype=int)
    row_indices = np.arange(len(lat), dtype=int) if source_row_indices is None else np.asarray(source_row_indices, dtype=int)
    if len(col_indices) != len(lon) or len(row_indices) != len(lat):
        raise ValueError("source index arrays must align with display axes")
    if np.any(col_indices < 0) or np.any(col_indices >= len(source_lon)):
        raise ValueError("source_col_indices are outside source_longitude")
    if np.any(row_indices < 0) or np.any(row_indices >= len(source_lat)):
        raise ValueError("source_row_indices are outside source_latitude")
    lon_edges = _axis_edges(source_lon)
    lat_edges = _axis_edges(source_lat)
    cells: list[dict[str, Any]] = []
    for row in range(len(lat)):
        for col in range(len(lon)):
            source_row = int(row_indices[row])
            source_col = int(col_indices[col])
            value = float(grid[row, col])
            cells.append(
                {
                    "row": source_row,
                    "col": source_col,
                    "longitude": float(lon[col]),
                    "latitude": float(lat[row]),
                    "west": float(min(lon_edges[source_col], lon_edges[source_col + 1])),
                    "east": float(max(lon_edges[source_col], lon_edges[source_col + 1])),
                    "south": float(min(lat_edges[source_row], lat_edges[source_row + 1])),
                    "north": float(max(lat_edges[source_row], lat_edges[source_row + 1])),
                    "value": value,
                    "selected": bool(value >= float(threshold)),
                }
            )
    return cells


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, set)):
        return [_json_value(item) for item in value]
    return value


def build_manifest(
    heights_m: np.ndarray,
    station_ids: np.ndarray,
    raw_grid_shape: tuple[int, int] | list[int],
    display_grid_shape: tuple[int, int] | list[int],
    layers: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    **extra_metadata: Any,
) -> dict[str, Any]:
    """Build a JSON-safe manifest shared by data layers and the HTML viewer."""

    protected_fields = {
        "schema_version",
        "heights_m",
        "station_count",
        "station_ids",
        "raw_grid_shape",
        "display_grid_shape",
        "layers",
    }
    supplied_metadata = dict(metadata or {})
    supplied_metadata.update(extra_metadata)
    conflicts = sorted(protected_fields.intersection(supplied_metadata))
    if conflicts:
        raise ValueError(
            "metadata must not override validated manifest fields: "
            + ", ".join(conflicts)
        )

    heights = np.asarray(heights_m, dtype=float)
    stations = np.asarray(station_ids).astype(str)
    if heights.ndim != 1 or len(heights) == 0 or not np.isfinite(heights).all():
        raise ValueError("heights_m must be a non-empty finite one-dimensional array")
    if stations.ndim != 1 or len(stations) == 0 or any(not item.strip() for item in stations):
        raise ValueError("station_ids must be a non-empty one-dimensional array")
    if len(set(stations.tolist())) != len(stations):
        raise ValueError("station_ids must be unique")
    raw_shape = tuple(int(value) for value in raw_grid_shape)
    display_shape = tuple(int(value) for value in display_grid_shape)
    if len(raw_shape) != 2 or any(value <= 0 for value in raw_shape):
        raise ValueError("raw_grid_shape must contain two positive dimensions")
    if len(display_shape) != 2 or any(value <= 0 for value in display_shape):
        raise ValueError("display_grid_shape must contain two positive dimensions")

    manifest: dict[str, Any] = {
        "schema_version": "p2rmnet-3d-coverage/v1",
        "heights_m": [float(value) for value in heights],
        "station_count": int(len(stations)),
        "station_ids": stations.tolist(),
        "raw_grid_shape": list(raw_shape),
        "display_grid_shape": list(display_shape),
        "layers": dict(layers or {}),
    }
    manifest.update(supplied_metadata)
    return _json_value(manifest)
