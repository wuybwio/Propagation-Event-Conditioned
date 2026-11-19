"""Shared, testable helpers for publication radio-coverage figures."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _polygon_exteriors(geometry: Mapping[str, object]) -> list[np.ndarray]:
    geometry_type = str(geometry.get("type", ""))
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list):
        return []
    if geometry_type == "Polygon":
        polygons = [coordinates]
    elif geometry_type == "MultiPolygon":
        polygons = coordinates
    else:
        return []
    rings: list[np.ndarray] = []
    for polygon in polygons:
        if not isinstance(polygon, list) or not polygon:
            continue
        ring = np.asarray(polygon[0], dtype=float)
        if ring.ndim != 2 or ring.shape[1] < 2 or len(ring) < 4:
            continue
        ring = ring[:, :2]
        if np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        if len(ring) >= 3 and np.isfinite(ring).all():
            rings.append(ring)
    return rings


def building_prisms_from_geojson(
    collection: Mapping[str, object],
    *,
    center_longitude: float,
    center_latitude: float,
    size_m: float,
    x_m: np.ndarray,
    y_m: np.ndarray,
    terrain_m: np.ndarray,
    default_height_m: float = 8.0,
) -> list[dict[str, object]]:
    """Convert WGS84 building polygons to independent local 3-D prisms."""
    features = collection.get("features")
    if not isinstance(features, list):
        raise ValueError("building GeoJSON must be a FeatureCollection")
    x_axis = np.asarray(x_m, dtype=float)
    y_axis = np.asarray(y_m, dtype=float)
    terrain = np.asarray(terrain_m, dtype=float)
    if terrain.shape != (len(y_axis), len(x_axis)):
        raise ValueError("terrain shape must align with y_m and x_m")
    if size_m <= 0.0 or not np.isfinite(size_m):
        raise ValueError("local scene size must be finite and positive")
    metres_per_lon = 111_320.0 * np.cos(np.radians(float(center_latitude)))
    metres_per_lat = 111_320.0
    offset = 0.5 * float(size_m)
    prisms: list[dict[str, object]] = []
    for feature_index, feature in enumerate(features):
        if not isinstance(feature, Mapping):
            continue
        properties = feature.get("properties")
        if not isinstance(properties, Mapping):
            properties = {}
        raw_height = properties.get("height", properties.get("building_height"))
        try:
            height = float(raw_height)
        except (TypeError, ValueError):
            levels = properties.get("levels", properties.get("num_floors"))
            try:
                height = 3.0 * float(levels)
            except (TypeError, ValueError):
                height = float(default_height_m)
        if not np.isfinite(height) or height < 2.0:
            continue
        geometry = feature.get("geometry")
        if not isinstance(geometry, Mapping):
            continue
        for part_index, ring in enumerate(_polygon_exteriors(geometry)):
            local = np.column_stack((
                offset + (ring[:, 0] - float(center_longitude)) * metres_per_lon,
                offset + (ring[:, 1] - float(center_latitude)) * metres_per_lat,
            ))
            if (
                local[:, 0].max() < x_axis.min()
                or local[:, 0].min() > x_axis.max()
                or local[:, 1].max() < y_axis.min()
                or local[:, 1].min() > y_axis.max()
            ):
                continue
            centroid = local.mean(axis=0)
            xi = nearest_index(x_axis, centroid[0])
            yi = nearest_index(y_axis, centroid[1])
            base = float(terrain[yi, xi])
            prisms.append({
                "building_id": str(properties.get("id", feature_index)),
                "part_index": int(part_index),
                "xy_m": local,
                "base_z_m": base,
                "top_z_m": base + height,
                "height_m": height,
            })
    return prisms


def publication_rcparams() -> dict[str, object]:
    """Return journal-oriented Matplotlib defaults with stable page layout."""
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.titleweight": "normal",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 180,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.08,
        "figure.constrained_layout.w_pad": 0.08,
        "savefig.dpi": 300,
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.08,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def activate_publication_style() -> str:
    """Require Times New Roman and activate the shared journal style.

    Matplotlib normally falls back silently when a requested font is absent.
    That behavior is unsuitable for final paper figures, so this function
    fails early with WSL-specific remediation instructions instead.
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidates = [
        Path("/mnt/c/Windows/Fonts"),
        Path("C:/Windows/Fonts"),
    ]
    for directory in candidates:
        if directory.is_dir():
            for font_file in sorted(directory.glob("times*.ttf")):
                font_manager.fontManager.addfont(str(font_file))
    try:
        font_path = font_manager.findfont(
            font_manager.FontProperties(family="Times New Roman"),
            fallback_to_default=False,
        )
    except ValueError as error:
        raise RuntimeError(
            "Times New Roman is unavailable. The plotter checked Matplotlib, "
            "/mnt/c/Windows/Fonts and C:/Windows/Fonts. In WSL, verify that "
            "/mnt/c/Windows/Fonts/times.ttf is readable."
        ) from error
    plt.rcParams.update(publication_rcparams())
    return str(font_path)


def validate_publication_layout(figure, *, tolerance_px: float = 3.0) -> None:
    """Reject clipped labels or overlapping panel titles before export."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvas = figure.bbox
    labelled_artists: list[tuple[str, object]] = []
    title_artists: list[tuple[str, object]] = []

    for index, artist in enumerate(figure.texts):
        if artist.get_visible() and artist.get_text().strip():
            labelled_artists.append((f"figure text {index}", artist))
    for index, axis in enumerate(figure.axes):
        candidates = [
            (f"axis {index} title", axis.title),
            (f"axis {index} x-label", axis.xaxis.label),
            (f"axis {index} y-label", axis.yaxis.label),
        ]
        if hasattr(axis, "zaxis"):
            candidates.append((f"axis {index} z-label", axis.zaxis.label))
        for name, artist in candidates:
            if artist.get_visible() and artist.get_text().strip():
                labelled_artists.append((name, artist))
                if name.endswith("title"):
                    title_artists.append((name, artist))
        legend = axis.get_legend()
        if legend is not None and legend.get_visible():
            labelled_artists.append((f"axis {index} legend", legend))

    issues: list[str] = []
    for name, artist in labelled_artists:
        bounds = artist.get_window_extent(renderer=renderer)
        if (
            bounds.x0 < canvas.x0 - tolerance_px
            or bounds.y0 < canvas.y0 - tolerance_px
            or bounds.x1 > canvas.x1 + tolerance_px
            or bounds.y1 > canvas.y1 + tolerance_px
        ):
            issues.append(f"{name} is clipped by the figure boundary")

    for left_index, (left_name, left_artist) in enumerate(title_artists):
        left = left_artist.get_window_extent(renderer=renderer)
        for right_name, right_artist in title_artists[left_index + 1:]:
            right = right_artist.get_window_extent(renderer=renderer)
            if left.overlaps(right):
                issues.append(f"{left_name} overlaps {right_name}")

    if issues:
        raise RuntimeError("publication layout validation failed: " + "; ".join(issues))


def save_publication_figure(figure, stem) -> list[str]:
    """Validate and export a figure as journal-ready PNG and vector PDF."""
    from pathlib import Path

    output = Path(stem)
    output.parent.mkdir(parents=True, exist_ok=True)
    validate_publication_layout(figure)
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    figure.savefig(png, dpi=300, facecolor="white")
    figure.savefig(pdf, facecolor="white")
    return [str(png), str(pdf)]


def nearest_index(values: np.ndarray, requested: float) -> int:
    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or axis.size == 0 or not np.isfinite(axis).all():
        raise ValueError("coordinate axis must be a non-empty finite vector")
    return int(np.argmin(np.abs(axis - float(requested))))


def building_display_axis(
    maximum_height_m: float,
    vertical_step_m: float,
    *,
    signal_heights_m: np.ndarray | None = None,
) -> np.ndarray:
    """Build an independent building-display axis starting at ground level.

    ``signal_heights_m`` is accepted only to make the independence explicit in
    callers and tests. It never clips or offsets the geometry axis.
    """
    maximum = float(maximum_height_m)
    step = float(vertical_step_m)
    if not np.isfinite(maximum) or maximum < 0.0:
        raise ValueError("maximum building height must be finite and non-negative")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("vertical display step must be finite and positive")
    if signal_heights_m is not None:
        signal = np.asarray(signal_heights_m, dtype=float)
        if signal.ndim != 1 or not np.isfinite(signal).all():
            raise ValueError("signal heights must be a finite vector")
    if maximum == 0.0:
        return np.asarray([0.0], dtype=float)
    levels = np.arange(0.0, maximum + 0.5 * step, step, dtype=float)
    levels = levels[levels < maximum]
    return np.concatenate((levels, np.asarray([maximum], dtype=float)))


def _support_lengths(distance_km: np.ndarray) -> np.ndarray:
    distance = np.asarray(distance_km, dtype=float)
    if distance.ndim != 1 or distance.size == 0 or not np.isfinite(distance).all():
        raise ValueError("route distance must be a non-empty finite vector")
    if distance.size == 1:
        return np.ones(1, dtype=float)
    intervals = np.diff(distance)
    if np.any(intervals <= 0.0):
        raise ValueError("distance_km must increase within each route")
    weights = np.empty_like(distance)
    weights[0] = intervals[0] / 2.0
    weights[-1] = intervals[-1] / 2.0
    if distance.size > 2:
        weights[1:-1] = (intervals[:-1] + intervals[1:]) / 2.0
    return weights


def audit_planning_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute discrete and route-length coverage from point-level output."""
    required = {
        "route_id",
        "height_m",
        "point_index",
        "distance_km",
        "longitude",
        "latitude",
        "baseline_probability",
        "optimized_probability",
        "probability_threshold",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"planning coverage is missing columns: {missing}")
    if frame.empty:
        raise ValueError("planning coverage is empty")
    values = frame.copy().reset_index(drop=True)
    numeric_columns = [
        "height_m",
        "point_index",
        "distance_km",
        "longitude",
        "latitude",
        "baseline_probability",
        "optimized_probability",
        "probability_threshold",
    ]
    numeric = values[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("planning coverage contains non-finite numeric values")
    values[numeric_columns] = numeric
    heights = values["height_m"].unique()
    if len(heights) != 1:
        raise ValueError("planning coverage must contain exactly one height")
    thresholds = values["probability_threshold"].to_numpy(float)
    if np.any((thresholds <= 0.0) | (thresholds >= 1.0)):
        raise ValueError("probability thresholds must lie strictly between zero and one")
    for column in ("baseline_probability", "optimized_probability"):
        probability = values[column].to_numpy(float)
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise ValueError(f"{column} must lie in [0, 1]")

    route_weights = np.zeros(len(values), dtype=float)
    for _, group in values.groupby("route_id", sort=False):
        ordered = group.sort_values(["distance_km", "point_index"])
        route_weights[ordered.index.to_numpy(int)] = _support_lengths(
            ordered["distance_km"].to_numpy(float)
        )
    total_weight = float(route_weights.sum())
    if total_weight <= 0.0:
        route_weights.fill(1.0)
        total_weight = float(len(route_weights))

    records: list[dict[str, object]] = []
    for label, column in (
        ("Baseline", "baseline_probability"),
        ("Optimized", "optimized_probability"),
    ):
        probability = values[column].to_numpy(float)
        covered = probability >= thresholds
        records.append({
            "method": label,
            "height_m": float(values["height_m"].iloc[0]),
            "threshold_min": float(thresholds.min()),
            "threshold_max": float(thresholds.max()),
            "points": int(len(values)),
            "covered_points": int(covered.sum()),
            "uncovered_points": int((~covered).sum()),
            "sample_coverage_fraction": float(np.mean(covered)),
            "route_length_coverage_fraction": float(
                np.sum(route_weights * covered) / total_weight
            ),
            "minimum_probability": float(probability.min()),
            "mean_probability": float(probability.mean()),
            "all_points_covered": bool(covered.all()),
        })
    return pd.DataFrame.from_records(records)


def route_probability_segments(
    frame: pd.DataFrame,
    probability_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return independent route line segments and their mean probabilities."""
    required = {"route_id", "point_index", "longitude", "latitude", probability_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"route data is missing columns: {missing}")
    segments: list[np.ndarray] = []
    values: list[float] = []
    for _, group in frame.groupby("route_id", sort=False):
        ordered = group.sort_values("point_index")
        xy = ordered[["longitude", "latitude"]].to_numpy(float)
        probability = ordered[probability_column].to_numpy(float)
        if len(xy) < 2:
            continue
        segments.extend(np.stack((xy[:-1], xy[1:]), axis=1))
        values.extend(0.5 * (probability[:-1] + probability[1:]))
    if not segments:
        return np.empty((0, 2, 2), dtype=float), np.empty(0, dtype=float)
    result_segments = np.asarray(segments, dtype=float)
    result_values = np.asarray(values, dtype=float)
    if not np.isfinite(result_segments).all() or not np.isfinite(result_values).all():
        raise ValueError("route coordinates and probabilities must be finite")
    return result_segments, result_values
