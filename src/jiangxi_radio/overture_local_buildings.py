"""Extract an audited local building collection from tiled Overture GeoJSON."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pyproj import Transformer
from shapely.geometry import GeometryCollection, MultiPolygon, box, mapping, shape
from shapely.ops import transform

from .building_heights import building_height


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _polygonal(geometry):
    if geometry.is_empty:
        return None
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry
    if isinstance(geometry, GeometryCollection):
        polygons = []
        for part in geometry.geoms:
            if part.geom_type == "Polygon":
                polygons.append(part)
            elif part.geom_type == "MultiPolygon":
                polygons.extend(part.geoms)
        if polygons:
            return MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]
    return None


def _intersects(bounds: dict[str, Any], roi_bounds: tuple[float, float, float, float]) -> bool:
    required = ("west", "south", "east", "north")
    if any(name not in bounds for name in required):
        raise ValueError(f"manifest grid_index is missing one of {required}")
    west, south, east, north = (float(bounds[name]) for name in required)
    roi_west, roi_south, roi_east, roi_north = roi_bounds
    return west < roi_east and east > roi_west and south < roi_north and north > roi_south


def extract_local_buildings(
    tiles_dir: str | Path,
    *,
    longitude: float,
    latitude: float,
    size_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return buildings intersecting a square UTM50N region of interest.

    The source directory must contain the project's ``cesium-geojson-grid``
    manifest and all files listed by it. Heights are not imputed: only positive
    source ``height_m`` or ``height`` values are retained.
    """
    root = Path(tiles_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Overture building manifest.json not found: {manifest_path}")
    if not (113.0 <= longitude <= 119.0 and 24.0 <= latitude <= 31.0):
        raise ValueError("local Overture extraction is restricted to the Jiangxi study region")
    if not 50.0 <= size_m <= 2000.0:
        raise ValueError("size_m must be between 50 and 2000 metres")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("type") != "cesium-geojson-grid" or not isinstance(manifest.get("tiles"), list):
        raise ValueError("unsupported Overture manifest; expected type=cesium-geojson-grid with tiles")

    forward = Transformer.from_crs(4326, 32650, always_xy=True)
    inverse = Transformer.from_crs(32650, 4326, always_xy=True)
    centre_x, centre_y = forward.transform(longitude, latitude)
    half = size_m / 2.0
    roi_projected = box(centre_x - half, centre_y - half, centre_x + half, centre_y + half)
    roi_geographic = transform(inverse.transform, roi_projected)

    selected = [
        tile for tile in manifest["tiles"]
        if _intersects(tile.get("grid_index") or {}, roi_geographic.bounds)
    ]
    if not selected:
        raise ValueError("the requested ROI is outside the indexed Overture building tiles")

    features: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    skipped_geometry = 0
    skipped_height = 0
    scanned = 0

    for tile in selected:
        relative = Path(str(tile.get("file", "")))
        source_path = (root / relative).resolve()
        try:
            source_path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"manifest tile escapes its source directory: {relative}") from error
        if not source_path.is_file():
            raise FileNotFoundError(f"manifest-listed Overture tile is missing: {source_path}")
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if payload.get("type") != "FeatureCollection":
            raise ValueError(f"not a GeoJSON FeatureCollection: {source_path}")
        kept = 0
        source_features = payload.get("features") or []
        for feature in source_features:
            scanned += 1
            try:
                geometry = shape(feature.get("geometry"))
            except (AttributeError, TypeError, ValueError):
                skipped_geometry += 1
                continue
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            if geometry.is_empty:
                skipped_geometry += 1
                continue
            projected = transform(forward.transform, geometry)
            if not projected.intersects(roi_projected):
                continue
            clipped = _polygonal(projected.intersection(roi_projected))
            if clipped is None or clipped.area <= 0.01:
                skipped_geometry += 1
                continue
            properties = dict(feature.get("properties") or {})
            observation = building_height(properties)
            if observation.height_m is None:
                skipped_height += 1
                continue
            properties["height"] = float(observation.height_m)
            properties["height_m"] = float(observation.height_m)
            properties["height_source"] = observation.source
            geographic = transform(inverse.transform, clipped)
            features.append({
                "type": "Feature",
                "properties": properties,
                "geometry": mapping(geographic),
            })
            kept += 1
        sources.append({
            "file": relative.as_posix(),
            "manifest_feature_count": tile.get("feature_count"),
            "scanned_feature_count": len(source_features),
            "kept_feature_count": kept,
        })

    audit = {
        "schema": "overture_local_v1",
        "source": manifest.get("source", "Overture Maps Buildings"),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": _sha256(manifest_path),
        "longitude": float(longitude),
        "latitude": float(latitude),
        "size_m": float(size_m),
        "coordinate_crs": "EPSG:32650",
        "height_evidence": "mixed_overture_attribute_and_rule_estimate",
        "measured_height_validation": False,
        "selected_tile_count": len(selected),
        "source_records_scanned": scanned,
        "building_components": len(features),
        "skipped_invalid_geometry": skipped_geometry,
        "skipped_missing_height": skipped_height,
        "source_tiles": sources,
    }
    return {"type": "FeatureCollection", "features": features, "audit": audit}, audit
