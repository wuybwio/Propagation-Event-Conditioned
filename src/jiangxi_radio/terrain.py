"""Lightweight SRTM HGT download and sampling utilities.

The reader intentionally uses only the Python standard library and NumPy so
the terrain layer can be used before optional GIS packages are installed.
SRTM 1 arc-second tiles contain 3601 x 3601 signed big-endian elevations.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from urllib.request import urlopen

import numpy as np


TILE_SIZE = 3601


def _tile_name(latitude: float, longitude: float) -> str:
    ns = "N" if latitude >= 0 else "S"
    ew = "E" if longitude >= 0 else "W"
    return f"{ns}{abs(int(np.floor(latitude))):02d}{ew}{abs(int(np.floor(longitude))):03d}"


def srtm_tile_urls(
    west: float,
    south: float,
    east: float,
    north: float,
) -> list[tuple[str, str]]:
    """Return ``(tile_name, public_url)`` pairs covering a bounding box."""
    names: list[tuple[str, str]] = []
    for latitude in range(int(np.floor(south)), int(np.ceil(north))):
        for longitude in range(int(np.floor(west)), int(np.ceil(east))):
            name = _tile_name(latitude, longitude)
            hemisphere_lat = "N" if latitude >= 0 else "S"
            hemisphere_lon = "E" if longitude >= 0 else "W"
            filename = f"{hemisphere_lat}{abs(latitude):02d}{hemisphere_lon}{abs(longitude):03d}.hgt.gz"
            url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{hemisphere_lat}{abs(latitude):02d}/{filename}"
            names.append((name, url))
    return names


def download_srtm_tiles(
    output_dir: str | Path,
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    overwrite: bool = False,
    timeout_s: float = 120.0,
) -> dict[str, int]:
    """Download public SRTM tiles as compressed ``.hgt.gz`` files."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    downloaded = skipped = failed = 0
    for name, url in srtm_tile_urls(west, south, east, north):
        path = target / f"{name}.hgt.gz"
        if path.exists() and path.stat().st_size > 1024 and not overwrite:
            skipped += 1
            continue
        try:
            with urlopen(url, timeout=timeout_s) as response, path.open("wb") as stream:
                stream.write(response.read())
            downloaded += 1
        except Exception:
            if path.exists():
                path.unlink()
            failed += 1
    return {"downloaded": downloaded, "skipped": skipped, "failed": failed}


def _read_tile(path: Path) -> np.ndarray:
    raw_path = path.with_suffix("") if path.suffix == ".gz" else path
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as stream:
            raw = stream.read()
    else:
        raw = path.read_bytes()
    expected = TILE_SIZE * TILE_SIZE * 2
    if len(raw) != expected:
        raise ValueError(f"invalid SRTM tile size: {path} ({len(raw)} bytes)")
    return np.frombuffer(raw, dtype=">i2").reshape(TILE_SIZE, TILE_SIZE).astype(np.float32)


def sample_srtm_elevation(
    points: np.ndarray,
    dem_dir: str | Path,
    *,
    nodata_m: float = 0.0,
) -> np.ndarray:
    """Sample SRTM elevations at longitude/latitude points by bilinear interpolation."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("points must have shape [n, 2]")
    root = Path(dem_dir)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"dem_dir does not exist or is not a directory: {root}")
    # DEM collections are commonly unpacked below a product subdirectory
    # (for example ``data/dem/srtm1``).  Build one filename index so both
    # flat and nested layouts are supported without repeated disk scans.
    tile_paths: dict[str, Path] = {}
    for candidate in root.rglob("*.hgt.gz"):
        tile_paths.setdefault(candidate.name[:-7], candidate)
    for candidate in root.rglob("*.hgt"):
        tile_paths.setdefault(candidate.stem, candidate)
    result = np.full(len(values), float(nodata_m), dtype=np.float32)
    cache: dict[str, np.ndarray] = {}
    groups: dict[str, list[int]] = {}
    for index, (lon, lat) in enumerate(values):
        groups.setdefault(_tile_name(float(lat), float(lon)), []).append(index)
    for name, indices in groups.items():
        path = tile_paths.get(name)
        if path is None:
            continue
        if name not in cache:
            cache[name] = _read_tile(path)
        tile = cache[name]
        selected = values[indices]
        lat0 = np.floor(selected[:, 1])
        lon0 = np.floor(selected[:, 0])
        x = np.clip((selected[:, 0] - lon0) * (TILE_SIZE - 1), 0, TILE_SIZE - 1.0)
        y = np.clip((selected[:, 1] - lat0) * (TILE_SIZE - 1), 0, TILE_SIZE - 1.0)
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        x1 = np.minimum(x0 + 1, TILE_SIZE - 1)
        y1 = np.minimum(y0 + 1, TILE_SIZE - 1)
        # HGT rows are north-to-south.
        row0 = TILE_SIZE - 1 - y0
        row1 = TILE_SIZE - 1 - y1
        dx = x - x0
        dy = y - y0
        sampled = (
            tile[row0, x0] * (1 - dx) * (1 - dy)
            + tile[row0, x1] * dx * (1 - dy)
            + tile[row1, x0] * (1 - dx) * dy
            + tile[row1, x1] * dx * dy
        )
        sampled = np.where(sampled < -1000, nodata_m, sampled)
        result[np.asarray(indices)] = sampled.astype(np.float32)
    return result
