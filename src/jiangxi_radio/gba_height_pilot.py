"""Non-destructive GBA download and building-height matching pilot."""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from http.client import HTTPException
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import time
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import ijson
from pyproj import Transformer
from shapely.geometry import box, mapping, shape
from shapely.ops import transform
from shapely.strtree import STRtree

from .building_heights import building_height, positive_number


HF = "https://huggingface.co"
LOD_REPO = "zhu-xlab/GBA.LoD1"
ODBL_REPO = "zhu-xlab/GBA.ODbLPolygon"
INDEX_PATH = "representative/lod1.geojson"
NANCHANG_BBOX = [115.64, 28.56, 115.89, 28.76]
USER_AGENT = "jiangxi-radio-height-pilot/1.0"


def validate_bbox(bbox: list[float]) -> list[float]:
    if len(bbox) != 4 or not all(math.isfinite(x) for x in bbox):
        raise ValueError("bbox must contain four finite numbers")
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -85 < south < north < 85):
        raise ValueError("bbox must be west,south,east,north within supported latitude range")
    return list(bbox)


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe relative path: {value}")
    if not value or not re.fullmatch(r"[A-Za-z0-9_./-]+", value):
        raise ValueError(f"Invalid relative path: {value}")
    return value


def select_tiles(index: dict[str, Any], bbox: list[float]) -> list[str]:
    roi = box(*validate_bbox(bbox))
    tiles = set()
    for feature in index["features"]:
        geometry = shape(feature["geometry"])
        if geometry.intersection(roi).area > 0:
            tiles.add(safe_relative(feature["properties"]["tile"]))
    if not tiles:
        raise ValueError("Official index has no intersecting tiles")
    return sorted(tiles)


def remote_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=45) as response:
        return json.load(response)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def make_plan(output: Path, bbox: list[float]) -> dict[str, Any]:
    """Resolve immutable repo revisions before querying index and file metadata."""
    if output.exists():
        raise FileExistsError(f"Plan already exists; reuse it rather than changing its pinned versions: {output}")
    revisions = {repo: remote_json(f"{HF}/api/datasets/{repo}")["sha"]
                 for repo in (LOD_REPO, ODBL_REPO)}
    index_url = f"{HF}/datasets/{LOD_REPO}/resolve/{revisions[LOD_REPO]}/{INDEX_PATH}"
    index = remote_json(index_url)
    tiles = select_tiles(index, bbox)
    assets = []
    directory_cache = {}
    for tile in tiles:
        for kind, repo, remote_path in (
            ("odbl", ODBL_REPO, f"{tile}.geojson"),
            ("polygon", LOD_REPO, f"Polygon/{tile}.geojson"),
            ("height", LOD_REPO, f"LoD1/{tile}.json"),
        ):
            parent = str(PurePosixPath(remote_path).parent)
            cache_key = (repo, parent)
            if cache_key not in directory_cache:
                url = f"{HF}/api/datasets/{repo}/tree/{revisions[repo]}/{parent}?limit=1000"
                # Directories currently contain fewer than 1000 tiles. Refuse an
                # incomplete listing rather than silently omitting a required part.
                request = Request(url, headers={"User-Agent": USER_AGENT})
                with urlopen(request, timeout=45) as response:
                    if 'rel="next"' in response.headers.get("Link", ""):
                        raise ValueError("Repository listing is paginated; update the planner before use")
                    directory_cache[cache_key] = json.load(response)
            entry = next((x for x in directory_cache[cache_key] if x["path"] == remote_path), None)
            if entry is None:
                raise ValueError(f"Required GBA part is absent: {repo}/{remote_path}")
            assets.append({
                "kind": kind, "tile": tile, "repo": repo, "revision": revisions[repo],
                "remote_path": remote_path, "local_path": f"{kind}/{tile}{Path(remote_path).suffix}",
                "url": f"{HF}/datasets/{repo}/resolve/{revisions[repo]}/{remote_path}",
                "size": int(entry["size"]), "sha256": entry.get("lfs", {}).get("oid"),
                "license": "ODbL-1.0" if kind == "odbl" else "CC-BY-NC-4.0",
            })
    plan = {
        "schema_version": 1, "bbox": validate_bbox(bbox), "tiles": tiles,
        "created_utc": datetime.now(timezone.utc).isoformat(), "revisions": revisions,
        "official_index_url": index_url, "source_crs": "EPSG:3857",
        "index_crs": "EPSG:4326", "assets": assets,
        "total_bytes": sum(x["size"] for x in assets),
        "evidence": "remote_sensing_prediction_not_measurement",
        "source_readme": "https://github.com/zhu-xlab/GlobalBuildingAtlas",
        "license_note": "Mixed ODbL and CC-BY-NC-4.0 sources; preserve attribution and review downstream restrictions.",
    }
    save_json(output, plan)
    save_json(output.with_name(output.stem + "_index.json"), index)
    return plan


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_matches(path: Path, size: int, sha256: str | None) -> bool:
    return path.is_file() and path.stat().st_size == size and (
        sha256 is None or sha256_file(path) == sha256
    )


def download_asset(asset: dict[str, Any], root: Path, *, deadline: float,
                   opener=urlopen) -> dict[str, Any]:
    relative = safe_relative(asset["local_path"])
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Download path escapes data root")
    expected_size = int(asset["size"])
    if expected_size < 0:
        raise ValueError("Negative expected file size")
    expected_hash = asset.get("sha256")
    if file_matches(path, expected_size, expected_hash):
        return {"local_path": relative, "size": expected_size, "sha256": sha256_file(path)}
    if path.exists():
        raise ValueError(f"Existing file failed integrity check; preserved without overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    receipt = part.with_suffix(part.suffix + ".json")
    identity = {"url": asset["url"], "size": expected_size, "sha256": expected_hash}
    if part.exists() and (not receipt.exists() or json.loads(receipt.read_text(encoding="utf-8")) != identity):
        raise ValueError(f"Partial file has different or missing provenance: {part}")
    save_json(receipt, identity)
    for attempt in range(3):
        if time.monotonic() >= deadline:
            raise TimeoutError("Download time limit reached; run the same command to resume")
        offset = part.stat().st_size if part.exists() else 0
        if offset > expected_size:
            raise ValueError(f"Oversized partial file: {part}")
        if offset == expected_size:
            break
        headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        # Refresh expiring CDN redirects; keep immutable content revision in URL.
        request = Request(asset["url"] + f"?download=true&request_time={time.time_ns()}", headers=headers)
        try:
            with opener(request, timeout=min(45, max(1, deadline - time.monotonic()))) as response:
                status = response.status
                if offset and status != 206:
                    raise ValueError("Server did not honor Range; partial file preserved, not appended")
                if status not in (200, 206):
                    raise ValueError(f"Unexpected download HTTP status: {status}")
                if status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-") or not content_range.endswith(f"/{expected_size}"):
                        raise ValueError(f"Unexpected Content-Range: {content_range}")
                last_report = time.monotonic()
                with part.open("ab") as stream:
                    while True:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Download time limit reached; partial retained for resume")
                        chunk = response.read(min(1024 * 1024, expected_size - offset + 1))
                        if not chunk:
                            break
                        if offset + len(chunk) > expected_size:
                            raise ValueError("Download exceeds declared file size")
                        stream.write(chunk)
                        offset += len(chunk)
                        if time.monotonic() - last_report >= 10:
                            print(f"{relative}: {offset / 1e6:.1f}/{expected_size / 1e6:.1f} MB", flush=True)
                            last_report = time.monotonic()
            if offset == expected_size:
                break
        except (OSError, HTTPException) as exc:
            print(f"Download attempt {attempt + 1}: {exc}", file=sys.stderr, flush=True)
            if attempt == 2 or time.monotonic() >= deadline:
                raise
    if not file_matches(part, expected_size, expected_hash):
        raise ValueError(f"Incomplete download or SHA256 mismatch; partial retained: {part}")
    digest = sha256_file(part)
    part.rename(path)
    return {"local_path": relative, "size": expected_size, "sha256": digest}


def download_plan(plan: dict[str, Any], root: Path, max_download_gb: float, max_seconds: float) -> None:
    if not math.isfinite(max_download_gb) or max_download_gb <= 0 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Download limits must be positive and finite")
    total = sum(int(x["size"]) for x in plan["assets"])
    if total > max_download_gb * 1e9:
        raise ValueError(f"Plan needs {total / 1e9:.3f} GB, above limit {max_download_gb} GB")
    for asset in plan["assets"]:
        if urlparse(asset["url"]).scheme != "https" or urlparse(asset["url"]).hostname != "huggingface.co":
            raise ValueError("Only official HTTPS Hugging Face assets are supported")
        safe_relative(asset["local_path"])
    deadline = time.monotonic() + max_seconds
    receipts = []
    for asset in sorted(plan["assets"], key=lambda x: x["size"]):
        print(f"Downloading/checking {asset['local_path']} ({asset['size'] / 1e6:.1f} MB)", flush=True)
        receipts.append(download_asset(asset, root, deadline=deadline))
        save_json(root / "download_receipts.json", {"assets": receipts, "complete": len(receipts) == len(plan["assets"])})


def iter_features(path: Path):
    with path.open("rb") as stream:
        yield from ijson.items(stream, "features.item", use_float=True)


def official_key(properties: dict[str, Any]) -> str:
    if not all(properties.get(key) is not None for key in ("source", "id", "region")):
        raise ValueError("GBA feature lacks source/id/region needed for official height join")
    return "".join(str(properties[key]) for key in ("source", "id", "region"))


def read_gba_roi(polygon_paths: list[Path], height_paths: list[Path],
                 bbox: list[float]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    roi = box(*validate_bbox(bbox))
    to_mercator = Transformer.from_crs(4326, 3857, always_xy=True).transform
    to_wgs84 = Transformer.from_crs(3857, 4326, always_xy=True).transform
    projected_roi = transform(to_mercator, roi)
    features = []
    keys = set()
    stats = Counter()
    for path in polygon_paths:
        print(f"Reading polygons: {path.name}", flush=True)
        for feature in iter_features(path):
            stats["scanned_polygons"] += 1
            if stats["scanned_polygons"] % 250000 == 0:
                print(f"Scanned {stats['scanned_polygons']} polygons; ROI retained {len(features)}", flush=True)
            try:
                geometry = shape(feature.get("geometry"))
            except (ValueError, TypeError, AttributeError):
                stats["invalid_geometry"] += 1
                continue
            if geometry.is_empty or not geometry.is_valid or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
                stats["invalid_geometry"] += 1
                continue
            bounds = geometry.bounds
            if all(abs(bounds[i]) <= (180 if i % 2 == 0 else 90) for i in range(4)):
                raise ValueError(f"Expected official EPSG:3857 metres but found degree-like coordinates in {path}")
            if not projected_roi.intersects(geometry):
                continue
            key = official_key(feature["properties"])
            if key in keys:
                raise ValueError(f"Duplicate GBA key in ROI: {key}; resolve duplicates before matching")
            keys.add(key)
            features.append({"type": "Feature", "geometry": mapping(transform(to_wgs84, geometry)),
                             "properties": {**feature["properties"], "gba_key": key,
                                            "height": None, "var": None}})
    values = {}
    height_records = 0
    for path in height_paths:
        print(f"Reading heights: {path.name}", flush=True)
        with path.open("rb") as stream:
            for key, value in ijson.kvitems(stream, "", use_float=True):
                height_records += 1
                if height_records % 1000000 == 0:
                    print(f"Scanned {height_records} height records; ROI keys found {len(values)}", flush=True)
                if key in keys:
                    if key in values:
                        raise ValueError(f"Duplicate height key in ROI: {key}")
                    values[key] = value
    for feature in features:
        properties = feature["properties"]
        value = values.get(properties["gba_key"])
        if value is not None:
            properties["height"] = positive_number(value.get("height"))
            raw_var = value.get("var")
            properties["var"] = float(raw_var) if isinstance(raw_var, (int, float)) and math.isfinite(raw_var) else None
            stats["height_joined"] += 1
        stats["valid_height"] += int(properties["height"] is not None)
    stats["roi_polygons"] = len(features)
    return features, dict(stats)


def overlap_matches(targets: list, donors: list, heights: list,
                    min_iou: float = 0.5, significant_overlap: float = 0.2) -> list[dict[str, Any]]:
    if len(donors) != len(heights) or not (0 < min_iou <= 1 and 0 < significant_overlap <= 1):
        raise ValueError("Invalid matching inputs or thresholds")
    valid_indices = [i for i, g in enumerate(donors) if g is not None and g.is_valid and not g.is_empty and g.area > 0]
    tree = STRtree([donors[i] for i in valid_indices])
    rows = []
    significant_claims = Counter()
    for target in targets:
        if target is None or target.is_empty or not target.is_valid or target.area <= 0:
            rows.append({"status": "invalid_geometry", "donor_index": None, "iou": 0.0, "significant_donors": []})
            continue
        candidates = []
        significant = []
        for local_index in tree.query(target, predicate="intersects"):
            i = valid_indices[int(local_index)]
            donor = donors[i]
            area = target.intersection(donor).area
            if area <= 0:
                continue
            iou = area / (target.area + donor.area - area)
            candidates.append((iou, i))
            if iou >= min_iou or area / min(target.area, donor.area) >= significant_overlap:
                significant.append(i)
                significant_claims[i] += 1
        if not candidates:
            rows.append({"status": "no_overlap", "donor_index": None, "iou": 0.0, "significant_donors": []})
            continue
        iou, i = max(candidates)
        status = "matched_candidate" if iou >= min_iou else "low_overlap"
        rows.append({"status": status, "donor_index": i, "iou": iou, "significant_donors": significant})
    for row in rows:
        i = row["donor_index"]
        if i is None:
            continue
        if len(row["significant_donors"]) > 1 or any(significant_claims[k] > 1 for k in row["significant_donors"]):
            row["status"] = "ambiguous_overlap"
        elif row["status"] == "matched_candidate" and positive_number(heights[i]) is None:
            row["status"] = "missing_gba_height"
    return rows


def read_overture_roi(directory: Path, bbox: list[float]) -> tuple[list, dict[str, str]]:
    roi = box(*bbox)
    result = []
    source_hashes = {}
    for path in sorted(directory.glob("tile_*.geojson")):
        match = re.match(r"tile_(-?\d+)_(-?\d+)", path.stem)
        if not match:
            continue
        x, y = map(int, match.groups())
        # One-tile margin accounts for footprints crossing tile boundaries.
        if not roi.intersects(box((x - 1) / 10, (y - 1) / 10, (x + 2) / 10, (y + 2) / 10)):
            continue
        source_hashes[path.name] = sha256_file(path)
        for index, feature in enumerate(iter_features(path)):
            try:
                geometry = shape(feature.get("geometry"))
            except (ValueError, TypeError, AttributeError):
                continue
            if geometry.is_empty or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
                continue
            if not all(-180 <= x <= 180 and -90 <= y <= 90 for x, y in
                       [(geometry.bounds[0], geometry.bounds[1]), (geometry.bounds[2], geometry.bounds[3])]):
                raise ValueError("Overture source requires EPSG:4326 longitude/latitude")
            in_roi = geometry.intersects(roi) if geometry.is_valid else box(*geometry.bounds).intersects(roi)
            if in_roi:
                feature["properties"] = {**(feature.get("properties") or {}),
                                         "pilot_source_file": path.name, "pilot_source_index": index}
                result.append(feature)
    if not result:
        raise ValueError("No local Overture footprints in ROI; check directory, tiling and CRS")
    return result, source_hashes


def height_review_flag(height: float | None) -> str:
    value = positive_number(height)
    if value is None:
        return "missing"
    if value < 2:
        return "review_below_2m"
    if value > 300:
        return "review_above_300m"
    return "within_screening_range_not_validated"


def match_plan(plan: dict[str, Any], data_root: Path, buildings_dir: Path,
               output_dir: Path, min_iou: float = 0.5) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Use a new output directory; existing outputs preserved: {output_dir}")
    bbox = validate_bbox(plan["bbox"])
    polygon_paths, height_paths = [], []
    for asset in plan["assets"]:
        path = data_root / safe_relative(asset["local_path"])
        if not file_matches(path, asset["size"], asset.get("sha256")):
            raise ValueError(f"Missing or unverified asset: {path}; complete download first")
        (height_paths if asset["kind"] == "height" else polygon_paths).append(path)
    gba, gba_stats = read_gba_roi(polygon_paths, height_paths, bbox)
    if not gba:
        raise ValueError("No GBA polygons in ROI; do not treat this as absent buildings")
    local, local_hashes = read_overture_roi(buildings_dir, bbox)
    center_lon = (bbox[0] + bbox[2]) / 2
    center_lat = (bbox[1] + bbox[3]) / 2
    zone = min(60, max(1, math.floor((center_lon + 180) / 6) + 1))
    metric_crs = (32600 if center_lat >= 0 else 32700) + zone
    metric = Transformer.from_crs(4326, metric_crs, always_xy=True).transform
    targets = [transform(metric, shape(feature["geometry"])) for feature in local]
    donors = [transform(metric, shape(feature["geometry"])) for feature in gba]
    rows = overlap_matches(targets, donors, [x["properties"]["height"] for x in gba], min_iou)
    records = []
    for feature, row in zip(local, rows):
        original = feature["properties"]
        old = building_height(original)
        candidate = gba[row["donor_index"]]["properties"] if row["donor_index"] is not None else {}
        accepted = row["status"] == "matched_candidate"
        record = {
            "source_file": original["pilot_source_file"], "source_index": original["pilot_source_index"],
            "original_height_m": old.height_m, "original_height_source": old.source,
            "match_status": row["status"], "match_iou": row["iou"],
            "overlap_count": len(row["significant_donors"]),
            "gba_key": candidate.get("gba_key"), "gba_overlap_height_m": candidate.get("height"),
            "gba_candidate_height_m": candidate.get("height") if accepted else None,
            "gba_var_raw": candidate.get("var"),
            "candidate_source": "GBA_remote_sensing_prediction" if accepted else None,
            "candidate_height_review_flag": height_review_flag(candidate.get("height")) if accepted else "not_a_candidate",
            "independently_validated": False,
        }
        # Deliberately do not change height_m, height_source or geometry.
        feature["properties"] = {**original, **record}
        records.append(record)
    statuses = Counter(row["status"] for row in rows)
    candidates = [r["gba_candidate_height_m"] for r in records if r["gba_candidate_height_m"] is not None]
    differences = [r["gba_candidate_height_m"] - r["original_height_m"] for r in records
                   if r["gba_candidate_height_m"] is not None and r["original_height_m"] is not None]
    summary = {
        "bbox": bbox, "metric_crs": f"EPSG:{metric_crs}", "min_iou": min_iou,
        "significant_overlap_fraction": 0.2, "local_records": len(local), "gba": gba_stats,
        "status_counts": dict(statuses), "candidate_count": len(candidates),
        "candidate_fraction": len(candidates) / len(local),
        "candidate_height_review_counts": dict(Counter(height_review_flag(value) for value in candidates)),
        "candidate_mean_height_m": sum(candidates) / len(candidates) if candidates else None,
        "mean_candidate_minus_old_m": sum(differences) / len(differences) if differences else None,
        "source_files_sha256": local_hashes, "gba_revisions": plan["revisions"],
        "notes": ["Remote-sensing candidates only; original heights and source files unchanged.",
                  "Height differences are NOT accuracy metrics without independent reference measurements.",
                  "Geometric matching does not establish temporal correspondence or physical height accuracy.",
                  "Raw GBA var preserved; it is not interpreted as a calibrated confidence interval.",
                  "Heights below 2m or above 300m flagged for review, not discarded. These are screening assumptions, not truth bounds.",
                  plan["license_note"]],
    }
    output_dir.mkdir(parents=True)
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "source_plan.json", plan)
    save_json(output_dir / "gba_roi.geojson", {"type": "FeatureCollection", "features": gba})
    save_json(output_dir / "height_candidates.geojson", {"type": "FeatureCollection", "features": local})
    with (output_dir / "height_matches.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return summary
