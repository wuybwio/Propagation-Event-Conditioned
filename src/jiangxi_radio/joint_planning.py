"""Utilities for shared candidate sites across multiple route scenarios."""

from __future__ import annotations

import pandas as pd


def deduplicate_shared_candidates(
    candidates: pd.DataFrame,
    coordinate_precision: int = 4,
) -> pd.DataFrame:
    """Merge candidates at the same rounded coordinate into one shared site."""
    required = {"candidate_id", "route_id", "longitude", "latitude"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"missing candidate columns: {sorted(missing)}")
    if coordinate_precision < 0:
        raise ValueError("coordinate_precision must be non-negative")
    if candidates.empty:
        return candidates.copy()
    result = candidates.copy()
    result["_lon_key"] = result["longitude"].round(coordinate_precision)
    result["_lat_key"] = result["latitude"].round(coordinate_precision)
    rows: list[dict[str, object]] = []
    for (lon_key, lat_key), group in result.groupby(["_lon_key", "_lat_key"], sort=False):
        first = group.iloc[0].to_dict()
        first["candidate_id"] = f"shared-{float(lon_key):.6f}-{float(lat_key):.6f}"
        first["longitude"] = float(group["longitude"].mean())
        first["latitude"] = float(group["latitude"].mean())
        first["shared_route_ids"] = "|".join(sorted({str(v) for v in group["route_id"]}))
        first["source_candidate_ids"] = "|".join(str(v) for v in group["candidate_id"])
        rows.append(first)
    return pd.DataFrame.from_records(rows).drop(columns=["_lon_key", "_lat_key"], errors="ignore")
