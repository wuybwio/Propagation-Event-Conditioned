from __future__ import annotations

import numpy as np
import pandas as pd

from .routes import haversine_distance_km


def infer_lte_engineering_band(earfcn: object) -> str | None:
    """Map LTE EARFCN to the engineering-data band family.

    Engineering files use human-readable deployment families (700M/2.6G/4.9G),
    while measurements carry LTE EARFCN.  The mapping is intentionally coarse:
    it prevents matching a reused PCI to a different carrier family without
    pretending that LTE EARFCN and NR-ARFCN are numerically interchangeable.
    """
    if earfcn is None or pd.isna(earfcn):
        return None
    try:
        value = int(float(earfcn))
    except (TypeError, ValueError):
        return None
    if 9210 <= value <= 9659:  # LTE band 28, 700 MHz
        return "700M"
    if 39650 <= value <= 41589:  # LTE band 41, 2.6 GHz
        return "2.6G"
    return "other"


def match_grid_to_engineering(
    grid: pd.DataFrame,
    engineering: pd.DataFrame,
    *,
    high_distance_km: float = 1.0,
    medium_distance_km: float = 3.0,
) -> pd.DataFrame:
    """Match LTE grid observations to the nearest engineering cell by PCI.

    If ``lte_band`` is present, candidates are first filtered to the same
    coarse engineering band family.  When no compatible candidate exists, the
    method falls back to PCI-only matching and records that fallback explicitly.
    """

    required_grid = {"grid_lon", "grid_lat", "pci"}
    required_eng = {"eci", "pci", "longitude", "latitude"}
    if missing := required_grid.difference(grid.columns):
        raise ValueError(f"missing grid columns: {sorted(missing)}")
    if missing := required_eng.difference(engineering.columns):
        raise ValueError(f"missing engineering columns: {sorted(missing)}")
    eng = engineering.copy()
    eng["pci"] = pd.to_numeric(eng["pci"], errors="coerce")
    if "band" in eng.columns:
        eng["band_family"] = eng["band"].astype(str).str.strip()
    grid_values = grid.copy()
    grid_values["pci"] = pd.to_numeric(grid_values["pci"], errors="coerce")
    rows: list[dict[str, object]] = []
    for row in grid_values.itertuples(index=False):
        candidates = eng[eng["pci"] == row.pci]
        record = row._asdict()
        if candidates.empty:
            record.update({"matched_eci": "", "matched_longitude": np.nan, "matched_latitude": np.nan, "match_distance_km": np.nan, "match_method": "unmatched_pci", "match_confidence": "none"})
            rows.append(record)
            continue
        method = "pci_spatial_nearest"
        requested_band = getattr(row, "lte_band", None)
        if requested_band not in (None, "") and "band_family" in candidates.columns:
            compatible = candidates[candidates["band_family"] == requested_band]
            if not compatible.empty:
                candidates = compatible
                method = "pci_band_spatial_nearest"
            else:
                method = "pci_spatial_nearest_fallback"
        elif requested_band not in (None, ""):
            method = "pci_spatial_nearest_fallback"
        distances = candidates.apply(
            lambda item: haversine_distance_km(
                float(row.grid_lon), float(row.grid_lat),
                float(item.longitude), float(item.latitude),
            ), axis=1,
        )
        best_index = distances.idxmin()
        best = candidates.loc[best_index]
        distance = float(distances.loc[best_index])
        confidence = "high" if distance <= high_distance_km else "medium" if distance <= medium_distance_km else "low"
        record.update({"matched_eci": str(best.eci), "matched_longitude": float(best.longitude), "matched_latitude": float(best.latitude), "match_distance_km": distance, "match_method": method, "match_confidence": confidence})
        rows.append(record)
    return pd.DataFrame.from_records(rows)
