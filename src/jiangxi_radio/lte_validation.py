"""LTE-only baselines. No fixed monitoring receivers or assumed flight heights."""

from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_lte_links(labels: pd.DataFrame, engineering: pd.DataFrame | None = None,
                      *, min_confidence: float = 0.0) -> tuple[pd.DataFrame, dict]:
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0,1]")
    required = {"grid_lon", "grid_lat", "matched_eci", "rsrp_mean_dbm", "label_weight", "sample_count"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Missing LTE columns: {sorted(missing)}")
    frame = labels.copy()
    frame["matched_eci"] = frame["matched_eci"].astype("string").str.strip()
    coords = ["matched_longitude", "matched_latitude"]
    if not set(coords).issubset(frame.columns):
        if engineering is None:
            raise ValueError("LTE cell coordinates missing: supply --engineering with eci,longitude,latitude; never use monitoring stations.csv")
        eng = engineering[["eci", "longitude", "latitude"]].copy()
        eng["eci"] = eng["eci"].astype("string").str.strip()
        eng = eng.drop_duplicates()
        if eng["eci"].duplicated().any():
            raise ValueError("Engineering ECI has conflicting coordinates")
        frame = frame.drop(columns=coords, errors="ignore").merge(
            eng.rename(columns={"eci": "matched_eci", "longitude": coords[0], "latitude": coords[1]}),
            how="left", on="matched_eci", validate="many_to_one")
    numeric = ["grid_lon", "grid_lat", "rsrp_mean_dbm", "label_weight", "sample_count", *coords]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(frame[numeric].to_numpy()).all(axis=1)
    valid &= frame["matched_eci"].notna() & ~frame["matched_eci"].str.lower().isin(["", "nan", "none", "<na>"])
    valid &= frame["label_weight"].between(min_confidence, 1) & (frame["label_weight"] > 0)
    valid &= frame["sample_count"] > 0
    for name in ("grid_lon", "matched_longitude"):
        valid &= frame[name].between(-180, 180)
    for name in ("grid_lat", "matched_latitude"):
        valid &= frame[name].between(-90, 90)
    audit = {"input_rows": len(labels), "retained_rows": int(valid.sum()),
             "excluded_rows": int((~valid).sum()), "min_confidence": min_confidence,
             "height_evidence": "not used; no flight altitude is assigned",
             "eci_caveat": "Matched ECI is a supplied association, not independently verified serving-cell identity"}
    if "match_method" in frame:
        audit["retained_match_methods"] = frame.loc[valid, "match_method"].value_counts().to_dict()
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise ValueError("No valid LTE links remain after filtering")
    # One record per measured spatial cell and matched LTE transmitter.
    if frame.groupby("matched_eci")[coords].nunique().gt(1).any().any():
        raise ValueError("A matched ECI has inconsistent engineering coordinates")
    frame["weighted_rsrp"] = frame["rsrp_mean_dbm"] * frame["sample_count"]
    links = frame.groupby(["grid_lon", "grid_lat", "matched_eci"], as_index=False).agg(
        weighted_rsrp=("weighted_rsrp", "sum"), sample_count=("sample_count", "sum"),
        label_weight=("label_weight", "mean"), matched_longitude=("matched_longitude", "first"),
        matched_latitude=("matched_latitude", "first"))
    links["rsrp_mean_dbm"] = links.pop("weighted_rsrp") / links["sample_count"]
    lon, lat, cell_lon, cell_lat = (np.radians(links[name].to_numpy(float)) for name in
                                   ("grid_lon", "grid_lat", "matched_longitude", "matched_latitude"))
    a = np.sin((lat - cell_lat) / 2)**2 + np.cos(lat)*np.cos(cell_lat)*np.sin((lon - cell_lon) / 2)**2
    links["horizontal_distance_km"] = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    audit["unique_grid_cell_links"] = len(links)
    audit["unique_spatial_cells"] = len(links[["grid_lon", "grid_lat"]].drop_duplicates())
    return links, audit


def split_lte_links(links: pd.DataFrame, *, val_fraction: float = 0.2, gap_deg: float = 0.002):
    if not 0 < val_fraction < 1 or not np.isfinite(gap_deg) or gap_deg < 0:
        raise ValueError("Invalid LTE spatial split parameters")
    latitudes = np.sort(links["grid_lat"].unique())
    if len(latitudes) < 3:
        raise ValueError("Need at least three latitude rows for spatial validation")
    cut = min(max(int(len(latitudes) * (1-val_fraction)), 1), len(latitudes)-1)
    boundary = float(latitudes[cut])
    train = links.loc[links.grid_lat < boundary - gap_deg].copy()
    val = links.loc[links.grid_lat >= boundary].copy()
    if len(train) < 3 or len(val) < 2:
        raise ValueError("Too few links after spatial split; inspect extent and buffer")
    return train, val, {"method": "latitude_holdout", "boundary_latitude": boundary,
                        "gap_deg": gap_deg, "val_fraction_requested": val_fraction,
                        "train_links": len(train), "val_links": len(val),
                        "discarded_buffer_links": len(links)-len(train)-len(val)}


def fit_predict_baselines(train: pd.DataFrame, val: pd.DataFrame) -> tuple[dict, dict]:
    y = train["rsrp_mean_dbm"].to_numpy(float)
    weights = train["label_weight"].to_numpy(float)
    # Empirical dBm regression, NOT absolute path loss: transmitter EIRP is unknown.
    x = np.log10(np.maximum(train["horizontal_distance_km"].to_numpy(float), 0.01))
    xv = np.log10(np.maximum(val["horizontal_distance_km"].to_numpy(float), 0.01))
    design = np.column_stack([np.ones(len(x)), x])
    coef = np.linalg.lstsq(design * np.sqrt(weights[:, None]), y * np.sqrt(weights), rcond=None)[0]
    residuals = pd.DataFrame({"eci": train["matched_eci"].to_numpy(), "wr": (y-design@coef)*weights, "w": weights})
    cell = residuals.groupby("eci")[["wr", "w"]].sum()
    # Fixed shrinkage specified before evaluation; unseen cells get zero correction.
    offsets = cell["wr"] / (cell["w"] + 10.0)
    distance_prediction = coef[0] + coef[1]*xv
    predictions = {
        "train_mean": np.full(len(val), np.mean(y)),
        "train_median": np.full(len(val), np.median(y)),
        "empirical_log_distance": distance_prediction,
        "empirical_log_distance_cell_offset": distance_prediction + val["matched_eci"].map(offsets).fillna(0).to_numpy(),
    }
    return predictions, {"train_mean_dbm": float(np.mean(y)), "train_median_dbm": float(np.median(y)),
                         "distance_intercept_dbm": float(coef[0]), "distance_slope_db_per_decade": float(coef[1]),
                         "cell_offset_shrinkage": 10.0, "cell_offsets_db": offsets.to_dict(),
                         "unseen_val_cell_links": int((~val["matched_eci"].isin(train["matched_eci"])).sum())}


def score_rsrp(links: pd.DataFrame, prediction: np.ndarray) -> dict:
    truth = links["rsrp_mean_dbm"].to_numpy(float)
    error = prediction - truth
    cells = links[["grid_lon", "grid_lat"]].copy()
    cells["absolute_error"] = np.abs(error)
    cells["squared_error"] = error**2
    macro = cells.groupby(["grid_lon", "grid_lat"])[["absolute_error", "squared_error"]].mean()
    return {"link_mae_db": float(np.mean(np.abs(error))), "link_rmse_db": float(np.sqrt(np.mean(error**2))),
            "bias_db": float(np.mean(error)), "grid_macro_mae_db": float(macro.absolute_error.mean()),
            "grid_macro_rmse_db": float(np.sqrt(macro.squared_error.mean())),
            "pearson_r": float(np.corrcoef(prediction, truth)[0, 1]) if np.std(prediction)>1e-8 and np.std(truth)>1e-8 else None}
