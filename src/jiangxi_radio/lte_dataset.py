from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_LTE_COLUMNS = frozenset({"longitude", "latitude", "rsrp_dbm", "rsrq_db"})


def aggregate_measurement_chunk(
    frame: pd.DataFrame,
    *,
    cell_size_deg: float = 0.001,
) -> pd.DataFrame:
    """Aggregate LTE measurements into spatial cells.

    The output keeps mean, standard deviation and sample count so the model can
    weight dense observations and learn heteroscedastic uncertainty. A 0.001°
    cell is roughly 100 m in Jiangxi and is only a default, not a physical law.
    """

    if cell_size_deg <= 0 or not np.isfinite(cell_size_deg):
        raise ValueError("cell_size_deg must be finite and positive")
    missing = REQUIRED_LTE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"missing LTE columns: {sorted(missing)}")
    values = frame.copy()
    for column in REQUIRED_LTE_COLUMNS:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    values = values.dropna(subset=list(REQUIRED_LTE_COLUMNS))
    values = values[np.isfinite(values[list(REQUIRED_LTE_COLUMNS)]).all(axis=1)]
    if values.empty:
        return pd.DataFrame(
            columns=["grid_lon", "grid_lat", "sample_count", "rsrp_mean_dbm", "rsrp_std_db", "rsrq_mean_db"]
        )
    values["grid_lon"] = np.floor(values["longitude"] / cell_size_deg) * cell_size_deg
    values["grid_lat"] = np.floor(values["latitude"] / cell_size_deg) * cell_size_deg
    grouped = values.groupby(["grid_lon", "grid_lat"], sort=True)
    result = grouped.agg(
        sample_count=("rsrp_dbm", "size"),
        rsrp_sum=("rsrp_dbm", "sum"),
        rsrp_sumsq=("rsrp_dbm", lambda values: float(np.square(values).sum())),
        rsrq_sum=("rsrq_db", "sum"),
        rsrp_mean_dbm=("rsrp_dbm", "mean"),
        rsrp_std_db=("rsrp_dbm", "std"),
        rsrq_mean_db=("rsrq_db", "mean"),
    ).reset_index()
    result["rsrp_std_db"] = result["rsrp_std_db"].fillna(0.0)
    return result
