"""Generate sparse physical probability-map labels from a scenario model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ScenarioConfig
from .detection import at_least_k_probability
from .physics_baseline import compute_route_station_probabilities


def generate_sparse_physical_labels(
    samples: pd.DataFrame,
    stations: pd.DataFrame,
    config: ScenarioConfig,
    *,
    scenario_version: str = "scenario.toml",
    additional_loss_db: float = 0.0,
) -> pd.DataFrame:
    """Generate reproducible labels for arbitrary sparse points and heights.

    ``label_evidence=physics_scenario`` deliberately distinguishes these labels
    from local LTE-calibrated observations and from a future terrain-aware
    propagation run.  The returned table is suitable for surrogate training.
    """
    required = {"longitude", "latitude", "height_m"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"missing sample columns: {sorted(missing)}")
    if stations.empty:
        raise ValueError("stations must not be empty")

    records: list[dict[str, object]] = []
    for height_m, group in samples.groupby("height_m", sort=False):
        height = float(height_m)
        if height <= 0:
            raise ValueError("sample heights must be positive")
        coordinates = group[["longitude", "latitude"]].to_numpy(dtype=float)
        station_probabilities = compute_route_station_probabilities(
            coordinates,
            height,
            stations,
            config.radio,
            config.station_grades,
            additional_loss_db=additional_loss_db,
            environment=config.environment,
        )
        for row_position, (row_index, row) in enumerate(group.iterrows()):
            probabilities = station_probabilities[row_position]
            record = row.to_dict()
            record.update(
                {
                    "p_at_least_1": at_least_k_probability(probabilities, 1),
                    "p_at_least_2": at_least_k_probability(probabilities, 2),
                    "p_at_least_3": at_least_k_probability(probabilities, 3),
                    "label_evidence": "physics_scenario",
                    "scenario_version": Path(scenario_version).name,
                }
            )
            records.append(record)
    return pd.DataFrame.from_records(records).reset_index(drop=True)
