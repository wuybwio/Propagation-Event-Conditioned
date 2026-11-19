"""Local LTE measurement preparation and leakage-safe residual calibration.

The Nanchang samples are not a province-wide label set.  This module therefore
only estimates an empirical correction and uncertainty for a supplied physical
baseline; callers must identify every output as locally measurement-calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _as_negative_radio_measurement(
    values: pd.Series, *, lower_db: float, upper_db: float
) -> pd.Series:
    """Normalize signed and magnitude-encoded radio measurements to negative dB."""
    numeric = pd.to_numeric(values, errors="coerce")
    # Some collection software records RSRP/RSRQ as the magnitude of a negative
    # dB value.  Positives in the valid magnitude range are interpreted as such.
    normalized = numeric.where(numeric <= 0, -numeric)
    return normalized.where(normalized.between(lower_db, upper_db))


def prepare_lte_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    """Select, normalize, and clean the fields usable for local calibration."""
    required = {"longitude", "latitude", "ltescrsrp", "ltescrsrq"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing LTE measurement columns: {sorted(missing)}")

    result = pd.DataFrame(
        {
            "longitude": pd.to_numeric(frame["longitude"], errors="coerce"),
            "latitude": pd.to_numeric(frame["latitude"], errors="coerce"),
            "rsrp_dbm": _as_negative_radio_measurement(
                frame["ltescrsrp"], lower_db=-160.0, upper_db=-30.0
            ),
            "rsrq_db": _as_negative_radio_measurement(
                frame["ltescrsrq"], lower_db=-35.0, upper_db=-1.0
            ),
        }
    )
    valid_coordinates = (
        result["longitude"].between(-180.0, 180.0)
        & result["latitude"].between(-90.0, 90.0)
    )
    return result.loc[
        valid_coordinates & result["rsrp_dbm"].notna() & result["rsrq_db"].notna()
    ].reset_index(drop=True)


def _block_keys(frame: pd.DataFrame, block_size_degrees: float) -> pd.Series:
    if block_size_degrees <= 0:
        raise ValueError("block_size_degrees must be positive")
    x = np.floor(frame["longitude"].to_numpy() / block_size_degrees).astype(int)
    y = np.floor(frame["latitude"].to_numpy() / block_size_degrees).astype(int)
    return pd.Series([f"{a}:{b}" for a, b in zip(x, y, strict=True)], index=frame.index)


def spatial_block_split(
    frame: pd.DataFrame,
    *,
    block_size_degrees: float = 0.02,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by spatial blocks so neighboring samples cannot leak across sets."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if len(frame) < 2:
        raise ValueError("at least two samples are required")
    keys = _block_keys(frame, block_size_degrees)
    unique_keys = np.asarray(sorted(keys.unique()))
    if len(unique_keys) < 2:
        raise ValueError("at least two spatial blocks are required")
    n_validation = max(1, int(round(len(unique_keys) * validation_fraction)))
    n_validation = min(n_validation, len(unique_keys) - 1)
    rng = np.random.default_rng(seed)
    validation_keys = set(rng.choice(unique_keys, size=n_validation, replace=False))
    validation_mask = keys.isin(validation_keys)
    return (
        frame.loc[~validation_mask].copy().reset_index(drop=True),
        frame.loc[validation_mask].copy().reset_index(drop=True),
    )


def spatial_block_kfold(
    frame: pd.DataFrame,
    *,
    block_size_degrees: float = 0.02,
    folds: int = 5,
    seed: int = 0,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Return leakage-safe spatial folds for sparse regional measurements.

    Assignment is made at block level, so every sample from one geographic
    block belongs to exactly one validation fold.  Blocks are shuffled with a
    fixed seed and distributed round-robin; this avoids the optimistic scores
    caused by random point-wise splits while remaining usable for small
    Nanchang samples.
    """
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if len(frame) < folds:
        raise ValueError("at least one sample per fold is required")
    keys = _block_keys(frame, block_size_degrees)
    unique_keys = np.asarray(sorted(keys.unique()), dtype=object)
    if len(unique_keys) < folds:
        raise ValueError("at least as many spatial blocks as folds are required")
    rng = np.random.default_rng(seed)
    shuffled = unique_keys.copy()
    rng.shuffle(shuffled)
    assignments = {str(key): index % folds for index, key in enumerate(shuffled)}
    result: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for fold in range(folds):
        validation_mask = keys.map(lambda key: assignments[str(key)] == fold)
        train = frame.loc[~validation_mask].copy().reset_index(drop=True)
        validation = frame.loc[validation_mask].copy().reset_index(drop=True)
        if train.empty or validation.empty:
            raise RuntimeError("spatial fold unexpectedly produced an empty partition")
        result.append((train, validation))
    return result


def cross_validate_residual_calibrator(
    frame: pd.DataFrame,
    *,
    block_size_degrees: float = 0.02,
    folds: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Evaluate residual calibration under spatially disjoint validation.

    The returned table is deliberately model-agnostic and can be written into
    a paper's supplementary results.  Every fold reports raw physical MAE and
    calibrated MAE, with the validation residual never used during fitting.
    """
    required = {"longitude", "latitude", "rsrp_dbm", "physical_prediction_dbm"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing calibration columns: {sorted(missing)}")
    records: list[dict[str, float | int]] = []
    for fold, (training, validation) in enumerate(
        spatial_block_kfold(
            frame,
            block_size_degrees=block_size_degrees,
            folds=folds,
            seed=seed,
        )
    ):
        calibrator = fit_residual_binning_calibrator(
            training, bin_size_degrees=block_size_degrees
        )
        observed = validation["rsrp_dbm"].to_numpy(float)
        physical = validation["physical_prediction_dbm"].to_numpy(float)
        calibrated = calibrator.calibrated_prediction(validation)
        records.append(
            {
                "fold": fold,
                "training_samples": int(len(training)),
                "validation_samples": int(len(validation)),
                "physical_mae_db": float(np.mean(np.abs(observed - physical))),
                "calibrated_mae_db": float(np.mean(np.abs(observed - calibrated))),
                "residual_bias_db": float(np.mean(observed - calibrated)),
            }
        )
    return pd.DataFrame.from_records(records)


@dataclass(frozen=True)
class ResidualBinningCalibrator:
    """A transparent residual model, fitted only on the training partition."""

    bin_size_degrees: float
    global_residual_db: float
    residual_std_db: float
    bin_residual_db: dict[str, float]

    def predict_residual(self, frame: pd.DataFrame) -> np.ndarray:
        keys = _block_keys(frame, self.bin_size_degrees)
        return np.asarray(
            [self.bin_residual_db.get(key, self.global_residual_db) for key in keys],
            dtype=float,
        )

    def calibrated_prediction(self, frame: pd.DataFrame) -> np.ndarray:
        if "physical_prediction_dbm" not in frame.columns:
            raise ValueError("physical_prediction_dbm is required")
        baseline = pd.to_numeric(frame["physical_prediction_dbm"], errors="raise")
        return baseline.to_numpy(dtype=float) + self.predict_residual(frame)


def fit_residual_binning_calibrator(
    training_frame: pd.DataFrame, *, bin_size_degrees: float = 0.02
) -> ResidualBinningCalibrator:
    """Fit a geographical residual table using a training partition only."""
    required = {"longitude", "latitude", "rsrp_dbm", "physical_prediction_dbm"}
    missing = required.difference(training_frame.columns)
    if missing:
        raise ValueError(f"missing calibration columns: {sorted(missing)}")
    if training_frame.empty:
        raise ValueError("training_frame must not be empty")
    observed = pd.to_numeric(training_frame["rsrp_dbm"], errors="raise")
    baseline = pd.to_numeric(training_frame["physical_prediction_dbm"], errors="raise")
    residual = observed - baseline
    keys = _block_keys(training_frame, bin_size_degrees)
    bin_means = residual.groupby(keys, sort=True).mean().to_dict()
    return ResidualBinningCalibrator(
        bin_size_degrees=bin_size_degrees,
        global_residual_db=float(residual.mean()),
        residual_std_db=float(residual.std(ddof=0)),
        bin_residual_db={str(key): float(value) for key, value in bin_means.items()},
    )
