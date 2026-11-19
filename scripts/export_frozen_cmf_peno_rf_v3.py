"""Export one leakage-safe frozen CMF-PENO-RF v3 prediction archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jiangxi_radio.frozen_v3_release import (
    conformal_multiplier,
    interval_metrics,
    normal_detection_probability,
    representative_case_index,
    validation_bias_shift,
)
from jiangxi_radio.uncertainty_calibration import load_temperature


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _require(data: dict[str, np.ndarray], names: set[str], source: Path) -> None:
    missing = sorted(names.difference(data))
    if missing:
        raise ValueError(f"{source} is missing fields: {missing}")


def _case_axis(data: dict[str, np.ndarray], name: str, indices: np.ndarray) -> np.ndarray:
    value = np.asarray(data[name])
    return value[indices] if value.ndim == 2 else value


def _case_surface(data: dict[str, np.ndarray], name: str, indices: np.ndarray) -> np.ndarray:
    value = np.asarray(data[name])
    return value[indices] if value.ndim == 3 else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target-coverage", type=float, default=0.975)
    parser.add_argument("--threshold-dbm", type=float, default=-100.0)
    parser.add_argument("--shadow-std-db", type=float, default=6.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing overwrite: {args.output}")

    dataset = _load_npz(args.dataset)
    validation = _load_npz(args.validation_predictions)
    test = _load_npz(args.test_predictions)
    prediction_fields = {"mean_dbm", "variance_db2", "target_dbm", "mask", "case_indices", "split"}
    _require(validation, prediction_fields, args.validation_predictions)
    _require(test, prediction_fields, args.test_predictions)
    _require(
        dataset,
        {"x_m", "y_m", "z_m", "terrain_m", "surface_m", "air_mask", "station_xyz_m", "split", "coarse_dbm"},
        args.dataset,
    )
    if str(validation["split"].item()) != "val" or str(test["split"].item()) != "test":
        raise ValueError("validation/test prediction archives have the wrong split contract")

    split = np.asarray(dataset["split"], dtype=int)
    val_indices = np.asarray(validation["case_indices"], dtype=np.int64)
    test_indices = np.asarray(test["case_indices"], dtype=np.int64)
    if np.any(split[val_indices] != 1) or np.any(split[test_indices] != 2):
        raise ValueError("prediction case indices do not match dataset splits")
    temperature = load_temperature(args.calibration_json)
    val_variance = np.asarray(validation["variance_db2"], dtype=np.float32) * temperature
    shift = validation_bias_shift(validation["mean_dbm"], validation["target_dbm"], validation["mask"])
    q = conformal_multiplier(
        np.asarray(validation["mean_dbm"], dtype=float) + shift,
        val_variance,
        validation["target_dbm"],
        validation["mask"],
        args.target_coverage,
    )

    mean = np.asarray(test["mean_dbm"], dtype=np.float32) + shift
    variance = np.asarray(test["variance_db2"], dtype=np.float32)
    target = np.asarray(test["target_dbm"], dtype=np.float32)
    mask = np.asarray(test["mask"], dtype=bool)
    if mean.shape != variance.shape or mean.shape != target.shape or mean.shape != mask.shape:
        raise ValueError("test prediction fields are not aligned")
    std = np.sqrt(np.maximum(variance, 1e-6))
    lower = mean - q * std
    upper = mean + q * std
    probability = normal_detection_probability(mean, variance, args.threshold_dbm, args.shadow_std_db)
    conservative_probability = normal_detection_probability(
        lower, 0.0, args.threshold_dbm, args.shadow_std_db
    )
    test_interval = interval_metrics(mean, variance, target, mask, q)
    representative = representative_case_index(mean, target, mask)

    air = np.asarray(dataset["air_mask"], dtype=bool)
    selected_air = air[test_indices] if air.ndim == 4 else np.broadcast_to(air, mask.shape)
    if not np.array_equal(selected_air, mask):
        raise ValueError("dataset air mask and prediction mask disagree")
    coarse = np.asarray(dataset["coarse_dbm"], dtype=np.float32)[test_indices]
    stations = np.asarray(dataset["station_xyz_m"], dtype=np.float32)[test_indices]
    source_station = (
        np.asarray(dataset["source_station_index"], dtype=np.int64)[test_indices]
        if "source_station_index" in dataset
        else test_indices.copy()
    )
    metadata = {
        "schema": "frozen_cmf_peno_rf_v3_release_v1",
        "evidence": "simulation_teacher_agreement_not_measurement_accuracy",
        "scene": args.scene,
        "seed": args.seed,
        "policy": "full_v3_validation_bias_corrected_conformal",
        "target_coverage": args.target_coverage,
        "threshold_dbm": args.threshold_dbm,
        "shadow_std_db": args.shadow_std_db,
        "validation_bias_shift_db": shift,
        "variance_temperature": temperature,
        "conformal_multiplier": q,
        "test_empirical_conformal_coverage": test_interval["coverage"],
        "test_mean_interval_width_db": test_interval["mean_width_db"],
        "representative_case_offset": representative,
        "dataset": str(args.dataset.resolve()),
        "validation_predictions": str(args.validation_predictions.resolve()),
        "test_predictions": str(args.test_predictions.resolve()),
        "calibration_json": str(args.calibration_json.resolve()),
        "planning_interpretation": "candidate monitoring sites under reciprocal propagation assumptions",
    }
    if "metadata_json" in dataset:
        try:
            metadata["dataset_metadata"] = json.loads(str(dataset["metadata_json"].item()))
        except (ValueError, TypeError, json.JSONDecodeError):
            metadata["dataset_metadata"] = {"raw": str(dataset["metadata_json"].item())}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema=np.asarray(metadata["schema"]),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        scene=np.asarray(args.scene), seed=np.asarray(args.seed, dtype=np.int64),
        case_indices=test_indices, source_station_index=source_station,
        station_xyz_m=stations,
        x_m=_case_axis(dataset, "x_m", test_indices),
        y_m=_case_axis(dataset, "y_m", test_indices),
        z_m=_case_axis(dataset, "z_m", test_indices),
        terrain_m=_case_surface(dataset, "terrain_m", test_indices),
        surface_m=_case_surface(dataset, "surface_m", test_indices),
        air_mask=mask, coarse_dbm=coarse, target_dbm=target,
        predicted_dbm=mean.astype(np.float32), mean_dbm=mean.astype(np.float32),
        variance_db2=variance, interval_lower_dbm=lower.astype(np.float32),
        interval_upper_dbm=upper.astype(np.float32),
        detection_probability=probability,
        conservative_detection_probability=conservative_probability,
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
