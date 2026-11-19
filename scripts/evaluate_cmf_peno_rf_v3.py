"""Evaluate a CMF-PENO-RF v3 checkpoint on one split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from jiangxi_radio.cmf_peno import combined_detection_probability
from jiangxi_radio.cmf_peno_data import load_cmf_peno_rf_dataset
from jiangxi_radio.cmf_peno_training import uncertainty_interval_metrics
from jiangxi_radio.cmf_peno_v3 import CMFPENOV3
from jiangxi_radio.uncertainty_calibration import apply_temperature, load_temperature, masked_probability_mse
from jiangxi_radio.volume_simulation import input_volume, sha256_file
from jiangxi_radio.volume_surrogate import detection_probability, field_metrics


def _device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else value if value != "auto" else "cpu")


def _aggregate(records):
    total = sum(int(record["voxels"]) for record in records)
    result = {"voxels": total}
    for name in ("mae_db", "bias_db", "probability_mse"):
        result[name] = float(sum(float(record[name]) * record["voxels"] for record in records) / total)
    result["rmse_db"] = float(np.sqrt(sum(float(record["rmse_db"]) ** 2 * record["voxels"] for record in records) / total))
    for name in ("tp", "fp", "fn", "tn"):
        result[name] = int(sum(int(record[name]) for record in records))
    tp, fp, fn, tn = (result[name] for name in ("tp", "fp", "fn", "tn"))
    ratio = lambda numerator, denominator: float(numerator / denominator) if denominator else None
    recall = ratio(tp, tp + fn)
    blind_recall = ratio(tn, tn + fp)
    result.update({
        "coverage_iou": ratio(tp, tp + fp + fn),
        "coverage_precision": ratio(tp, tp + fp),
        "coverage_recall": recall,
        "blind_iou": ratio(tn, tn + fp + fn),
        "blind_recall": blind_recall,
        "false_coverage_rate": ratio(fp, tn + fp),
        "balanced_accuracy": (recall + blind_recall) / 2 if recall is not None and blind_recall is not None else None,
    })
    return result


def _build_model(checkpoint, input_channels):
    arguments = checkpoint.get("arguments", {})
    modes = tuple(int(value) for value in arguments.get("modes", [4, 6, 6]))
    model = CMFPENOV3(
        input_channels=int(checkpoint.get("input_channels", input_channels)),
        width=int(arguments.get("width", 24)),
        event_hidden=int(arguments.get("event_hidden", 32)),
        event_feature_count=int(checkpoint.get("event_feature_count", 14)),
        modes=modes,
        use_events=bool(checkpoint.get("variant_spec", {}).get("use_events", True)),
        use_local=bool(checkpoint.get("variant_spec", {}).get("use_local", True)),
        use_spectral=bool(checkpoint.get("variant_spec", {}).get("use_spectral", True)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "val", "calibration", "test"), default="val"
    )
    parser.add_argument("--threshold-dbm", type=float, default=-100.0)
    parser.add_argument("--shadow-std-db", type=float, default=6.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument(
        "--allow-cross-dataset", action="store_true",
        help="Evaluate a frozen checkpoint on a compatible external dataset and record the transfer.",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"use a new output directory: {args.output_dir}")
    data = load_cmf_peno_rf_dataset(args.dataset)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "cmf_peno_model_v3":
        raise ValueError("checkpoint is not a CMF-PENO-RF v3 model")
    dataset_hash_matches = (
        not checkpoint.get("dataset_sha256")
        or checkpoint["dataset_sha256"] == sha256_file(args.dataset)
    )
    if not dataset_hash_matches and not args.allow_cross_dataset:
        raise ValueError("checkpoint dataset hash does not match evaluation dataset")
    split_id = {"train": 0, "val": 1, "test": 2, "calibration": 3}[args.split]
    indices = np.flatnonzero(np.asarray(data["split"]) == split_id)
    if not len(indices):
        raise ValueError(f"split {args.split} is empty")
    device = _device(args.device)
    model = _build_model(checkpoint, input_volume(data, int(indices[0])).shape[0]).to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    inference_seconds = 0.0
    predictions, variances, stages, targets, masks = [], [], [], [], []
    records = []
    with torch.no_grad():
        for index in indices:
            index = int(index)
            volume = torch.from_numpy(input_volume(data, index)[None]).float().to(device)
            coarse = torch.from_numpy(np.asarray(data["coarse_dbm"][index][None, None])).float().to(device)
            events = torch.from_numpy(np.asarray(data["event_features"][index][None])).float().to(device)
            event_type = torch.from_numpy(np.asarray(data["event_type"][index][None])).long().to(device)
            event_mask = torch.from_numpy(np.asarray(data["event_mask"][index][None])).bool().to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_start = time.perf_counter()
            output = model(volume, coarse, events, event_type, event_mask)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - forward_start
            prediction = output["mean_dbm"].cpu().numpy()[0, 0]
            variance = output["variance_db2"].cpu().numpy()[0, 0]
            stage = output["stage1_dbm"].cpu().numpy()[0, 0]
            target = np.asarray(data["fidelity_dbm"][index, -1], dtype=np.float32)
            air = np.asarray(data["air_mask"])
            mask = np.asarray(air[index] if air.ndim == 4 else air, dtype=bool)
            record = field_metrics(
                prediction, target, mask, args.threshold_dbm, args.shadow_std_db
            )
            record["case_index"] = index
            if "scene_id" in data:
                record["scene_id"] = str(np.asarray(data["scene_id"])[index])
            if "geo_group" in data:
                record["geo_group"] = str(np.asarray(data["geo_group"])[index])
            if "source_station_index" in data:
                record["source_station_index"] = int(
                    np.asarray(data["source_station_index"])[index]
                )
            records.append(record)
            predictions.append(prediction)
            variances.append(variance)
            stages.append(stage)
            targets.append(target)
            masks.append(mask)
    inference_ms_per_case = 1000.0 * inference_seconds / len(indices)
    peak_gpu_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
        if device.type == "cuda" else None
    )
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    active_parameter_count = int(sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    ))
    for record in records:
        record["inference_ms_per_case"] = inference_ms_per_case
        record["parameter_count"] = parameter_count
        record["active_parameter_count"] = active_parameter_count
        record["peak_gpu_memory_mb"] = peak_gpu_memory_mb
    metrics = _aggregate(records)
    prediction_array = np.asarray(predictions, dtype=np.float32)
    variance_array = np.asarray(variances, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    mask_array = np.asarray(masks, dtype=bool)
    uncertainty_applicable = bool(
        checkpoint.get("variant_spec", {}).get("use_uncertainty", True)
    )
    if not uncertainty_applicable:
        variance_array.fill(0.0)
    temperature = 1.0
    if args.calibration_json is not None:
        if not uncertainty_applicable:
            raise ValueError("deterministic checkpoints cannot use variance calibration")
        temperature = load_temperature(args.calibration_json)
        variance_array = apply_temperature(variance_array, temperature)
    if uncertainty_applicable:
        valid = torch.from_numpy(mask_array)
        uncertainty = uncertainty_interval_metrics(
            torch.from_numpy(prediction_array)[valid],
            torch.from_numpy(variance_array)[valid],
            torch.from_numpy(target_array)[valid],
        )
        metrics.update(uncertainty)
    else:
        metrics.update({
            "coverage_95": None,
            "mean_interval_width": None,
            "standardized_error_rmse": None,
            "mean_std_db": None,
        })
    metrics["uncertainty_applicable"] = uncertainty_applicable
    metrics["inference_seconds"] = inference_seconds
    metrics["inference_ms_per_case"] = inference_ms_per_case
    metrics["parameter_count"] = parameter_count
    metrics["active_parameter_count"] = active_parameter_count
    metrics["peak_gpu_memory_mb"] = peak_gpu_memory_mb
    probability = combined_detection_probability(prediction_array, variance_array, args.threshold_dbm, args.shadow_std_db)
    target_probability = detection_probability(target_array, args.threshold_dbm, args.shadow_std_db)
    probability_values = [masked_probability_mse(probability[i], target_probability[i], mask_array[i]) for i in range(len(indices))]
    counts = [int(mask.sum()) for mask in mask_array]
    metrics["combined_probability_mse"] = float(sum(value * count for value, count in zip(probability_values, counts)) / sum(counts))
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        case_indices=np.asarray(indices, dtype=np.int64), split=np.asarray(args.split),
        mean_dbm=prediction_array, variance_db2=variance_array,
        stage1_dbm=np.asarray(stages, dtype=np.float32), target_dbm=target_array,
        detection_probability=probability.astype(np.float32), mask=mask_array,
    )
    pd.DataFrame(records).to_csv(
        args.output_dir / "case_metrics.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "schema": "cmf_peno_evaluation_v3",
        "evidence": "simulation_teacher_agreement_not_measurement_accuracy",
        "dataset": str(args.dataset.resolve()), "dataset_sha256": sha256_file(args.dataset),
        "checkpoint": str(args.checkpoint.resolve()), "split": args.split,
        "teacher_components_used_as_model_input": False,
        "variance_temperature": temperature,
        "cross_dataset_evaluation": not dataset_hash_matches,
        "calibration_json": str(args.calibration_json.resolve()) if args.calibration_json else None,
        "device": str(device), "metrics": metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "split": args.split, **metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
