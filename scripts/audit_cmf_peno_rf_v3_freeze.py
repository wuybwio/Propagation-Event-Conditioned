"""Audit the frozen CMF-PENO-RF v3 before ablation and planning experiments.

The script performs two leakage-safe checks:

1. It validates that the LOSO summary contains complete scene-seed units and
   emits descriptive/paired statistics using held-out scenes as the inference
   unit.
2. It fits a scalar mean-bias correction on each run's validation archive,
   applies it once to the test archive, and compares it with the current
   variance-calibrated test prediction.  The checkpoint is never changed.

The bias experiment is a diagnostic, not a new model.  It answers whether the
positive v3 bias is a transferable calibration error or a scene-specific
failure.  All targets remain simulation-derived.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

RUN_PATTERN = re.compile(r"^(?P<scene>.+)_seed(?P<seed>[0-9]+)$")
METRICS = (
    "mae_db", "rmse_db", "bias_db", "probability_mse", "combined_probability_mse",
    "coverage_iou", "blind_iou", "balanced_accuracy", "false_coverage_rate",
    "coverage_95", "mean_interval_width", "standardized_error_rmse",
)
AUDIT_METRICS = METRICS + (
    "conformal_coverage_95", "conformal_mean_interval_width",
)


def _normal_cdf(value: np.ndarray) -> np.ndarray:
    flat = np.asarray(value, dtype=float).ravel()
    return (0.5 * (1.0 + np.fromiter(
        (math.erf(float(item)) for item in flat), dtype=float, count=len(flat)
    ))).reshape(np.asarray(value).shape)


def _detection_probability(power: np.ndarray, threshold: float, shadow: float) -> np.ndarray:
    return _normal_cdf((np.asarray(power, dtype=float) - threshold) / (shadow * math.sqrt(2.0)))


def _combined_detection_probability(
    mean: np.ndarray, variance: np.ndarray, threshold: float, shadow: float
) -> np.ndarray:
    denominator = np.sqrt(shadow**2 + np.maximum(np.asarray(variance, dtype=float), 0.0))
    return _normal_cdf((np.asarray(mean, dtype=float) - threshold) / denominator)


def _field_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    shadow: float,
) -> dict[str, float]:
    valid = np.asarray(mask, dtype=bool)
    predicted = np.asarray(prediction, dtype=float)[valid]
    truth = np.asarray(target, dtype=float)[valid]
    p_pred = _detection_probability(predicted, threshold, shadow)
    p_truth = _detection_probability(truth, threshold, shadow)
    pred_cover = p_pred >= 0.9
    truth_cover = p_truth >= 0.9
    tp = int(np.sum(pred_cover & truth_cover))
    fp = int(np.sum(pred_cover & ~truth_cover))
    fn = int(np.sum(~pred_cover & truth_cover))
    tn = int(np.sum(~pred_cover & ~truth_cover))
    ratio = lambda numerator, denominator: float(numerator / denominator) if denominator else float("nan")
    recall = ratio(tp, tp + fn)
    blind_recall = ratio(tn, tn + fp)
    return {
        "voxels": int(len(predicted)),
        "mae_db": float(np.mean(np.abs(predicted - truth))),
        "rmse_db": float(np.sqrt(np.mean(np.square(predicted - truth)))),
        "bias_db": float(np.mean(predicted - truth)),
        "probability_mse": float(np.mean(np.square(p_pred - p_truth))),
        "coverage_iou": ratio(tp, tp + fp + fn),
        "blind_iou": ratio(tn, tn + fp + fn),
        "balanced_accuracy": (recall + blind_recall) / 2.0,
        "false_coverage_rate": ratio(fp, tn + fp),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        required = {"mean_dbm", "variance_db2", "target_dbm", "mask"}
        missing = sorted(required.difference(source.files))
        if missing:
            raise ValueError(f"{path} is missing fields: {missing}")
        return {name: np.asarray(source[name]) for name in source.files}


def _mean_bias(validation: dict[str, np.ndarray]) -> float:
    mean = np.asarray(validation["mean_dbm"], dtype=float)
    target = np.asarray(validation["target_dbm"], dtype=float)
    mask = np.broadcast_to(np.asarray(validation["mask"], dtype=bool), mean.shape)
    if mean.shape != target.shape or not np.any(mask):
        raise ValueError("validation archive has invalid mean/target/mask alignment")
    value = float(np.mean(target[mask] - mean[mask]))
    if not np.isfinite(value):
        raise ValueError("validation mean bias is non-finite")
    return value


def _uncertainty_metrics_np(
    mean: np.ndarray,
    variance: np.ndarray,
    target: np.ndarray,
    *,
    z_value: float,
) -> dict[str, float]:
    mean = np.asarray(mean, dtype=float)
    variance = np.asarray(variance, dtype=float)
    target = np.asarray(target, dtype=float)
    if mean.shape != variance.shape or mean.shape != target.shape:
        raise ValueError("uncertainty arrays must be aligned")
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or not np.isfinite(target).all():
        raise ValueError("uncertainty arrays must be finite")
    if np.any(variance < 0.0) or not np.isfinite(z_value) or z_value <= 0.0:
        raise ValueError("variance and z value must be valid")
    std = np.sqrt(np.maximum(variance, 1e-6))
    error = target - mean
    return {
        "coverage_95": float(np.mean(np.abs(error) <= z_value * std)),
        "mean_interval_width": float(np.mean(2.0 * z_value * std)),
        "standardized_error_rmse": float(np.sqrt(np.mean(np.square(error / std)))),
        "mean_std_db": float(np.mean(std)),
    }


def _metrics(
    archive: dict[str, np.ndarray],
    *,
    mean_shift_db: float,
    threshold_dbm: float,
    shadow_std_db: float,
) -> dict[str, float]:
    mean = np.asarray(archive["mean_dbm"], dtype=np.float32) + float(mean_shift_db)
    variance = np.asarray(archive["variance_db2"], dtype=np.float32)
    target = np.asarray(archive["target_dbm"], dtype=np.float32)
    mask = np.asarray(archive["mask"], dtype=bool)
    if mean.shape != variance.shape or mean.shape != target.shape or mean.shape != mask.shape:
        raise ValueError("test prediction archive fields are not aligned")
    records = []
    probabilities = []
    target_probabilities = []
    valid_count = 0
    for index in range(mean.shape[0]):
        records.append(_field_metrics(mean[index], target[index], mask[index], threshold_dbm, shadow_std_db))
        probability = _combined_detection_probability(
            mean[index], variance[index], threshold_dbm, shadow_std_db
        )
        target_probability = _detection_probability(target[index], threshold_dbm, shadow_std_db)
        valid = mask[index]
        probabilities.append(float(np.mean(np.square(probability[valid] - target_probability[valid]))))
        target_probabilities.append(int(np.sum(valid)))
        valid_count += int(np.sum(valid))
    if not records or not valid_count:
        raise ValueError("test archive has no valid voxels")
    result: dict[str, float] = {}
    for name in ("mae_db", "bias_db", "probability_mse"):
        result[name] = float(sum(row[name] * row["voxels"] for row in records) / valid_count)
    result["rmse_db"] = float(np.sqrt(sum(row["rmse_db"] ** 2 * row["voxels"] for row in records) / valid_count))
    for name in ("tp", "fp", "fn", "tn"):
        result[name] = float(sum(row[name] for row in records))
    tp, fp, fn, tn = (int(result[name]) for name in ("tp", "fp", "fn", "tn"))
    ratio = lambda numerator, denominator: float(numerator / denominator) if denominator else float("nan")
    recall = ratio(tp, tp + fn)
    blind_recall = ratio(tn, tn + fp)
    result.update({
        "coverage_iou": ratio(tp, tp + fp + fn),
        "blind_iou": ratio(tn, tn + fp + fn),
        "balanced_accuracy": (recall + blind_recall) / 2.0,
        "false_coverage_rate": ratio(fp, tn + fp),
        "combined_probability_mse": float(
            sum(value * count for value, count in zip(probabilities, target_probabilities)) / valid_count
        ),
    })
    uncertainty = _uncertainty_metrics_np(mean[mask], variance[mask], target[mask], z_value=1.96)
    result.update({name: float(value) for name, value in uncertainty.items()})
    return result


def _conformal_multiplier(
    validation: dict[str, np.ndarray],
    *,
    variance_temperature: float,
    mean_shift_db: float,
    target_coverage: float,
) -> float:
    mean = np.asarray(validation["mean_dbm"], dtype=float) + float(mean_shift_db)
    variance = np.asarray(validation["variance_db2"], dtype=float) * float(variance_temperature)
    target = np.asarray(validation["target_dbm"], dtype=float)
    mask = np.broadcast_to(np.asarray(validation["mask"], dtype=bool), mean.shape)
    if not 0.0 < target_coverage < 1.0 or variance_temperature <= 0.0:
        raise ValueError("conformal coverage and variance temperature must be positive")
    scores = np.abs(target[mask] - mean[mask]) / np.sqrt(np.maximum(variance[mask], 1e-6))
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("validation conformal scores are empty or non-finite")
    level = min(1.0, np.ceil((len(scores) + 1) * target_coverage) / len(scores))
    return max(float(np.quantile(scores, level, method="higher")), 1e-6)


def _conformal_test_metrics(
    test: dict[str, np.ndarray],
    *,
    mean_shift_db: float,
    multiplier: float,
) -> tuple[float, float]:
    mean = np.asarray(test["mean_dbm"], dtype=np.float32) + float(mean_shift_db)
    variance = np.asarray(test["variance_db2"], dtype=np.float32)
    target = np.asarray(test["target_dbm"], dtype=np.float32)
    mask = np.asarray(test["mask"], dtype=bool)
    metrics = _uncertainty_metrics_np(
        mean[mask], variance[mask], target[mask], z_value=float(multiplier)
    )
    return float(metrics["coverage_95"]), float(metrics["mean_interval_width"])


def _bootstrap(values: np.ndarray, seed: int = 2026, repetitions: int = 10000) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[samples].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _summary_audit(path: Path, output_dir: Path) -> None:
    frame = pd.read_csv(path)
    required = {"scene", "seed", "variant"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"summary is missing columns: {missing}")
    key = ["scene", "seed", "variant"]
    if frame.duplicated(key).any():
        raise ValueError("summary contains duplicate scene-seed-variant rows")
    variants = sorted(frame["variant"].astype(str).unique())
    inventory = {
        "rows": int(len(frame)),
        "scenes": sorted(frame["scene"].astype(str).unique()),
        "seeds": sorted(int(value) for value in frame["seed"].unique()),
        "variants": variants,
    }
    if "full_v3" not in variants:
        raise ValueError("summary must contain full_v3")
    rows = []
    for variant in variants:
        subset = frame[frame["variant"].astype(str) == variant]
        for metric in METRICS:
            if metric not in subset:
                continue
            run_values = pd.to_numeric(subset[metric], errors="coerce")
            by_scene = pd.DataFrame({"scene": subset["scene"], "value": run_values}).dropna()
            scene_values = by_scene.groupby("scene")["value"].mean().to_numpy(float)
            values = run_values.dropna().to_numpy(float)
            low, high = _bootstrap(scene_values)
            rows.append({
                "variant": variant, "metric": metric,
                "scene_seed_rows": len(values), "independent_scenes": len(scene_values),
                "run_mean": float(np.mean(values)) if len(values) else np.nan,
                "run_std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "scene_mean": float(np.mean(scene_values)) if len(scene_values) else np.nan,
                "scene_std": float(np.std(scene_values, ddof=1)) if len(scene_values) > 1 else np.nan,
                "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
            })
    descriptive = pd.DataFrame(rows)
    descriptive.to_csv(output_dir / "summary_descriptive.csv", index=False, encoding="utf-8-sig")
    (output_dir / "summary_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print("=== LOSO inventory ===")
    print(json.dumps(inventory, ensure_ascii=False, indent=2))
    selected = descriptive[
        (descriptive["variant"] == "full_v3")
        & descriptive["metric"].isin(["mae_db", "rmse_db", "coverage_iou", "blind_iou", "false_coverage_rate", "coverage_95"])
    ]
    print("=== Full v3 scene-level descriptive statistics ===")
    print(selected.to_string(index=False))


def _calibration_audit(
    root: Path,
    output_dir: Path,
    threshold: float,
    shadow: float,
    target_coverage: float,
) -> None:
    rows = []
    for run_dir in sorted(root.glob("*_seed*/full_v3")):
        match = RUN_PATTERN.match(run_dir.parent.name)
        if match is None:
            continue
        validation_path = run_dir / "eval_val" / "predictions.npz"
        test_path = run_dir / "eval_test" / "predictions.npz"
        if not validation_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"missing val/test predictions under {run_dir}")
        validation = _archive(validation_path)
        test = _archive(test_path)
        calibration_path = run_dir / "variance_calibration.json"
        if not calibration_path.exists():
            raise FileNotFoundError(f"missing variance calibration: {calibration_path}")
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        temperature = float(calibration.get("temperature", float("nan")))
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(f"invalid variance temperature: {calibration_path}")
        metrics_path = run_dir / "eval_test" / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"missing test metadata: {metrics_path}")
        test_metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
        applied_temperature = float(test_metadata.get("variance_temperature", float("nan")))
        if not np.isclose(applied_temperature, temperature, rtol=1e-6, atol=1e-8):
            raise ValueError(
                f"test archive is not calibrated with {calibration_path}: "
                f"metadata temperature={applied_temperature}, expected={temperature}"
            )
        bias = _mean_bias(validation)
        raw = _metrics(test, mean_shift_db=0.0, threshold_dbm=threshold, shadow_std_db=shadow)
        corrected = _metrics(test, mean_shift_db=bias, threshold_dbm=threshold, shadow_std_db=shadow)
        policies = (("current_test", 0.0, raw), ("val_bias_corrected", bias, corrected))
        for policy, shift, metrics in policies:
            multiplier = _conformal_multiplier(
                validation,
                variance_temperature=temperature,
                mean_shift_db=shift,
                target_coverage=target_coverage,
            )
            conformal_coverage, conformal_width = _conformal_test_metrics(
                test, mean_shift_db=shift, multiplier=multiplier
            )
            rows.append({
                "scene": match.group("scene"), "seed": int(match.group("seed")),
                "policy": policy, "validation_bias_db": bias,
                "variance_temperature": temperature,
                "conformal_multiplier": multiplier,
                "conformal_coverage_95": conformal_coverage,
                "conformal_mean_interval_width": conformal_width,
                **metrics,
            })
    if not rows:
        raise FileNotFoundError(f"no *_seed*/full_v3 predictions under {root}")
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "bias_calibration_raw.csv", index=False, encoding="utf-8-sig")
    overall = frame.groupby("policy")[list(AUDIT_METRICS)].agg(["mean", "std", "count"])
    overall.columns = ["_".join(str(part) for part in column if str(part)) for column in overall.columns]
    overall_frame = overall.reset_index()
    overall_frame.to_csv(output_dir / "bias_calibration_overall.csv", index=False, encoding="utf-8-sig")
    current = frame[frame["policy"] == "current_test"].set_index(["scene", "seed"])
    corrected = frame[frame["policy"] == "val_bias_corrected"].set_index(["scene", "seed"])
    paired = []
    for metric in AUDIT_METRICS:
        if metric not in current or metric not in corrected:
            continue
        delta = corrected[metric] - current[metric]
        scene_seed_values = delta.dropna().to_numpy(float)
        scene_values = delta.dropna().groupby(level="scene").mean().to_numpy(float)
        low, high = _bootstrap(scene_values)
        paired.append({
            "metric": metric, "scene_seed_rows": len(scene_seed_values),
            "independent_scenes": len(scene_values),
            "mean_delta_corrected_minus_current": float(np.mean(scene_values)) if len(scene_values) else np.nan,
            "ci95_low": low, "ci95_high": high,
        })
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(output_dir / "bias_calibration_paired.csv", index=False, encoding="utf-8-sig")
    display_metrics = [
        "mae_db", "rmse_db", "bias_db", "coverage_iou", "blind_iou",
        "false_coverage_rate", "coverage_95", "conformal_coverage_95",
        "mean_interval_width", "conformal_mean_interval_width",
    ]
    display = frame.groupby("policy")[display_metrics].mean().reset_index()
    print("=== Validation-fitted calibration diagnostic ===")
    print(display.to_string(index=False))
    print("=== Bias correction paired deltas: corrected - current ===")
    print(paired_frame[paired_frame["metric"].isin(display_metrics)].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-dbm", type=float, default=-100.0)
    parser.add_argument("--shadow-std-db", type=float, default=6.0)
    parser.add_argument("--target-coverage", type=float, default=0.95)
    args = parser.parse_args()
    if (
        not np.isfinite(args.threshold_dbm)
        or not np.isfinite(args.shadow_std_db)
        or args.shadow_std_db <= 0
        or not 0.0 < args.target_coverage < 1.0
    ):
        parser.error("threshold must be finite; shadow std and target coverage must be valid")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _summary_audit(args.summary, args.output_dir)
    _calibration_audit(
        args.root, args.output_dir, args.threshold_dbm, args.shadow_std_db, args.target_coverage
    )
    metadata = {
        "summary": str(args.summary.resolve()), "root": str(args.root.resolve()),
        "threshold_dbm": args.threshold_dbm, "shadow_std_db": args.shadow_std_db,
        "target_coverage": args.target_coverage,
        "bias_fit_split": "val", "bias_applied_split": "test",
        "inference_unit": "held_out_scene; seeds are nested repeats",
        "conformal_fit_split": "val", "conformal_applied_split": "test",
        "simulation_only": True,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("saved:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
