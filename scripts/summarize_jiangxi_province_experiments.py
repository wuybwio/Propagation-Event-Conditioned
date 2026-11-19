"""Summarize province experiments using geographical groups as inference units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


LOWER_IS_BETTER = {
    "mae_db", "rmse_db", "probability_mse", "false_coverage_rate",
}
RESOURCE_METRICS = {"inference_ms_per_case", "peak_gpu_memory_mb", "active_parameter_count"}
HIGHER_IS_BETTER = {
    "coverage_iou", "blind_iou", "balanced_accuracy",
    "coverage_precision", "coverage_recall", "blind_recall",
}
# Preserve the metric order used by the published archive. Iterating sets made
# per-contrast RNG offsets depend on Python hash randomization across processes.
METRIC_ORDER = (
    "false_coverage_rate", "mae_db", "probability_mse", "rmse_db",
    "blind_recall", "balanced_accuracy", "coverage_iou", "coverage_precision",
    "coverage_recall", "blind_iou", "bias_db",
)


def _identity(path: Path) -> tuple[str, int, str]:
    run = path.parents[2].name
    match = re.fullmatch(r"(?P<fold>.+)_seed(?P<seed>\d+)", run)
    if match is None:
        raise ValueError(f"cannot parse fold and seed from {path}")
    return match.group("fold"), int(match.group("seed")), path.parents[1].name


def _bootstrap_ci(values: np.ndarray, seed: int, samples: int = 10_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-variant", default="full_v3")
    parser.add_argument("--evaluation-dir", default="eval_test")
    parser.add_argument(
        "--baseline-glob",
        help="Optional root-relative glob for baseline case_metrics.csv files.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"use a new output directory: {args.output_dir}")
    rows = []
    for path in sorted(args.root.glob(f"*_seed*/*/{args.evaluation_dir}/case_metrics.csv")):
        fold, seed, variant = _identity(path)
        frame = pd.read_csv(path)
        required = {"scene_id", "geo_group", "voxels"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing province identity columns: {missing}")
        frame.insert(0, "fold", fold)
        frame.insert(1, "seed", seed)
        frame.insert(2, "variant", variant)
        rows.append(frame)
    if args.baseline_glob:
        for path in sorted(args.root.glob(args.baseline_glob)):
            match = re.fullmatch(r"(?P<fold>.+)_baselines(?:_independent)?", path.parent.name)
            if match is None:
                raise ValueError(f"cannot parse baseline fold from {path}")
            frame = pd.read_csv(path)
            required = {"variant", "scene_id", "geo_group", "voxels"}
            missing = sorted(required - set(frame.columns))
            if missing:
                raise ValueError(f"{path} is missing baseline columns: {missing}")
            frame.insert(0, "fold", match.group("fold"))
            frame.insert(1, "seed", 0)
            rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"no province case_metrics.csv files under {args.root}")
    raw = pd.concat(rows, ignore_index=True)
    identity = ["fold", "seed", "variant", "geo_group", "scene_id"]
    metrics = [name for name in METRIC_ORDER if name in raw]
    resources = [name for name in sorted(RESOURCE_METRICS) if name in raw]
    scene_seed = raw.groupby(identity, as_index=False)[metrics].mean()
    scene = scene_seed.groupby(["variant", "geo_group", "scene_id"], as_index=False)[metrics].mean()
    overall_rows = []
    for variant, frame in scene.groupby("variant"):
        group_frame = frame.groupby("geo_group", as_index=False)[metrics].mean()
        row: dict[str, object] = {
            "variant": variant,
            "evaluated_scenes": int(frame["scene_id"].nunique()),
            "independent_geographical_groups": int(frame["geo_group"].nunique()),
            "inference_unit": "geographical_group",
        }
        for offset, metric in enumerate(metrics):
            values = frame[metric].to_numpy(float)
            group_values = group_frame[metric].to_numpy(float)
            low, high = _bootstrap_ci(group_values, args.bootstrap_seed + offset)
            row.update({
                f"{metric}_mean": float(np.nanmean(values)),
                f"{metric}_std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else np.nan,
                f"{metric}_group_bootstrap_ci95_low": low,
                f"{metric}_group_bootstrap_ci95_high": high,
            })
        overall_rows.append(row)
    overall = pd.DataFrame(overall_rows)
    reference = scene.loc[scene.variant == args.reference_variant].set_index(
        ["geo_group", "scene_id"]
    )
    if reference.empty:
        raise ValueError(f"reference variant is absent: {args.reference_variant}")
    paired_rows = []
    for variant in sorted(set(scene.variant) - {args.reference_variant}):
        other = scene.loc[scene.variant == variant].set_index(["geo_group", "scene_id"])
        if set(other.index) != set(reference.index):
            raise ValueError(f"scene mismatch between {args.reference_variant} and {variant}")
        for metric in metrics:
            delta = other[metric] - reference[metric]
            if metric in LOWER_IS_BETTER:
                improvement = delta
            elif metric in HIGHER_IS_BETTER:
                improvement = -delta
            elif metric == "bias_db":
                improvement = other[metric].abs() - reference[metric].abs()
            else:
                continue
            grouped_improvement = improvement.groupby(level="geo_group").mean()
            contrast_seed = args.bootstrap_seed + len(paired_rows)
            low, high = _bootstrap_ci(grouped_improvement.to_numpy(float), contrast_seed)
            paired_rows.append({
                "reference_variant": args.reference_variant,
                "variant": variant,
                "metric": metric,
                "evaluated_scenes": len(improvement),
                "independent_geographical_groups": len(grouped_improvement),
                "inference_unit": "geographical_group",
                "bootstrap_seed": contrast_seed,
                "bootstrap_resamples": 10_000,
                "mean_improvement_reference_over_variant": float(improvement.mean()),
                "variant_minus_reference_mean": float(delta.mean()),
                "paired_ci95_low": low,
                "paired_ci95_high": high,
            })
    args.output_dir.mkdir(parents=True)
    raw.to_csv(args.output_dir / "province_case_metrics.csv", index=False, encoding="utf-8-sig")
    scene_seed.to_csv(args.output_dir / "province_scene_seed_metrics.csv", index=False, encoding="utf-8-sig")
    scene.to_csv(args.output_dir / "province_scene_metrics.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(args.output_dir / "province_overall.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(paired_rows).to_csv(
        args.output_dir / "province_paired_ablations.csv", index=False, encoding="utf-8-sig"
    )
    (args.output_dir / "bootstrap_protocol.json").write_text(json.dumps({
        "base_seed": args.bootstrap_seed,
        "replicates": 10_000,
        "paired_seed_rule": "base_seed + zero_based_output_row",
        "metric_order": metrics,
        "comparator_order": sorted(set(scene.variant) - {args.reference_variant}),
        "sampling_unit": "paired_geographical_group",
        "resample_size": "number_of_geographical_groups",
        "replacement": True,
        "interval_percentiles": [0.025, 0.975],
        "quantile_method": "linear",
        "rng": "numpy.random.default_rng",
    },indent=2),encoding="utf-8")
    if resources:
        resource_frame = raw.groupby(["fold", "seed", "variant"], as_index=False)[resources].mean()
        resource_frame.to_csv(
            args.output_dir / "province_resource_metrics.csv",
            index=False, encoding="utf-8-sig",
        )
    print(overall.to_string(index=False))
    print(f"saved: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
