"""Run direction-aware paired statistics for CMF-PENO-RF LOSO results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    "mae_db",
    "rmse_db",
    "bias_db",
    "probability_mse",
    "combined_probability_mse",
    "coverage_iou",
    "blind_iou",
    "balanced_accuracy",
    "false_coverage_rate",
    "coverage_95",
    "mean_interval_width",
    "standardized_error_rmse",
)
HIGHER_IS_BETTER = {"coverage_iou", "blind_iou", "balanced_accuracy"}
# Calibration metrics are losses around a target, not monotone rewards:
# coverage should be nominal and standardized residual RMSE should be one.
TARGET_VALUE = {
    "coverage_95": 0.95,
    "standardized_error_rmse": 1.0,
}


def _score(metric: str, values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    if metric in HIGHER_IS_BETTER:
        return -values
    if metric in TARGET_VALUE:
        return (values - TARGET_VALUE[metric]).abs()
    if metric == "bias_db":
        return values.abs()
    return values


def _bootstrap_ci(values: np.ndarray, seed: int = 2026) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_test(values: np.ndarray) -> tuple[float, float]:
    """Return Wilcoxon p-value and rank-biserial effect size when available."""
    try:
        from scipy.stats import rankdata, wilcoxon
    except ImportError:
        return float("nan"), float("nan")
    nonzero = values[values != 0.0]
    if len(nonzero) == 0:
        return 1.0, 0.0
    try:
        p_value = float(wilcoxon(nonzero, alternative="two-sided").pvalue)
    except ValueError:
        p_value = float("nan")
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0.0].sum())
    negative = float(ranks[nonzero < 0.0].sum())
    denominator = positive + negative
    effect = float((positive - negative) / denominator) if denominator else 0.0
    return p_value, effect


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-variant", default="detached_nll")
    parser.add_argument(
        "--expected-units", type=int, default=0,
        help="required scene-seed row count; 0 accepts the complete count present",
    )
    parser.add_argument(
        "--expected-scenes", type=int, default=0,
        help="required independent held-out scene count; 0 accepts the count present",
    )
    parser.add_argument(
        "--inference-unit",
        choices=("scene", "scene_seed"),
        default="scene",
        help=(
            "unit used by bootstrap and Wilcoxon tests; scene is the valid default "
            "because random seeds are repeated runs within a held-out scene"
        ),
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.summary)
    required = {"scene", "seed", "variant"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"summary is missing required columns: {missing}")
    keys = ["scene", "seed", "variant"]
    if frame.duplicated(keys).any():
        raise ValueError("summary contains duplicate scene-seed-variant rows")
    reference = frame[frame["variant"] == args.reference_variant].copy()
    if reference.empty:
        raise ValueError(f"reference variant is absent: {args.reference_variant}")
    reference_keys = set(zip(reference["scene"], reference["seed"]))
    reference_scenes = set(reference["scene"])
    if args.expected_units > 0 and len(reference_keys) != args.expected_units:
        raise ValueError(
            f"reference variant has {len(reference_keys)} scene-seed units; "
            f"expected {args.expected_units}"
        )
    if args.expected_scenes > 0 and len(reference_scenes) != args.expected_scenes:
        raise ValueError(
            f"reference variant has {len(reference_scenes)} held-out scenes; "
            f"expected {args.expected_scenes}"
        )

    rows: list[dict[str, object]] = []
    for variant in sorted(set(frame["variant"]) - {args.reference_variant}):
        candidate = frame[frame["variant"] == variant].copy()
        candidate_keys = set(zip(candidate["scene"], candidate["seed"]))
        if candidate_keys != reference_keys:
            raise ValueError(
                f"scene-seed mismatch for {variant}: "
                f"missing={sorted(reference_keys - candidate_keys)}, "
                f"extra={sorted(candidate_keys - reference_keys)}"
            )
        left = reference.set_index(["scene", "seed"])
        right = candidate.set_index(["scene", "seed"])
        joined = left.join(right, lsuffix="_reference", rsuffix="_candidate", how="inner")
        for metric in METRICS:
            left_name = f"{metric}_reference"
            right_name = f"{metric}_candidate"
            if left_name not in joined or right_name not in joined:
                continue
            valid = joined[[left_name, right_name]].notna().all(axis=1)
            if not valid.any():
                continue
            improvement_by_scene_seed = (
                _score(metric, joined.loc[valid, left_name])
                - _score(metric, joined.loc[valid, right_name])
            )
            if args.inference_unit == "scene":
                improvement_series = improvement_by_scene_seed.groupby(level="scene").mean()
            else:
                improvement_series = improvement_by_scene_seed
            improvement = improvement_series.to_numpy(float)
            ci_low, ci_high = _bootstrap_ci(improvement)
            p_value, effect = _paired_test(improvement)
            rows.append({
                "reference_variant": args.reference_variant,
                "variant": variant,
                "metric": metric,
                "inference_unit": args.inference_unit,
                "scene_seed_rows": int(valid.sum()),
                "paired_units": int(len(improvement)),
                "mean_improvement": float(np.mean(improvement)),
                "std_improvement": float(np.std(improvement, ddof=1)) if len(improvement) > 1 else 0.0,
                "median_improvement": float(np.median(improvement)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "better_count": int(np.sum(improvement > 0.0)),
                "wilcoxon_p": p_value,
                "rank_biserial": effect,
            })

    if not rows:
        raise ValueError("no comparable metric rows found")
    result = pd.DataFrame(rows)
    finite_p = result["wilcoxon_p"].notna()
    result["wilcoxon_p_holm"] = np.nan
    if finite_p.any():
        ordered = result.loc[finite_p, "wilcoxon_p"].sort_values()
        adjusted = np.maximum.accumulate(
            np.minimum(ordered.to_numpy(float) * np.arange(len(ordered), 0, -1), 1.0)
        )
        result.loc[ordered.index, "wilcoxon_p_holm"] = adjusted
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "paired_stats.csv", index=False, encoding="utf-8-sig")
    overall_metrics = [metric for metric in METRICS if metric in frame.columns]
    overall = frame.groupby("variant")[overall_metrics].agg(["mean", "std", "count"])
    overall.columns = ["_".join(str(part) for part in column if str(part)) for column in overall.columns]
    overall.reset_index().to_csv(args.output_dir / "overall.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "summary": str(args.summary.resolve()),
        "reference_variant": args.reference_variant,
        "scene_seed_units": len(reference_keys),
        "held_out_scenes": len(reference_scenes),
        "inference_unit": args.inference_unit,
        "inference_note": (
            "Random seeds are repeated runs nested within each held-out scene. "
            "For scene-level inference, seed-wise paired improvements are averaged "
            "within scene before bootstrap confidence intervals and Wilcoxon tests."
        ),
        "multiple_comparison_correction": "Holm across all reported paired Wilcoxon tests",
        "direction_rule": {
            "higher_is_better": sorted(HIGHER_IS_BETTER),
            "absolute_bias": True,
            "target_values": TARGET_VALUE,
        },
        "simulation_only": True,
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False))
    print("saved:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
