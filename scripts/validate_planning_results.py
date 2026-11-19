"""Create a quantitative baseline-versus-optimized planning validation table."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    frame = pd.read_csv(args.summary)
    required = {
        "feasible",
        "height_m",
        "mean_baseline_coverage",
        "mean_optimized_coverage",
        "mean_baseline_gap_km",
        "mean_optimized_gap_km",
        "total_baseline_gap_km",
        "total_optimized_gap_km",
        "total_cost",
        "selected_actions",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"planning summary is missing columns: {missing}")

    result = frame[[
        "height_m", "feasible", "total_cost", "selected_actions",
        "mean_baseline_coverage", "mean_optimized_coverage",
        "mean_baseline_gap_km", "mean_optimized_gap_km",
        "total_baseline_gap_km", "total_optimized_gap_km",
    ]].copy()
    result["coverage_gain"] = (
        result["mean_optimized_coverage"] - result["mean_baseline_coverage"]
    )
    result["mean_gap_reduction_km"] = (
        result["mean_baseline_gap_km"] - result["mean_optimized_gap_km"]
    )
    result["total_gap_reduction_km"] = (
        result["total_baseline_gap_km"] - result["total_optimized_gap_km"]
    )
    result["coverage_non_decreasing"] = result["coverage_gain"] >= -args.tolerance
    result["gap_non_increasing"] = result["total_gap_reduction_km"] >= -args.tolerance
    result["validation_pass"] = (
        result["feasible"].astype(bool)
        & result["coverage_non_decreasing"]
        & result["gap_non_increasing"]
    )

    numeric = result.select_dtypes(include=[np.number]).to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("planning validation contains non-finite values")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.sort_values("height_m").to_csv(args.output, index=False, encoding="utf-8-sig")
    print(result.sort_values("height_m").to_string(index=False))
    print("saved:", args.output.resolve())
    if not result["validation_pass"].all():
        failed = result.loc[~result["validation_pass"], "height_m"].tolist()
        raise SystemExit(f"planning validation failed at heights: {failed}")


if __name__ == "__main__":
    main()
