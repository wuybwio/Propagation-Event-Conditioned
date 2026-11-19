"""Summarize province planning with scenes nested in geographical groups."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd


METRICS = (
    "mean_detection_probability",
    "coverage_fraction",
    "blind_fraction",
    "minimum_detection_probability",
)


def _bootstrap(values: np.ndarray, seed: int, samples: int = 10_000) -> tuple[float, float]:
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
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"use a new output directory: {args.output_dir}")
    rows = []
    for path in sorted(args.root.glob("*_seed*/planning_results_raw.csv")):
        match = re.fullmatch(r"(?P<fold>.+)_seed(?P<seed>\d+)", path.parent.name)
        if match is None:
            raise ValueError(f"cannot parse planning run: {path}")
        frame = pd.read_csv(path)
        frame.insert(0, "fold", match.group("fold"))
        frame.insert(1, "seed", int(match.group("seed")))
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"no planning_results_raw.csv under {args.root}")
    raw = pd.concat(rows, ignore_index=True)
    required = {
        "planning_scope", "planning_unit", "method", "budget", "repeat",
        "scene_id", "geo_group", *METRICS,
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"planning results are missing columns: {missing}")

    # Random repetitions estimate one randomized policy and are averaged before
    # seeds, scenes, or geographical groups enter the statistical summary.
    run_scene = raw.groupby(
        ["fold", "seed", "planning_scope", "planning_unit", "geo_group", "scene_id", "method", "budget"],
        as_index=False,
    )[list(METRICS)].mean()
    scene = run_scene.groupby(
        ["planning_scope", "planning_unit", "geo_group", "scene_id", "method", "budget"], as_index=False
    )[list(METRICS)].mean()

    rows_out = []
    for (scope, method, budget), frame in scene.groupby(["planning_scope", "method", "budget"]):
        groups = frame.groupby("geo_group", as_index=False)[list(METRICS)].mean()
        record: dict[str, object] = {
            "method": method,
            "planning_scope": scope,
            "budget": int(budget),
            "evaluated_scenes": int(frame.scene_id.nunique()),
            "independent_geographical_groups": int(frame.geo_group.nunique()),
            "inference_unit": "geographical_group",
        }
        for offset, metric in enumerate(METRICS):
            low, high = _bootstrap(
                groups[metric].to_numpy(float), args.bootstrap_seed + offset
            )
            record[f"{metric}_mean"] = float(frame[metric].mean())
            record[f"{metric}_std"] = float(frame[metric].std(ddof=1))
            record[f"{metric}_group_bootstrap_ci95_low"] = low
            record[f"{metric}_group_bootstrap_ci95_high"] = high
        rows_out.append(record)
    overall = pd.DataFrame(rows_out)

    exact = scene.loc[scene.method == "ExactReference"].set_index(
        ["planning_scope", "geo_group", "scene_id", "budget"]
    )
    gaps = []
    if not exact.empty:
        for method in sorted(set(scene.method) - {"ExactReference"}):
            current = scene.loc[scene.method == method].set_index(
                ["planning_scope", "geo_group", "scene_id", "budget"]
            )
            common = exact.index.intersection(current.index)
            for index in common:
                gaps.append({
                    "method": method,
                    "planning_scope": index[0],
                    "geo_group": index[1],
                    "scene_id": index[2],
                    "budget": int(index[3]),
                    "mean_probability_optimality_gap": float(
                        exact.loc[index, "mean_detection_probability"]
                        - current.loc[index, "mean_detection_probability"]
                    ),
                    "coverage_fraction_difference_at_mean_optimum": float(
                        exact.loc[index, "coverage_fraction"]
                        - current.loc[index, "coverage_fraction"]
                    ),
                })

    args.output_dir.mkdir(parents=True)
    raw.to_csv(args.output_dir / "planning_raw.csv", index=False, encoding="utf-8-sig")
    run_scene.to_csv(args.output_dir / "planning_run_scene.csv", index=False, encoding="utf-8-sig")
    scene.to_csv(args.output_dir / "planning_scene.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(args.output_dir / "planning_overall.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(gaps).to_csv(
        args.output_dir / "planning_exact_optimality_gaps.csv",
        index=False, encoding="utf-8-sig",
    )
    print(overall.to_string(index=False))
    print(f"saved: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
