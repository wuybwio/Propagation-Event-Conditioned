"""Combine held-out fold timing records and write paper-ready summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.root.glob("region_*/per_case.csv"))
    if len(paths) != 5:
        raise ValueError(f"expected five fold records under {args.root}, found {len(paths)}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame.insert(0, "fold", path.parent.name)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.root / "all_cases.csv", index=False)
    fields = [name for name in combined.columns if name.endswith("_ms")]
    summary = pd.DataFrame({
        "metric": fields,
        "mean_ms": [combined[name].mean() for name in fields],
        "sd_ms": [combined[name].std(ddof=1) for name in fields],
        "median_ms": [combined[name].median() for name in fields],
        "n_cases": len(combined),
        "n_scenes": combined.scene_id.nunique(),
    })
    summary.to_csv(args.root / "all_summary.csv", index=False)
    scene_means = combined.groupby("scene_id", as_index=False)[fields].mean()
    scene_summary = pd.DataFrame({
        "metric": fields,
        "mean_ms": [scene_means[name].mean() for name in fields],
        "sd_ms": [scene_means[name].std(ddof=1) for name in fields],
        "median_ms": [scene_means[name].median() for name in fields],
        "n_scenes": scene_means.scene_id.nunique(),
    })
    scene_summary.to_csv(args.root / "scene_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nScene-level summary:")
    print(scene_summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"max reference reconstruction difference: {combined.reference_max_abs_diff_db.max():.6g} dB")


if __name__ == "__main__":
    main()
