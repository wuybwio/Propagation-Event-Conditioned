"""Summarize CMF-PENO-RF ablation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = (
    "mae_db", "rmse_db", "bias_db", "probability_mse", "coverage_iou",
    "blind_iou", "balanced_accuracy", "false_coverage_rate",
    "coverage_95", "mean_interval_width", "standardized_error_rmse",
    "combined_probability_mse",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    rows = []
    for path in sorted(args.run_root.glob(f"*/eval_{args.split}/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("teacher_components_used_as_model_input", True):
            raise ValueError(f"refusing legacy/leaky result: {path}")
        if payload.get("split") != args.split:
            raise ValueError(f"split metadata mismatch: {path}")
        row = {"variant": path.parts[-3], "split": payload["split"]}
        row.update(payload.get("metrics", {}))
        rows.append(row)
    if not rows:
        raise FileNotFoundError(
            f"no eval_{args.split}/metrics.json under {args.run_root}"
        )
    frame = pd.DataFrame(rows)
    columns = ["variant", "split"] + [name for name in METRICS if name in frame.columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame[columns].to_csv(args.output, index=False, encoding="utf-8-sig")
    print(frame[columns].to_string(index=False))
    print("saved:", args.output)


if __name__ == "__main__":
    main()
