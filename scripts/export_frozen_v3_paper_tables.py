"""Export final CMF-PENO-RF v3 main, calibration and planning tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SIGNAL_METRICS = (
    "mae_db", "rmse_db", "bias_db", "probability_mse", "combined_probability_mse",
    "coverage_iou", "blind_iou", "balanced_accuracy", "false_coverage_rate",
    "coverage_95", "mean_interval_width", "standardized_error_rmse",
    "conformal_coverage_95", "conformal_mean_interval_width",
)


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty input: {path}")
    return frame


def _summary_table(path: Path) -> pd.DataFrame:
    frame = _read(path)
    if {"variant", "scene", "seed"}.issubset(frame.columns):
        metrics = [name for name in SIGNAL_METRICS if name in frame.columns]
        result = frame.groupby("variant", as_index=False)[metrics].agg(["mean", "std", "count"])
        result.columns = ["variant"] + ["{}_{}".format(metric, stat) for metric, stat in result.columns.tolist()[1:]]
        return result
    if "variant" in frame.columns and any(name.endswith("_mean") for name in frame.columns):
        return frame.copy()
    raise ValueError(f"summary must contain scene/seed/variant rows or variant aggregate rows: {path}")


def _write(frame: pd.DataFrame, output_dir: Path, stem: str) -> None:
    csv_path = output_dir / f"{stem}.csv"
    tex_path = output_dir / f"{stem}.tex"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    def latex_value(value: object) -> str:
        if pd.isna(value):
            return "--"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("_", "\\_").replace("%", "\\%")
    alignment = "l" + "r" * max(0, len(frame.columns) - 1)
    lines = [f"\\begin{{tabular}}{{{alignment}}}", "\\toprule"]
    lines.append(" & ".join(str(name).replace("_", "\\_") for name in frame.columns) + " \\\\")
    lines.append("\\midrule")
    for row in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(latex_value(value) for value in row) + " \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(frame.to_string(index=False)); print("saved:", csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--calibration-overall", type=Path)
    parser.add_argument("--calibration-raw", type=Path)
    parser.add_argument("--planning-results", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"use a new output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    raw_table = _summary_table(args.summary)
    raw_table.insert(0, "policy", "uncorrected_full_v3")
    _write(raw_table, args.output_dir, "table_uncorrected_v3_trace")
    if args.calibration_raw is not None:
        calibrated = _read(args.calibration_raw)
        required = {"scene", "seed", "policy"}
        if not required.issubset(calibrated.columns):
            raise ValueError(f"{args.calibration_raw} is missing {sorted(required - set(calibrated.columns))}")
        calibrated = calibrated[calibrated["policy"] == "val_bias_corrected"].copy()
        if calibrated.empty:
            raise ValueError("calibration raw table has no val_bias_corrected rows")
        metrics = [name for name in SIGNAL_METRICS if name in calibrated.columns]
        main_table = calibrated.groupby("policy", as_index=False)[metrics].agg(["mean", "std", "count"])
        main_table.columns = ["policy"] + [f"{metric}_{stat}" for metric, stat in main_table.columns.tolist()[1:]]
        by_scene = calibrated.groupby(["scene", "policy"], as_index=False)[metrics].agg(["mean", "std", "count"])
        by_scene.columns = ["scene", "policy"] + [f"{metric}_{stat}" for metric, stat in by_scene.columns.tolist()[2:]]
        _write(main_table, args.output_dir, "table_main_frozen_v3")
        _write(by_scene, args.output_dir, "table_main_frozen_v3_by_scene")
    else:
        main_table = raw_table.copy()
        main_table["policy"] = "uncorrected_full_v3"
        _write(main_table, args.output_dir, "table_main_frozen_v3")
    if args.calibration_overall is not None:
        calibration = _read(args.calibration_overall)
        _write(calibration, args.output_dir, "table_calibration_policy")
    if args.planning_results:
        frames = []
        for path in args.planning_results:
            frame = _read(path)
            frame.insert(0, "source", str(path))
            frames.append(frame)
        planning = pd.concat(frames, ignore_index=True)
        keys = [name for name in ("method", "budget") if name in planning.columns]
        metrics = [name for name in ("mean_detection_probability", "coverage_fraction", "blind_fraction", "minimum_detection_probability") if name in planning.columns]
        grouped = planning.groupby(keys, as_index=False)[metrics].agg(["mean", "std", "count"])
        grouped.columns = keys + [f"{metric}_{stat}" for metric, stat in grouped.columns.tolist()[len(keys):]]
        _write(grouped, args.output_dir, "table_planning_budget")


if __name__ == "__main__":
    main()
