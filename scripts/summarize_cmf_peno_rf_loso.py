"""Summarize CMF-PENO-RF structure ablations over held-out scenes/seeds."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


METRICS = (
    "mae_db", "rmse_db", "bias_db", "probability_mse", "combined_probability_mse",
    "coverage_iou", "blind_iou", "balanced_accuracy", "false_coverage_rate",
    "coverage_95", "mean_interval_width", "standardized_error_rmse",
)


def _identity(path: Path) -> tuple[str, int]:
    match = re.fullmatch(r"(?P<scene>.+)_seed(?P<seed>\d+)", path.name)
    if match is None:
        raise ValueError(f"cannot parse scene/seed from {path}")
    return match.group("scene"), int(match.group("seed"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-variant",
        default="full",
        help="variant used as the paired reference; default: full",
    )
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob("*_seed*/summary_test.csv")):
        scene, seed = _identity(path.parent)
        frame = pd.read_csv(path)
        if "variant" not in frame.columns or len(frame) == 0:
            raise ValueError(f"invalid summary file: {path}")
        frame.insert(0, "scene", scene)
        frame.insert(1, "seed", seed)
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"no *_seed*/summary_test.csv under {args.root}")
    raw = pd.concat(rows, ignore_index=True)
    # Older single-run summaries intentionally omitted split because the
    # filename already encoded it.  This LOSO table is test-only by contract.
    if "split" not in raw.columns:
        raw["split"] = "test"
    else:
        raw["split"] = raw["split"].fillna("test")
    metric_columns = [value for value in METRICS if value in raw.columns]
    columns = ["scene", "seed", "variant", "split"] + metric_columns
    raw = raw[columns]
    keys = ["scene", "seed", "variant"]
    if raw.duplicated(keys).any():
        duplicated = raw.loc[raw.duplicated(keys, keep=False), keys].drop_duplicates()
        raise ValueError(
            "duplicate scene-seed-variant rows: "
            + duplicated.to_dict(orient="records").__repr__()
        )
    if set(raw["split"].astype(str)) != {"test"}:
        raise ValueError("LOSO summary must contain test rows only")
    overall = raw.groupby("variant")[metric_columns].agg(["mean", "std", "count"])
    overall.columns = ["_".join(str(part) for part in column if str(part)) for column in overall.columns]
    overall = overall.reset_index()
    by_scene = raw.groupby(["scene", "variant"])[metric_columns].agg(["mean", "std", "count"])
    by_scene.columns = ["_".join(str(part) for part in column if str(part)) for column in by_scene.columns]
    by_scene = by_scene.reset_index()
    reference = raw[raw["variant"] == args.reference_variant].set_index(["scene", "seed"])
    if reference.empty:
        raise ValueError(
            f"reference variant {args.reference_variant!r} is absent from the summary"
        )
    paired = []
    for variant in sorted(set(raw["variant"]) - {args.reference_variant}):
        other = raw[raw["variant"] == variant].set_index(["scene", "seed"])
        reference_units = set(reference.index)
        variant_units = set(other.index)
        if reference_units != variant_units:
            missing = sorted(reference_units - variant_units)
            extra = sorted(variant_units - reference_units)
            raise ValueError(
                f"scene-seed unit mismatch for variant {variant!r}; "
                f"missing={missing}, extra={extra}"
            )
        joined = reference.join(
            other,
            lsuffix=f"_{args.reference_variant}",
            rsuffix=f"_{variant}",
            how="inner",
        )
        row = {
            "reference_variant": args.reference_variant,
            "variant": variant,
            "paired_units": len(joined),
        }
        for metric in metric_columns:
            left, right = f"{metric}_{args.reference_variant}", f"{metric}_{variant}"
            if left in joined and right in joined:
                row[
                    f"{metric}_delta_variant_minus_{args.reference_variant}_mean"
                ] = float((joined[right] - joined[left]).mean())
                finite = joined[[left, right]].notna().all(axis=1)
                if finite.any():
                    left_values = joined.loc[finite, left]
                    right_values = joined.loc[finite, right]
                    if metric in {"bias_db", "coverage_95"}:
                        if metric == "bias_db":
                            better = right_values.abs() < left_values.abs()
                        else:
                            better = (right_values - 0.95).abs() < (left_values - 0.95).abs()
                    elif metric in {"coverage_iou", "blind_iou", "balanced_accuracy"}:
                        better = right_values > left_values
                    else:
                        better = right_values < left_values
                    row[f"{metric}_variant_better_count"] = int(better.sum())
                    row[f"{metric}_finite_paired_units"] = int(finite.sum())
                else:
                    row[f"{metric}_variant_better_count"] = 0
                    row[f"{metric}_finite_paired_units"] = 0
        paired.append(row)
    paired_frame = pd.DataFrame(paired)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output, index=False, encoding="utf-8-sig")
    overall.to_csv(args.output.with_name(args.output.stem + "_overall.csv"), index=False, encoding="utf-8-sig")
    by_scene.to_csv(args.output.with_name(args.output.stem + "_by_scene.csv"), index=False, encoding="utf-8-sig")
    paired_frame.to_csv(args.output.with_name(args.output.stem + "_paired.csv"), index=False, encoding="utf-8-sig")
    print("overall:")
    print(overall.to_string(index=False))
    print("saved:", args.output)


if __name__ == "__main__":
    main()
