"""Evaluate non-learned propagation baselines on one province test fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jiangxi_radio.cmf_peno_data import load_cmf_peno_rf_dataset
from jiangxi_radio.field_evaluation import field_metrics


def _external_reference(path: Path, indices: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as source:
        if "case_indices" not in source or not np.array_equal(source["case_indices"], indices):
            raise ValueError("external reference case_indices do not match the test split")
        if "reference_dbm" not in source:
            raise ValueError("external reference is missing reference_dbm")
        target = np.asarray(source["reference_dbm"], dtype=float)
        metadata = (
            json.loads(str(source["metadata_json"].item()))
            if "metadata_json" in source else {}
        )
    return target, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-dbm", type=float, default=-100.0)
    parser.add_argument("--shadow-std-db", type=float, default=6.0)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--require-independent-reference", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"use a new output directory: {args.output_dir}")
    data = load_cmf_peno_rf_dataset(args.dataset)
    indices = np.flatnonzero(np.asarray(data["split"]) == 2)
    if not len(indices):
        raise ValueError("province fold contains no test cases")
    air = np.asarray(data["air_mask"])
    rows = []
    fields = {
        "FSPL": np.asarray(data["coarse_dbm"]),
        "GeometricTeacher": np.asarray(data["fidelity_dbm"][:, 1]),
    }
    reference_metadata: dict[str, object] = {}
    if args.reference:
        target, reference_metadata = _external_reference(args.reference, indices)
        independent = bool(reference_metadata.get("independent_of_training_generator", False))
        if args.require_independent_reference and not independent:
            raise ValueError("reference is not declared independent of the training generator")
    else:
        target = np.asarray(data["fidelity_dbm"])[indices, -1]
        independent = False
    for position, index in enumerate(indices):
        mask = air[index] if air.ndim == 4 else air
        for method, field in fields.items():
            row = field_metrics(
                field[index], target[position], mask,
                args.threshold_dbm, args.shadow_std_db,
            )
            row.update({
                "variant": method,
                "case_index": int(index),
                "scene_id": str(np.asarray(data["scene_id"])[index]),
                "geo_group": str(np.asarray(data["geo_group"])[index]),
                "source_station_index": int(np.asarray(data["source_station_index"])[index]),
                "reference_independent": independent,
            })
            rows.append(row)
    args.output_dir.mkdir(parents=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "case_metrics.csv", index=False, encoding="utf-8-sig")
    summary = frame.groupby("variant", as_index=False).agg(
        cases=("case_index", "count"),
        mae_db=("mae_db", "mean"),
        rmse_db=("rmse_db", "mean"),
        bias_db=("bias_db", "mean"),
        probability_mse=("probability_mse", "mean"),
        coverage_iou=("coverage_iou", "mean"),
        blind_iou=("blind_iou", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        false_coverage_rate=("false_coverage_rate", "mean"),
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "schema": "jiangxi_field_baselines_v1",
        "evidence": reference_metadata.get(
            "evidence", "internal_multi_mechanism_teacher_agreement"
        ),
        "reference": str(args.reference.resolve()) if args.reference else None,
        "reference_metadata": reference_metadata,
        "dataset": str(args.dataset.resolve()),
        "methods": list(fields),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
