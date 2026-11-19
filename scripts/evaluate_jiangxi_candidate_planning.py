"""Select candidate sites on model fields and evaluate them on one reference field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jiangxi_radio.cmf_peno_data import load_cmf_peno_rf_dataset
from jiangxi_radio.frozen_v3_release import normal_detection_probability
from jiangxi_radio.province_experiments import evaluate_candidate_planning


def _item(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must be VARIANT=NPZ")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("prediction must be VARIANT=NPZ")
    return name, Path(path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _reference(
    path: Path | None,
    data: dict[str, np.ndarray],
    test_indices: np.ndarray,
    threshold_dbm: float,
    shadow_std_db: float,
) -> tuple[np.ndarray, dict[str, object]]:
    if path is None:
        power = np.asarray(data["fidelity_dbm"][test_indices, -1], dtype=float)
        return normal_detection_probability(power, 0.0, threshold_dbm, shadow_std_db), {
            "evidence": "internal_training_generator",
            "independent_of_training_generator": False,
        }
    archive = _load_npz(path)
    if "case_indices" not in archive or not np.array_equal(archive["case_indices"], test_indices):
        raise ValueError("independent reference case_indices do not match the test split")
    if "reference_probability" in archive:
        probability = np.asarray(archive["reference_probability"], dtype=float)
    elif "reference_dbm" in archive:
        probability = normal_detection_probability(
            archive["reference_dbm"], 0.0, threshold_dbm, shadow_std_db
        )
    else:
        raise ValueError("reference NPZ needs reference_probability or reference_dbm")
    metadata = {}
    if "metadata_json" in archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
    return probability, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prediction", type=_item, action="append", required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--require-independent-reference", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--threshold-dbm", type=float, default=-100.0)
    parser.add_argument("--shadow-std-db", type=float, default=6.0)
    parser.add_argument("--probability-threshold", type=float, default=0.9)
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument(
        "--planning-scope", choices=("scene", "group", "both"), default="both",
    )
    parser.add_argument(
        "--exact-budget-max", type=int, default=3,
        help="Enumerate the exact reference optimum up to this budget; use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"use a new output directory: {args.output_dir}")
    data = load_cmf_peno_rf_dataset(args.dataset)
    test_indices = np.flatnonzero(np.asarray(data["split"]) == 2)
    if not len(test_indices):
        raise ValueError("province fold contains no test cases")
    reference, reference_meta = _reference(
        args.reference, data, test_indices, args.threshold_dbm, args.shadow_std_db
    )
    independent = bool(reference_meta.get("independent_of_training_generator", False))
    if args.require_independent_reference and not independent:
        raise ValueError("reference metadata does not declare independent_of_training_generator=true")
    selections: dict[str, np.ndarray] = {
        "FSPL": normal_detection_probability(
            np.asarray(data["coarse_dbm"])[test_indices], 0.0,
            args.threshold_dbm, args.shadow_std_db,
        )
    }
    for name, path in args.prediction:
        archive = _load_npz(path)
        if not np.array_equal(archive.get("case_indices"), test_indices):
            raise ValueError(f"prediction indices do not match test split: {path}")
        if "detection_probability" in archive:
            selections[name] = np.asarray(archive["detection_probability"], dtype=float)
        else:
            selections[name] = normal_detection_probability(
                archive["mean_dbm"], archive.get("variance_db2", 0.0),
                args.threshold_dbm, args.shadow_std_db,
            )
    scene_ids = np.asarray(data["scene_id"])[test_indices].astype(str)
    geo_groups = np.asarray(data["geo_group"])[test_indices].astype(str)
    stations = np.asarray(data["station_xyz_m"])[test_indices]
    air = np.asarray(data["air_mask"])
    air = air[test_indices] if air.ndim == 4 else np.repeat(air[None], len(test_indices), axis=0)
    records: list[dict[str, object]] = []
    selection_payload: dict[str, dict[str, list[int]]] = {}
    if args.planning_scope in {"scene", "both"}:
        for scene_id in sorted(set(scene_ids)):
            selected = np.flatnonzero(scene_ids == scene_id)
            mask = air[selected[0]]
            scene_records, scene_selections = evaluate_candidate_planning(
                {name: value[selected] for name, value in selections.items()},
                reference[selected], stations[selected], mask,
                budgets=args.budgets,
                probability_threshold=args.probability_threshold,
                random_repeats=args.random_repeats,
                seed=args.seed,
                exact_budget_max=args.exact_budget_max,
            )
            for record in scene_records:
                record.update({
                    "planning_scope": "local_scene",
                    "planning_unit": scene_id,
                    "scene_id": scene_id,
                    "geo_group": geo_groups[selected[0]],
                    "reference_independent": independent,
                    "reference_evidence": reference_meta.get("evidence", "unspecified"),
                })
            records.extend(scene_records)
            selection_payload[f"scene:{scene_id}"] = scene_selections

    if args.planning_scope in {"group", "both"}:
        for geo_group in sorted(set(geo_groups)):
            selected = np.flatnonzero(geo_groups == geo_group)
            group_scenes = sorted(set(scene_ids[selected]))
            offsets: dict[str, tuple[int, int]] = {}
            masks = []
            cursor = 0
            for scene_id in group_scenes:
                local = selected[scene_ids[selected] == scene_id]
                local_mask = air[local[0]].reshape(-1)
                offsets[scene_id] = (cursor, cursor + local_mask.size)
                cursor += local_mask.size
                masks.append(local_mask)
            group_mask = np.concatenate(masks)

            def block_diagonal_fields(value: np.ndarray) -> np.ndarray:
                combined = np.zeros((len(selected), cursor), dtype=np.float32)
                for position, case_index in enumerate(selected):
                    start, stop = offsets[scene_ids[case_index]]
                    combined[position, start:stop] = value[case_index].reshape(-1)
                return combined

            group_records, group_selections = evaluate_candidate_planning(
                {name: block_diagonal_fields(value) for name, value in selections.items()},
                block_diagonal_fields(reference), stations[selected], group_mask,
                budgets=args.budgets,
                probability_threshold=args.probability_threshold,
                random_repeats=args.random_repeats,
                seed=args.seed,
                exact_budget_max=0,
            )
            for record in group_records:
                record.update({
                    "planning_scope": "geographical_group",
                    "planning_unit": geo_group,
                    "scene_id": f"{geo_group}_combined",
                    "geo_group": geo_group,
                    "reference_independent": independent,
                    "reference_evidence": reference_meta.get("evidence", "unspecified"),
                })
            records.extend(group_records)
            selection_payload[f"group:{geo_group}"] = group_selections
    args.output_dir.mkdir(parents=True)
    result = pd.DataFrame(records)
    result.to_csv(args.output_dir / "planning_results_raw.csv", index=False, encoding="utf-8-sig")
    summary = result.groupby(["planning_scope", "method", "budget"], as_index=False).agg(
        planning_units=("planning_unit", "nunique"),
        mean_detection_probability=("mean_detection_probability", "mean"),
        coverage_fraction=("coverage_fraction", "mean"),
        blind_fraction=("blind_fraction", "mean"),
        minimum_detection_probability=("minimum_detection_probability", "mean"),
    )
    summary.to_csv(args.output_dir / "planning_results_summary.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "selected_indices.json").write_text(
        json.dumps(selection_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "schema": "jiangxi_candidate_planning_evaluation_v1",
        "selection_models": sorted(selections),
        "evaluation_reference": str(args.reference.resolve()) if args.reference else None,
        "reference_metadata": reference_meta,
        "independent_reference_required": args.require_independent_reference,
        "test_scene_count": int(len(set(scene_ids))),
        "random_repeats": args.random_repeats,
        "exact_budget_max": args.exact_budget_max,
        "planning_scope": args.planning_scope,
    }
    (args.output_dir / "planning_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"saved: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
