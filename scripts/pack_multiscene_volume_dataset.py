"""Pack local station-volume scenes without mixing their geometry fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_scene(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    meta = json.loads(str(data["metadata_json"].item()))
    if meta.get("schema") != "station_volume_v1":
        raise ValueError(f"{path}: unsupported dataset schema")
    required = (
        "x_m", "y_m", "z_m", "terrain_m", "surface_m", "air_mask",
        "station_xyz_m", "split", "coarse_dbm", "teacher_dbm",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")
    n = len(data["station_xyz_m"])
    shape = (len(data["z_m"]), len(data["y_m"]), len(data["x_m"]))
    if data["coarse_dbm"].shape != (n, *shape):
        raise ValueError(f"{path}: invalid coarse_dbm shape")
    if data["teacher_dbm"].shape != (n, *shape):
        raise ValueError(f"{path}: invalid teacher_dbm shape")
    if data["air_mask"].shape != shape:
        raise ValueError(f"{path}: invalid air_mask shape")
    if set(np.unique(data["split"]).tolist()) != {0, 1, 2}:
        raise ValueError(f"{path}: split must contain 0, 1 and 2")
    return data


def scene_records(entries: dict[str, Path], held_out: str) -> dict[str, np.ndarray]:
    names = list(entries)
    scenes = [load_scene(entries[name]) for name in names]
    records: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "x_m", "y_m", "z_m", "terrain_m", "surface_m", "air_mask",
            "station_xyz_m", "split", "coarse_dbm", "teacher_dbm",
        )
    }
    scene_ids: list[np.ndarray] = []
    source_indices: list[np.ndarray] = []

    for name, data in zip(names, scenes):
        n = len(data["station_xyz_m"])
        is_test_scene = name == held_out
        # Keep the original train/validation split in source scenes. Their
        # original test stations become validation stations, so only the held-
        # out scene contributes to the final test set.
        split = np.asarray(data["split"], dtype=np.int64).copy()
        if is_test_scene:
            split.fill(2)
        else:
            split[split == 2] = 1

        records["x_m"].append(np.repeat(data["x_m"][None], n, axis=0))
        records["y_m"].append(np.repeat(data["y_m"][None], n, axis=0))
        records["z_m"].append(np.repeat(data["z_m"][None], n, axis=0))
        records["terrain_m"].append(np.repeat(data["terrain_m"][None], n, axis=0))
        records["surface_m"].append(np.repeat(data["surface_m"][None], n, axis=0))
        records["air_mask"].append(np.repeat(data["air_mask"][None], n, axis=0))
        records["station_xyz_m"].append(data["station_xyz_m"])
        records["split"].append(split)
        records["coarse_dbm"].append(data["coarse_dbm"])
        records["teacher_dbm"].append(data["teacher_dbm"])
        scene_ids.append(np.full(n, name))
        source_indices.append(np.arange(n, dtype=np.int64))

    packed = {key: np.concatenate(value, axis=0) for key, value in records.items()}
    packed["scene_id"] = np.concatenate(scene_ids)
    packed["source_station_index"] = np.concatenate(source_indices)
    return packed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="append", required=True,
                        help="scene_name=dataset.npz; repeat for each scene")
    parser.add_argument("--held-out", required=True,
                        help="scene name used only for the test split")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    entries = {}
    for item in args.scene:
        if "=" not in item:
            parser.error("--scene must be NAME=DATASET_NPZ")
        name, filename = item.split("=", 1)
        if not name or name in entries:
            parser.error("scene names must be nonempty and unique")
        entries[name] = Path(filename)
    if args.held_out not in entries:
        parser.error("--held-out must match one --scene name")

    # Parent directory names are used in metadata; make them match the CLI
    # names even if the source files were moved or renamed.
    packed = scene_records(entries, args.held_out)
    meta = {
        "schema": "station_volume_v1",
        "evidence": "simulation_teacher_agreement_not_measurement_accuracy",
        "measurement_validation": False,
        "multi_scene": True,
        "held_out_scene": args.held_out,
        "scene_names": list(entries),
        "source_datasets": {name: str(path) for name, path in entries.items()},
        "split_policy": "source train/val preserved; source test reassigned to val; held-out scene=test",
        "input_contract": "per-case x_m,y_m,z_m,terrain_m,surface_m,air_mask",
    }
    packed["metadata_json"] = np.asarray(json.dumps(meta, ensure_ascii=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **packed)
    output.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "held_out": args.held_out,
        "cases": int(len(packed["station_xyz_m"])),
        "split_counts": {
            str(value): int(np.sum(packed["split"] == value))
            for value in (0, 1, 2)
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
