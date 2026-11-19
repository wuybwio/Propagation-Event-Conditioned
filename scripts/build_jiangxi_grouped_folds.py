"""Pack province scenes into group-disjoint folds and add RF event fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from jiangxi_radio.province_experiments import build_grouped_roles, pack_grouped_scenes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--roles", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--event-samples", type=int, default=32)
    parser.add_argument("--max-events", type=int, default=4)
    parser.add_argument("--frequency-mhz", type=float, default=2500.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig")
    required = {"scene_id", "geo_group", "dataset_path"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"built scene manifest is missing columns: {missing}")
    roles = (
        pd.read_csv(args.roles, encoding="utf-8-sig")
        if args.roles else build_grouped_roles(manifest)
    )
    entries = {
        str(row.scene_id): Path(str(row.dataset_path))
        for row in manifest.itertuples(index=False)
    }
    missing_paths = [str(path) for path in entries.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"scene datasets are missing: {missing_paths[:5]}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    roles.to_csv(args.output_root / "grouped_roles.csv", index=False, encoding="utf-8-sig")
    for fold in sorted(roles["fold"].astype(str).unique()):
        fold_dir = args.output_root / fold
        base = fold_dir / "base_station_volume.npz"
        rf = fold_dir / f"cmf_peno_rf_s{args.event_samples}.npz"
        fold_dir.mkdir(parents=True, exist_ok=True)
        if not base.exists():
            packed = pack_grouped_scenes(entries, roles, fold=fold)
            np.savez_compressed(base, **packed)
            base.with_suffix(".json").write_text(
                json.dumps(
                    json.loads(str(packed["metadata_json"].item())),
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )
        elif not args.resume:
            raise FileExistsError(f"refusing overwrite: {base}")
        if not rf.exists():
            command = [
                sys.executable, "scripts/build_cmf_peno_rf_dataset.py",
                "--base-dataset", str(base), "--output", str(rf),
                "--samples", str(args.event_samples),
                "--max-events", str(args.max_events),
                "--frequency-mhz", str(args.frequency_mhz),
            ]
            print("+ " + " ".join(command), flush=True)
            subprocess.run(command, check=True)
        elif not args.resume:
            raise FileExistsError(f"refusing overwrite: {rf}")
    print(f"saved grouped folds: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
