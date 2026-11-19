"""Build audited local 3-D station-volume datasets from a province scene manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    buildings = parser.add_mutually_exclusive_group(required=True)
    buildings.add_argument("--buildings-dir", type=Path)
    buildings.add_argument("--tileset", type=Path)
    parser.add_argument("--dem-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--terrain-step-m", type=float, default=10.0)
    parser.add_argument("--shape", type=int, nargs=3, default=(16, 24, 24))
    parser.add_argument("--top-agl-m", type=float, default=160.0)
    parser.add_argument("--stations-per-scene", type=int, default=30)
    parser.add_argument("--antenna-agl-m", type=float, default=20.0)
    parser.add_argument("--frequency-mhz", type=float, default=2500.0)
    parser.add_argument("--emitter-dbm", type=float, default=23.0)
    parser.add_argument("--receiver-gain-dbi", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.buildings_dir and not (args.buildings_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"Overture manifest is missing: {args.buildings_dir / 'manifest.json'}")
    if args.tileset and not args.tileset.is_file():
        raise FileNotFoundError(f"building tileset is missing: {args.tileset}")
    if not args.dem_dir.is_dir():
        raise FileNotFoundError(f"FABDEM directory is missing: {args.dem_dir}")
    frame = pd.read_csv(args.manifest, encoding="utf-8-sig")
    required = {"scene_id", "center_longitude", "center_latitude", "size_m"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"scene manifest is missing columns: {missing}")
    output_manifest = args.output_manifest or args.output_root / "built_scene_manifest.csv"
    if output_manifest.exists() and not args.resume:
        raise FileExistsError(f"refusing overwrite: {output_manifest}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    dataset_paths = []
    for position, row in enumerate(frame.itertuples(index=False)):
        scene_dir = args.output_root / str(row.scene_id)
        building_dir = scene_dir / "buildings"
        building_geojson = building_dir / "buildings.geojson"
        dataset = scene_dir / "dataset.npz"
        if dataset.exists():
            if not args.resume:
                raise FileExistsError(f"refusing overwrite: {dataset}")
            dataset_paths.append(str(dataset.resolve()))
            continue
        scene_dir.mkdir(parents=True, exist_ok=True)
        if not building_geojson.exists():
            if building_dir.exists() and any(building_dir.iterdir()):
                raise FileExistsError(f"partial building directory requires review: {building_dir}")
            if args.buildings_dir:
                extract_command = [
                    sys.executable, "scripts/extract_overture_local_buildings.py",
                    "--buildings-dir", str(args.buildings_dir),
                    "--longitude", str(float(row.center_longitude)),
                    "--latitude", str(float(row.center_latitude)),
                    "--size-m", str(float(row.size_m)),
                    "--output-dir", str(building_dir),
                ]
            else:
                extract_command = [
                    sys.executable, "scripts/extract_nc_buildings.py",
                    "--tileset", str(args.tileset),
                    "--longitude", str(float(row.center_longitude)),
                    "--latitude", str(float(row.center_latitude)),
                    "--size-m", str(float(row.size_m)),
                    "--same-id-policy", "preserve-mesh-parts",
                    "--output-dir", str(building_dir),
                ]
            _run(extract_command)
        _run([
            sys.executable, "scripts/build_station_volume_dataset.py",
            "--overture-geojson" if args.buildings_dir else "--nc-build-geojson",
            str(building_geojson),
            "--dem-source", "fabdem", "--dem-dir", str(args.dem_dir),
            "--longitude", str(float(row.center_longitude)),
            "--latitude", str(float(row.center_latitude)),
            "--size-m", str(float(row.size_m)),
            "--terrain-step-m", str(args.terrain_step_m),
            "--shape", *(str(value) for value in args.shape),
            "--top-agl-m", str(args.top_agl_m),
            "--stations", str(args.stations_per_scene),
            "--antenna-agl-m", str(args.antenna_agl_m),
            "--frequency-mhz", str(args.frequency_mhz),
            "--emitter-dbm", str(args.emitter_dbm),
            "--receiver-gain-dbi", str(args.receiver_gain_dbi),
            "--seed", str(args.seed + position),
            "--output", str(dataset),
        ])
        _run([
            sys.executable, "scripts/audit_station_volume_dataset.py",
            "--dataset", str(dataset),
        ])
        dataset_paths.append(str(dataset.resolve()))
    frame["dataset_path"] = dataset_paths
    frame["reference_provenance"] = "internal_geometry_rule_generator"
    frame["independent_reference_available"] = False
    frame.to_csv(output_manifest, index=False, encoding="utf-8-sig")
    metadata = {
        "schema": "jiangxi_province_scene_build_v1",
        "manifest": str(args.manifest.resolve()),
        "scene_count": len(frame),
        "measurement_validation": False,
        "independent_reference_available": False,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    output_manifest.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved: {output_manifest.resolve()}")


if __name__ == "__main__":
    main()
