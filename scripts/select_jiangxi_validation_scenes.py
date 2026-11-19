"""Select geographically dispersed Jiangxi validation scenes from real stations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from jiangxi_radio.province_experiments import (
    build_grouped_roles,
    select_spatial_scene_centres,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", type=Path, default=Path("data/stations.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roles-output", type=Path)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--scenes-per-group", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--size-m", type=float, default=500.0)
    parser.add_argument("--scene-class", default="station_centred_spatial_sample")
    args = parser.parse_args()
    roles_output = args.roles_output or args.output.with_name("province_grouped_roles.csv")
    for path in (args.output, roles_output):
        if path.exists():
            raise FileExistsError(f"refusing overwrite: {path}")
    stations = pd.read_csv(args.stations, encoding="utf-8-sig")
    scenes = select_spatial_scene_centres(
        stations,
        group_count=args.groups,
        scenes_per_group=args.scenes_per_group,
        seed=args.seed,
    )
    scenes["size_m"] = float(args.size_m)
    scenes["scene_class"] = str(args.scene_class)
    roles = build_grouped_roles(scenes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    roles_output.parent.mkdir(parents=True, exist_ok=True)
    scenes.to_csv(args.output, index=False, encoding="utf-8-sig")
    roles.to_csv(roles_output, index=False, encoding="utf-8-sig")
    print(scenes.to_string(index=False))
    print(f"saved scenes: {args.output.resolve()}")
    print(f"saved roles: {roles_output.resolve()}")


if __name__ == "__main__":
    main()
