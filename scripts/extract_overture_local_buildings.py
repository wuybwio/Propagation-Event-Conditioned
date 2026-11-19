"""Extract one audited local building GeoJSON from the Overture tile directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jiangxi_radio.overture_local_buildings import extract_local_buildings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buildings-dir", type=Path, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--size-m", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"choose a fresh output directory: {args.output_dir}")
    collection, audit = extract_local_buildings(
        args.buildings_dir,
        longitude=args.longitude,
        latitude=args.latitude,
        size_m=args.size_m,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "buildings.geojson").write_text(
        json.dumps(collection, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "buildings": audit["building_components"],
        "selected_tiles": audit["selected_tile_count"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
