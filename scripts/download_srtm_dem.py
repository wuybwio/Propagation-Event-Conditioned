"""Download public SRTM 1 arc-second terrain tiles for a bounding box."""

from __future__ import annotations

import argparse

from jiangxi_radio.terrain import download_srtm_tiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/dem/srtm1")
    parser.add_argument("--west", type=float, default=113.5)
    parser.add_argument("--south", type=float, default=24.4)
    parser.add_argument("--east", type=float, default=118.5)
    parser.add_argument("--north", type=float, default=30.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = download_srtm_tiles(
        args.output_dir,
        args.west,
        args.south,
        args.east,
        args.north,
        overwrite=args.overwrite,
    )
    print({"output_dir": args.output_dir, "bounds": [args.west, args.south, args.east, args.north], **result})


if __name__ == "__main__":
    main()
