"""Merge aligned v2 and v3 LOSO summaries for a controlled comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-summary", type=Path, required=True)
    parser.add_argument("--v3-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = [pd.read_csv(args.v2_summary), pd.read_csv(args.v3_summary)]
    required = {"scene", "seed", "variant"}
    for path, frame in zip((args.v2_summary, args.v3_summary), frames):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
    merged = pd.concat(frames, ignore_index=True)
    if merged.duplicated(["scene", "seed", "variant"]).any():
        raise ValueError("duplicate scene-seed-variant rows in merged summaries")
    merged = merged.sort_values(["scene", "seed", "variant"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(merged.to_string(index=False))
    print("saved:", args.output.resolve())


if __name__ == "__main__":
    main()
