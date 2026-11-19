"""Fit one CMF-PENO variance temperature from held-out predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jiangxi_radio.uncertainty_calibration import fit_prediction_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-predictions", type=Path, required=True,
        help="Prediction archive from the held-out val or calibration split.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing overwrite: {args.output}")
    payload = fit_prediction_archive(args.validation_predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
