"""Add propagation events and multi-fidelity targets to a volume dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jiangxi_radio.cmf_peno_data import build_cmf_peno_dataset
from jiangxi_radio.volume_simulation import sha256_file


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--high-fidelity", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--max-events", type=int, default=4)
    parser.add_argument("--frequency-mhz", type=float, default=2500.0)
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("--output must end with .npz")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing overwrite: {args.output}")
    base = _load(args.base_dataset)
    high = None if args.high_fidelity is None else _load(args.high_fidelity)
    result = build_cmf_peno_dataset(
        base,
        high_fidelity=high,
        samples=args.samples,
        max_events=args.max_events,
        frequency_mhz=args.frequency_mhz,
        progress=True,
    )
    metadata = json.loads(str(result["metadata_json"]))
    metadata["source_dataset"] = str(args.base_dataset.resolve())
    metadata["source_dataset_sha256"] = sha256_file(args.base_dataset)
    metadata["high_fidelity_source"] = (
        None if args.high_fidelity is None else str(args.high_fidelity.resolve())
    )
    metadata["high_fidelity_sha256"] = (
        None if args.high_fidelity is None else sha256_file(args.high_fidelity)
    )
    result["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "cases": int(len(result["station_xyz_m"])),
        "fidelities": metadata["fidelity_names"],
        "event_shape": list(result["event_features"].shape),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
