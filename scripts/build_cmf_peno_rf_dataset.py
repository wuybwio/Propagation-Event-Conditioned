"""Build a mechanism-aware CMF-PENO-RF dataset from station-volume data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from jiangxi_radio.cmf_peno_data import build_cmf_peno_rf_dataset
from jiangxi_radio.multi_mechanism_propagation import default_multi_mechanism_scenarios


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--max-events", type=int, default=4)
    parser.add_argument("--frequency-mhz", type=float, default=2500.0)
    parser.add_argument("--no-reflection", action="store_true")
    parser.add_argument("--no-multi-edge", action="store_true")
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("--output must end with .npz")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing overwrite: {args.output}")
    result = build_cmf_peno_rf_dataset(
        _load(args.base_dataset),
        samples=args.samples,
        max_events=args.max_events,
        frequency_mhz=args.frequency_mhz,
        scenarios=default_multi_mechanism_scenarios(),
        use_multi_edge=not args.no_multi_edge,
        use_reflection=not args.no_reflection,
        progress=True,
    )
    metadata = json.loads(str(result["metadata_json"]))
    metadata["source_dataset"] = str(args.base_dataset.resolve())
    metadata["source_dataset_sha256"] = sha256_file(args.base_dataset)
    metadata["leakage_free"] = True
    metadata["diagnostic_only_fields"] = [
        "rf_mechanism_db", "scenario_teacher_dbm", "scenario_mechanism_db",
    ]
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
        "scenarios": metadata["scenario_names"],
        "event_shape": list(result["event_features"].shape),
        "mechanism_shape": list(result["rf_mechanism_db"].shape),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
