"""Benchmark the internal rule reference and CMF-PENO-RF on identical held-out cases.

The benchmark starts from gridded terrain, surface, station and axes. Dataset
loading and checkpoint loading are excluded. Timing separates the shared FSPL
and event construction from the rule synthesis and neural inference.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jiangxi_radio.cmf_peno_data import _bilinear_sample, load_cmf_peno_rf_dataset
from jiangxi_radio.cmf_peno_v3 import CMFPENOV3
from jiangxi_radio.multi_mechanism_propagation import (
    combine_multi_edge_losses_db,
    terrain_roughness_loss_db,
    two_ray_reflection_gain_db,
    synthesize_multi_mechanism_teacher,
)
from jiangxi_radio.propagation_events import build_propagation_events
from jiangxi_radio.volume_simulation import input_volume


def case_array(data: dict, key: str, index: int, case_ndim: int) -> np.ndarray:
    value = np.asarray(data[key])
    return np.asarray(value[index] if value.ndim == case_ndim else value)


def fspl_prior(x: np.ndarray, y: np.ndarray, z: np.ndarray, station: np.ndarray,
               frequency_mhz: float) -> np.ndarray:
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    distance = np.maximum(np.sqrt((xx - station[0]) ** 2 +
                                  (yy - station[1]) ** 2 +
                                  (zz - station[2]) ** 2), 1.0)
    return (31.0 - 32.45 - 20.0 * np.log10(frequency_mhz)
            - 20.0 * np.log10(distance / 1000.0)).astype(np.float32)


def augment_events(events: dict, max_events: int) -> np.ndarray:
    features = np.asarray(events["event_features"], dtype=np.float32)
    rank = np.broadcast_to(
        (np.arange(max_events, dtype=np.float32) / float(max_events))[:, None, None, None],
        features.shape[:1] + features.shape[2:],
    )
    valid = np.asarray(events["event_mask"], dtype=np.float32)
    return np.concatenate((features, rank[:, None], valid[:, None]), axis=1)


def nominal_reference(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                      terrain: np.ndarray, station: np.ndarray,
                      coarse: np.ndarray, event_features: np.ndarray,
                      event_mask: np.ndarray, frequency_mhz: float,
                      parameters: dict) -> np.ndarray:
    edges = np.where(event_mask, event_features[:, 11] * 60.0, 0.0)
    edge_loss = combine_multi_edge_losses_db(
        np.moveaxis(edges, 0, -1),
        secondary_weight=0.35 if parameters.get("use_multi_edge", True) else 0.0,
        tertiary_weight=0.15 if parameters.get("use_multi_edge", True) else 0.0,
        max_loss_db=60.0,
    )
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    horizontal = np.sqrt((xx - station[0]) ** 2 + (yy - station[1]) ** 2)
    path = np.sqrt(horizontal**2 + (zz - station[2]) ** 2)
    station_ground = _bilinear_sample(terrain, x, y, float(station[0]), float(station[1]))
    tx_agl = np.full_like(horizontal, max(station[2] - station_ground, 1.0))
    rx_agl = np.maximum(zz - terrain[None, :, :], 1.0)
    reflection = two_ray_reflection_gain_db(
        horizontal, tx_agl, rx_agl,
        frequency_mhz=float(parameters.get("frequency_mhz", frequency_mhz)),
        reflection_coefficient=float(parameters.get("reflection_coefficient", -0.65))
        if parameters.get("use_reflection", True) else 0.0,
        ground_roughness_m=float(parameters.get("ground_roughness_m", 0.15)),
        max_gain_db=float(parameters.get("max_reflection_gain_db", 6.0)),
    )
    environment = terrain_roughness_loss_db(
        terrain, x, y, path, scale=float(parameters.get("environment_scale", 1.0))
    )
    adjustment = 20.0 * np.log10(float(parameters.get("frequency_mhz", frequency_mhz)) /
                                 frequency_mhz)
    return synthesize_multi_mechanism_teacher(
        coarse - adjustment, edge_loss, reflection, environment
    )["teacher_dbm"]


def build_model(checkpoint: dict, input_channels: int) -> CMFPENOV3:
    arguments = checkpoint.get("arguments", {})
    specification = checkpoint.get("variant_spec", {})
    model = CMFPENOV3(
        input_channels=int(checkpoint.get("input_channels", input_channels)),
        width=int(arguments.get("width", 24)),
        event_hidden=int(arguments.get("event_hidden", 32)),
        event_feature_count=int(checkpoint.get("event_feature_count", 14)),
        modes=tuple(int(value) for value in arguments.get("modes", [4, 6, 6])),
        use_events=bool(specification.get("use_events", True)),
        use_local=bool(specification.get("use_local", True)),
        use_spectral=bool(specification.get("use_spectral", True)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def median_ms(operation, repeats: int, synchronize=None) -> tuple[float, object]:
    values = []
    result = None
    for _ in range(repeats):
        if synchronize:
            synchronize()
        start = time.perf_counter_ns()
        result = operation()
        if synchronize:
            synchronize()
        values.append((time.perf_counter_ns() - start) / 1e6)
    return float(np.median(values)), result


def model_inputs(data: dict, index: int, device: torch.device):
    return (
        torch.from_numpy(input_volume(data, index)[None]).float().to(device),
        torch.from_numpy(np.asarray(data["coarse_dbm"][index][None, None])).float().to(device),
        torch.from_numpy(np.asarray(data["event_features"][index][None])).float().to(device),
        torch.from_numpy(np.asarray(data["event_type"][index][None])).long().to(device),
        torch.from_numpy(np.asarray(data["event_mask"][index][None])).bool().to(device),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "calibration"), default="test")
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cuda", action="store_true", help="Also time cached CUDA forward pass")
    args = parser.parse_args()
    if args.max_cases < 1 or args.repeats < 1 or args.threads < 1:
        parser.error("max-cases, repeats, and threads must be positive")
    torch.set_num_threads(args.threads)
    data = load_cmf_peno_rf_dataset(args.dataset)
    metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "cmf_peno_model_v3":
        raise ValueError("checkpoint is not a CMF-PENO-RF model")
    if checkpoint.get("variant") != "full_v3":
        raise ValueError("benchmark requires the full model checkpoint")
    split_id = {"train": 0, "val": 1, "test": 2, "calibration": 3}[args.split]
    available = np.flatnonzero(np.asarray(data["split"]) == split_id)
    if not len(available):
        raise ValueError("requested split is empty")
    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(available, size=min(args.max_cases, len(available)), replace=False))
    model_cpu = build_model(checkpoint, input_volume(data, int(indices[0])).shape[0])
    model_cuda = None
    if args.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        model_cuda = build_model(checkpoint, input_volume(data, int(indices[0])).shape[0]).cuda()
    frequency = float(metadata.get("base_frequency_mhz", 2500.0))
    max_events = int(metadata["max_events"])
    samples = int(metadata["event_samples"])
    nominal = metadata["scenario_parameters"][int(metadata["nominal_scenario_index"])].copy()
    nominal["use_multi_edge"] = metadata.get("use_multi_edge", True)
    nominal["use_reflection"] = metadata.get("use_reflection", True)
    rows = []
    with torch.inference_mode():
        for index in indices:
            index = int(index)
            x = case_array(data, "x_m", index, 2)
            y = case_array(data, "y_m", index, 2)
            z = case_array(data, "z_m", index, 2)
            terrain = case_array(data, "terrain_m", index, 3)
            surface = case_array(data, "surface_m", index, 3)
            station = np.asarray(data["station_xyz_m"][index], dtype=float)
            prior_ms, prior = median_ms(lambda: fspl_prior(x, y, z, station, frequency), args.repeats)
            stored_prior = np.asarray(data["coarse_dbm"][index], dtype=np.float32)
            prior_error = float(np.max(np.abs(prior - stored_prior)))
            if prior_error > 0.002:
                raise ValueError(f"FSPL prior differs from dataset at case {index}: {prior_error:.6f} dB")
            event_ms, events = median_ms(
                lambda: build_propagation_events(
                    x, y, z, terrain, surface, station,
                    samples=samples, max_events=max_events, frequency_mhz=frequency,
                ), args.repeats,
            )
            event_error = float(np.max(np.abs(
                augment_events(events, max_events) - np.asarray(data["event_features"][index])
            )))
            if event_error > 0.002:
                raise ValueError(f"event tensor differs from dataset at case {index}: {event_error:.6f}")
            rule = lambda: nominal_reference(
                x, y, z, terrain, station, stored_prior,
                events["event_features"], events["event_mask"], frequency, nominal,
            )
            rule_ms, direct = median_ms(rule, args.repeats)
            target = np.asarray(data["fidelity_dbm"][index, -1], dtype=np.float32)
            reference_error = float(np.max(np.abs(direct - target)))
            if reference_error > 0.003:
                raise ValueError(f"direct reference differs from target at case {index}: {reference_error:.6f} dB")
            cpu_prep_ms, cpu_inputs = median_ms(
                lambda: model_inputs(data, index, torch.device("cpu")), args.repeats
            )
            model_cpu(*cpu_inputs)
            cpu_ms, _ = median_ms(lambda: model_cpu(*cpu_inputs), args.repeats)
            row = {
                "case_index": index,
                "scene_id": str(data["scene_id"][index]) if "scene_id" in data else "",
                "prior_ms": prior_ms,
                "event_ms": event_ms,
                "direct_rule_ms": rule_ms,
                "direct_total_ms": prior_ms + event_ms + rule_ms,
                "network_cpu_prepare_ms": cpu_prep_ms,
                "network_cpu_forward_ms": cpu_ms,
                "network_cpu_total_ms": prior_ms + event_ms + cpu_prep_ms + cpu_ms,
                "prior_max_abs_diff_db": prior_error,
                "events_max_abs_diff": event_error,
                "reference_max_abs_diff_db": reference_error,
            }
            if model_cuda is not None:
                cuda_device = torch.device("cuda")
                gpu_prep_ms, gpu_inputs = median_ms(
                    lambda: model_inputs(data, index, cuda_device), args.repeats,
                    synchronize=torch.cuda.synchronize,
                )
                model_cuda(*gpu_inputs)
                gpu_ms, _ = median_ms(
                    lambda: model_cuda(*gpu_inputs), args.repeats,
                    synchronize=torch.cuda.synchronize,
                )
                row.update(network_cuda_prepare_ms=gpu_prep_ms,
                           network_cuda_forward_ms=gpu_ms,
                           network_cuda_total_ms=prior_ms + event_ms + gpu_prep_ms + gpu_ms)
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "per_case.csv", index=False)
    numeric = frame.select_dtypes(include="number").drop(columns="case_index")
    summary = pd.DataFrame({"metric": numeric.columns,
                            "mean": numeric.mean().values,
                            "median": numeric.median().values,
                            "sd": numeric.std(ddof=1).values,
                            "n": len(frame)})
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    provenance = {
        "dataset": str(args.dataset.resolve()), "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split, "sampled_indices": indices.tolist(), "repeats": args.repeats,
        "event_samples_per_path": samples, "max_events": max_events,
        "grid_shape_zyx": list(np.asarray(data["coarse_dbm"]).shape[1:]),
        "timing_scope": "Dataset/checkpoint load excluded; prior/event/rule on CPU; model preparation and forward separated; totals sum stage medians.",
        "reference": "internal nominal multi-mechanism rule; not independent ray tracing",
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_model": next((line.split(":", 1)[1].strip() for line in
                           Path("/proc/cpuinfo").read_text(errors="ignore").splitlines()
                           if line.lower().startswith("model name")), None)
                      if Path("/proc/cpuinfo").exists() else platform.processor(),
        "numpy": np.__version__,
        "pytorch": torch.__version__, "cpu_threads": args.threads,
        "cuda_runtime": torch.version.cuda if model_cuda is not None else None,
        "cuda_device": torch.cuda.get_device_name(0) if model_cuda is not None else None,
        "cuda_driver": subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        ).stdout.strip().splitlines()[0]
        if model_cuda is not None and subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=False,
        ).returncode == 0 else None,
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Saved {len(frame)} cases to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
