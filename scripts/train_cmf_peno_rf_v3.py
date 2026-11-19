"""Train the upgraded CMF-PENO-RF v3 overall model."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from jiangxi_radio.cmf_peno import remove_strongest_obstruction
from jiangxi_radio.cmf_peno_data import load_cmf_peno_rf_dataset
from jiangxi_radio.cmf_peno_v3 import (
    CMFPENOV3,
    multi_edge_counterfactual_gain_db,
    v3_mean_loss,
    v3_variance_loss,
)
from jiangxi_radio.province_experiments import variant_names, variant_spec
from jiangxi_radio.volume_simulation import input_volume, sha256_file


class RFV3Cases(Dataset):
    def __init__(self, data: dict[str, np.ndarray], split: int):
        self.data = data
        self.indices = np.flatnonzero(np.asarray(data["split"]) == split)
        self.inputs = [input_volume(data, int(index)) for index in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = int(self.indices[position])
        air = np.asarray(self.data["air_mask"])
        air = air[index] if air.ndim == 4 else air
        scenario = np.asarray(self.data["scenario_teacher_dbm"][index], dtype=np.float32)
        return (
            torch.from_numpy(self.inputs[position]).float(),
            torch.from_numpy(np.asarray(self.data["coarse_dbm"][index][None])).float(),
            torch.from_numpy(np.asarray(self.data["fidelity_dbm"][index])).float(),
            torch.from_numpy(np.asarray(air[None])).float(),
            torch.from_numpy(np.asarray(self.data["event_features"][index])).float(),
            torch.from_numpy(np.asarray(self.data["event_type"][index])).long(),
            torch.from_numpy(np.asarray(self.data["event_mask"][index])).bool(),
            torch.from_numpy(np.asarray(self.data["counterfactual_event_gain_db"][index][None])).float(),
            torch.from_numpy(scenario).float(),
        )


def _device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else value if value != "auto" else "cpu")


def _batch_forward(model, batch, device, *, counterfactual: bool = True):
    values = [value.to(device) for value in batch]
    volume, coarse, fidelity, mask, event_features, event_type, event_mask, gain, scenario = values
    output = model(volume, coarse, event_features, event_type, event_mask)
    counterfactual_output = None
    if counterfactual:
        removed_mask = remove_strongest_obstruction(event_features, event_type, event_mask)
        counterfactual_output = model(volume, coarse, event_features, event_type, removed_mask)
    return values, output, counterfactual_output, scenario


def _mean_epoch(model, loader, device, optimizer, args):
    training = optimizer is not None
    model.train(training)
    totals = {name: 0.0 for name in ("total", "stage1", "mean", "low", "high", "orthogonal", "counterfactual", "upper")}
    absolute_error = 0.0
    voxels = 0
    for batch in loader:
        values, output, counterfactual_output, _ = _batch_forward(
            model, batch, device, counterfactual=args.variant_spec.use_event_consistency
        )
        volume, coarse, fidelity, mask, event_features, event_type, event_mask, _, _ = values
        if training:
            optimizer.zero_grad(set_to_none=True)
        gain = multi_edge_counterfactual_gain_db(event_features, event_mask)
        selected = mask.bool() & (gain >= float(args.minimum_counterfactual_gain_db))
        losses = v3_mean_loss(
            output, fidelity, mask, coarse=coarse,
            counterfactual_gain_db=gain,
            counterfactual_output=counterfactual_output,
            counterfactual_mask=selected,
            stage_weight=args.stage_weight,
            mean_weight=args.mean_weight,
            frequency_weight=args.frequency_weight,
            orthogonal_weight=args.orthogonal_weight,
            counterfactual_weight=args.counterfactual_weight,
            upper_weight=args.upper_weight,
            allowed_gain_db=args.allowed_gain_db,
            modes=args.modes,
        )
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError("non-finite v3 mean loss")
        if training:
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        count = int(mask.sum().item())
        voxels += count
        for name in totals:
            totals[name] += float(losses[name].detach()) * max(len(volume), 1)
        absolute_error += float((output["mean_dbm"].detach() - fidelity[:, -1:]).abs()[mask > 0].sum())
    denominator = max(len(loader.dataset), 1)
    return {**{name: value / denominator for name, value in totals.items()}, "mae_db": absolute_error / max(voxels, 1)}


def _variance_epoch(model, loader, device, optimizer):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for batch in loader:
        values, output, _, scenario = _batch_forward(model, batch, device, counterfactual=False)
        mask = values[3]
        if training:
            optimizer.zero_grad(set_to_none=True)
        loss = v3_variance_loss(output, scenario, mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite v3 variance loss")
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.variance_head.parameters(), 5.0)
            optimizer.step()
        total += float(loss.detach()) * max(len(values[0]), 1)
        count += max(len(values[0]), 1)
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--variance-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--event-hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--variance-lr", type=float, default=5e-4)
    parser.add_argument("--stage-weight", type=float, default=0.5)
    parser.add_argument("--mean-weight", type=float, default=1.0)
    parser.add_argument("--frequency-weight", type=float, default=0.5)
    parser.add_argument("--orthogonal-weight", type=float, default=0.01)
    parser.add_argument("--counterfactual-weight", type=float, default=0.2)
    parser.add_argument("--upper-weight", type=float, default=0.02)
    parser.add_argument("--allowed-gain-db", type=float, default=7.0)
    parser.add_argument("--minimum-counterfactual-gain-db", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--variant", choices=variant_names(), default="full_v3")
    args = parser.parse_args()
    args.modes = (4, 6, 6)
    args.variant_spec = variant_spec(args.variant)
    if not args.variant_spec.use_multilevel_supervision:
        args.stage_weight = 0.0
        args.frequency_weight = 0.0
        args.orthogonal_weight = 0.0
    if not args.variant_spec.use_event_consistency:
        args.counterfactual_weight = 0.0
    if args.output_dir.exists():
        raise FileExistsError(f"use a new run directory: {args.output_dir}")
    if args.epochs <= 0 or args.variance_epochs <= 0 or args.batch_size <= 0:
        parser.error("mean/variance epochs and batch size must be positive")
    if args.lr <= 0.0 or args.variance_lr <= 0.0:
        parser.error("learning rates must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = _device(args.device)
    data = load_cmf_peno_rf_dataset(args.dataset)
    train = RFV3Cases(data, 0)
    validation = RFV3Cases(data, 1)
    if not len(train) or not len(validation):
        raise ValueError("train and validation station cases are required")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation, batch_size=1, shuffle=False)
    model = CMFPENOV3(
        input_channels=train[0][0].shape[0], width=args.width,
        event_hidden=args.event_hidden, event_feature_count=int(train[0][4].shape[1]),
        modes=args.modes,
        use_events=args.variant_spec.use_events,
        use_local=args.variant_spec.use_local,
        use_spectral=args.variant_spec.use_spectral,
    ).to(device)
    mean_parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("variance_head")
        and name != "log_variance_bias"
    ]
    variance_parameters = list(model.variance_head.parameters()) + [model.log_variance_bias]
    mean_optimizer = torch.optim.AdamW(mean_parameters, lr=args.lr, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True)
    metadata = {
        "schema": "cmf_peno_model_v3",
        "architecture": "factorized_event_local_global_learnable_fourier_detached_variance",
        "variant": args.variant,
        "variant_spec": {
            key: value for key, value in vars(args.variant_spec).items()
        },
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items() if key != "variant_spec"
        },
        "input_channels": model.input_channels,
        "event_feature_count": int(train[0][4].shape[1]),
        "teacher_components_used_as_model_input": False,
        "measurement_validation": False,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "active_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "uncertainty_training": (
            "independent_variance_head_on_detached_mean_features"
            if args.variant_spec.use_uncertainty else "not_applicable"
        ),
        "fourier_modes": list(args.modes),
    }
    history = []
    best_mean = float("inf")
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        mean_train = _mean_epoch(model, train_loader, device, mean_optimizer, args)
        with torch.no_grad():
            mean_val = _mean_epoch(model, validation_loader, device, None, args)
        row = {"phase": "mean", "epoch": epoch, **{f"train_{k}": v for k, v in mean_train.items()}, **{f"val_{k}": v for k, v in mean_val.items()}}
        history.append(row)
        print(json.dumps(row), flush=True)
        if mean_val["mae_db"] < best_mean:
            best_mean = mean_val["mae_db"]
            torch.save({**metadata, "epoch": epoch, "best_val_mae_db": best_mean, "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}}, args.output_dir / "best_mean.pt")

    best_mean_checkpoint = torch.load(args.output_dir / "best_mean.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_mean_checkpoint["state_dict"])
    if not args.variant_spec.use_uncertainty:
        torch.save(
            {
                **best_mean_checkpoint,
                "variant": args.variant,
                "variant_spec": metadata["variant_spec"],
                "uncertainty_training": "not_applicable",
            },
            args.output_dir / "best.pt",
        )
        metadata.update({
            "best_val_mae_db": best_mean,
            "best_val_variance_nll": None,
            "history": history,
            "training_seconds": time.perf_counter() - start,
        })
        (args.output_dir / "training.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({
            "output_dir": str(args.output_dir), "best_val_mae_db": best_mean,
            "best_val_variance_nll": None, "device": str(device),
        }))
        return
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in variance_parameters:
        parameter.requires_grad_(True)
    variance_optimizer = torch.optim.AdamW(variance_parameters, lr=args.variance_lr, weight_decay=1e-4)
    best_variance = float("inf")
    for epoch in range(1, args.variance_epochs + 1):
        variance_train = _variance_epoch(model, train_loader, device, variance_optimizer)
        with torch.no_grad():
            variance_val = _variance_epoch(model, validation_loader, device, None)
        row = {"phase": "variance", "epoch": epoch, "train_variance": variance_train, "val_variance": variance_val}
        history.append(row)
        print(json.dumps(row), flush=True)
        if variance_val < best_variance:
            best_variance = variance_val
            torch.save({**metadata, "epoch": epoch, "best_val_mae_db": best_mean, "best_val_variance_nll": best_variance, "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}}, args.output_dir / "best.pt")
    metadata.update({"best_val_mae_db": best_mean, "best_val_variance_nll": best_variance, "history": history, "training_seconds": time.perf_counter() - start})
    (args.output_dir / "training.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "best_val_mae_db": best_mean, "best_val_variance_nll": best_variance, "device": str(device)}))


if __name__ == "__main__":
    main()
