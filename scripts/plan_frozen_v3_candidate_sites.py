"""Plan candidate monitoring sites from frozen v3 held-out station fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jiangxi_radio.frozen_v3_release import (
    greedy_select_sites,
    normal_detection_probability,
    planning_metrics,
)
from jiangxi_radio.paper_visualization import activate_publication_style, save_publication_figure


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _metadata(data: dict[str, np.ndarray]) -> dict[str, object]:
    value = data.get("metadata_json")
    if value is None:
        return {}
    try:
        return json.loads(str(value.item()))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _axis(data: dict[str, np.ndarray], name: str) -> np.ndarray:
    value = np.asarray(data[name], dtype=float)
    return value[0] if value.ndim == 2 else value


def _validate_archives(archives: list[dict[str, np.ndarray]], paths: list[Path]) -> None:
    required = {"station_xyz_m", "target_dbm", "coarse_dbm", "air_mask", "conservative_detection_probability"}
    for path, archive in zip(paths, archives, strict=True):
        missing = sorted(required.difference(archive))
        if missing:
            raise ValueError(f"{path} is missing frozen release fields: {missing}")
    reference = archives[0]
    for path, archive in zip(paths[1:], archives[1:], strict=True):
        for name in ("station_xyz_m", "target_dbm", "coarse_dbm", "air_mask"):
            if not np.allclose(archive[name], reference[name], rtol=0.0, atol=1e-5):
                raise ValueError(f"{path} is not geometry-aligned with the first archive: {name}")


def _plot_planning(
    output: Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    mask: np.ndarray,
    teacher_union: dict[str, np.ndarray],
    selections: dict[str, list[int]],
    stations: np.ndarray,
    budget: int,
    threshold: float,
) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.colors import Normalize

    activate_publication_style()
    valid_layers = np.count_nonzero(mask, axis=(1, 2))
    if not np.any(valid_layers):
        raise ValueError("planning mask contains no valid horizontal layer")
    candidates = np.flatnonzero(valid_layers > 0)
    zi = int(candidates[np.argmin(np.abs(z[candidates] - float(np.median(z[candidates]))))])
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), sharex=True, sharey=True, layout="constrained")
    # The union probability is often close to one after several stations. Plot
    # the complementary risk so method differences remain visible instead of
    # collapsing into a uniformly yellow panel.
    risk_fields = {name: np.clip(1.0 - field, 1e-6, 1.0) for name, field in teacher_union.items()}
    values = np.concatenate([field[zi][mask[zi]] for field in risk_fields.values()])
    finite = values[np.isfinite(values)]
    low = float(max(np.percentile(finite, 1.0), 1e-6))
    high = float(max(np.percentile(finite, 99.0), low * 10.0))
    from matplotlib.colors import LogNorm
    norm = LogNorm(low, high)
    cmap = colormaps["viridis"]
    methods = list(teacher_union)
    for axis, method in zip(axes, methods, strict=True):
        field = risk_fields[method][zi]
        masked = np.where(mask[zi], field, np.nan)
        image = axis.pcolormesh(x / 1000.0, y / 1000.0, masked, shading="auto", cmap=cmap, norm=norm)
        chosen = np.asarray(selections[method], dtype=int)
        axis.scatter(stations[:, 0] / 1000.0, stations[:, 1] / 1000.0, s=11, facecolors="none", edgecolors="#4b5563", linewidths=0.35)
        axis.scatter(stations[chosen, 0] / 1000.0, stations[chosen, 1] / 1000.0, s=35, c="#d95f02", marker="^", edgecolors="white", linewidths=0.6, zorder=3)
        axis.set_title(f"{method} (K={budget})")
        axis.set_xlabel("X (km)")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("Y (km)")
    fig.colorbar(image, ax=axes, shrink=0.82, label="Teacher non-detection risk (1 − P)")
    return save_publication_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--threshold-dbm", type=float, default=-100.0)
    parser.add_argument("--shadow-std-db", type=float, default=6.0)
    parser.add_argument("--probability-threshold", type=float, default=0.9)
    parser.add_argument("--figure-budget", type=int, default=None)
    parser.add_argument("--skip-figure", action="store_true", help="write CSV/JSON only when plotting dependencies are unavailable")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"use a new output directory: {args.output_dir}")
    paths = [path.resolve() for path in args.archive]
    archives = [_load(path) for path in paths]
    _validate_archives(archives, paths)
    first = archives[0]
    station = np.asarray(first["station_xyz_m"], dtype=float)
    target = np.asarray(first["target_dbm"], dtype=float)
    coarse = np.asarray(first["coarse_dbm"], dtype=float)
    mask = np.asarray(first["air_mask"], dtype=bool)
    if mask.ndim == 4:
        mask = mask[0]
    frozen = np.mean(np.stack([archive["conservative_detection_probability"] for archive in archives]), axis=0)
    teacher = normal_detection_probability(target, 0.0, args.threshold_dbm, args.shadow_std_db)
    fspl = normal_detection_probability(coarse, 0.0, args.threshold_dbm, args.shadow_std_db)
    # Candidate fields have one row per station/case and a common [Z,Y,X] mask.
    methods = {"FSPL": fspl, "CMF-PENO-RF": frozen, "OracleTeacher": teacher}
    budgets = sorted(set(int(value) for value in args.budgets))
    if any(value < 1 or value > station.shape[0] for value in budgets):
        raise ValueError("budgets must be between one and the number of candidate stations")
    records: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    figure_union: dict[str, np.ndarray] = {}
    figure_selection: dict[str, list[int]] = {}
    figure_budget = args.figure_budget if args.figure_budget is not None else max(budgets)
    for budget in budgets:
        for method, selection_field in methods.items():
            selected, _ = greedy_select_sites(selection_field, budget, mask)
            union_teacher = 1.0 - np.prod(1.0 - teacher[selected], axis=0)
            metric = planning_metrics(union_teacher, mask, args.probability_threshold)
            records.append({"method": method, "budget": budget, **metric})
            for rank, candidate in enumerate(selected, start=1):
                selected_rows.append({
                    "method": method, "budget": budget, "rank": rank,
                    "candidate_index": int(candidate),
                    "x_m": float(station[candidate, 0]), "y_m": float(station[candidate, 1]),
                    "z_m": float(station[candidate, 2]),
                })
            if budget == figure_budget:
                figure_union[method] = union_teacher
                figure_selection[method] = selected
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir / "planning_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(args.output_dir / "selected_sites.csv", index=False, encoding="utf-8-sig")
    figure_paths = []
    if not args.skip_figure:
        figure_paths = _plot_planning(
            args.output_dir / f"planning_teacher_union_budget{figure_budget}",
            _axis(first, "x_m"), _axis(first, "y_m"), _axis(first, "z_m"),
            mask, figure_union, figure_selection, station, figure_budget, args.probability_threshold,
        )
    metadata = {
        "schema": "frozen_v3_candidate_planning_v1",
        "evidence": "simulation_teacher_agreement_not_measurement_accuracy",
        "archives": [str(path) for path in paths],
        "ensemble_seeds": [_metadata(archive).get("seed") for archive in archives],
        "methods": list(methods), "budgets": budgets,
        "selection_probability": "FSPL, frozen v3 conservative probability, or teacher probability",
        "evaluation_probability": "teacher detection probability, union assumes conditionally independent station shadowing",
        "probability_threshold": args.probability_threshold,
        "figure_paths": figure_paths,
    }
    (args.output_dir / "planning_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(records).to_string(index=False))
    print("saved:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
