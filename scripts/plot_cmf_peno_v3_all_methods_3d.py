#!/usr/bin/env python3
"""Compare 3-D monitoring-probability fields and orthogonal cuts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jiangxi_radio.frozen_v3_release import normal_detection_probability
from jiangxi_radio.paper_visualization import activate_publication_style, save_publication_figure

METHOD_ORDER = (
    "deterministic",
    "cnn_residual",
    "fno_residual",
    "no_multilevel",
    "no_events",
    "no_local",
    "no_spectral",
    "no_event_consistency",
    "full_v3",
)
METHOD_LABELS = {
    "FSPL": "FSPL",
    "GeometricTeacher": "Geometric teacher",
    "deterministic": "Deterministic model",
    "cnn_residual": "CNN residual",
    "fno_residual": "FNO residual",
    "no_multilevel": "No multi-fidelity",
    "no_events": "No event encoding",
    "no_local": "No local operator",
    "no_spectral": "No spectral operator",
    "no_event_consistency": "No event consistency",
    "full_v3": "CMF-PENO-RF (proposed)",
}


def _ordered_methods(available: list[str] | tuple[str, ...]) -> list[str]:
    supplied = set(available)
    baselines = [name for name in ("FSPL", "GeometricTeacher") if name in supplied]
    proposed = [name for name in ("full_v3",) if name in supplied]
    known = [name for name in METHOD_ORDER if name in supplied and name not in proposed]
    extras = sorted(supplied.difference(baselines).difference(known).difference(proposed))
    return baselines + proposed + known + extras


def _prediction_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must be VARIANT=predictions.npz")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("prediction must be VARIANT=predictions.npz")
    return name, Path(path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _case_axis(data: dict[str, np.ndarray], key: str, index: int) -> np.ndarray:
    value = np.asarray(data[key])
    return np.asarray(value[index] if value.ndim == 2 else value, dtype=float)


def _case_field(data: dict[str, np.ndarray], key: str, index: int) -> np.ndarray:
    value = np.asarray(data[key])
    return np.asarray(value[index] if value.ndim == 3 else value, dtype=float)


def _case_mask(data: dict[str, np.ndarray], index: int) -> np.ndarray:
    value = np.asarray(data["air_mask"])
    return np.asarray(value[index] if value.ndim == 4 else value, dtype=bool)


def _select_case(
    data: dict[str, np.ndarray],
    predictions: dict[str, dict[str, np.ndarray]],
    scene_id: str,
    requested_index: int | None,
) -> tuple[int, int]:
    reference = predictions["full_v3"]
    indices = np.asarray(reference["case_indices"], dtype=int)
    if requested_index is not None:
        matches = np.flatnonzero(indices == requested_index)
        if len(matches) != 1:
            raise ValueError(f"dataset case index {requested_index} is not in the aligned test predictions")
        position = int(matches[0])
        if str(data["scene_id"][requested_index]) != scene_id:
            raise ValueError(f"case {requested_index} is scene {data['scene_id'][requested_index]}, not {scene_id}")
        return requested_index, position

    scene_names = np.asarray(data["scene_id"]).astype(str)
    candidates = [
        (position, int(case_index))
        for position, case_index in enumerate(indices)
        if scene_names[int(case_index)] == scene_id
    ]
    if not candidates:
        available = sorted(set(scene_names[indices].tolist()))
        raise ValueError(f"scene {scene_id!r} has no aligned test predictions; available={available}")
    # Choose the median full-model absolute error within the requested scene.
    scores = []
    for position, case_index in candidates:
        prediction = np.asarray(reference["mean_dbm"][position], dtype=float)
        target = np.asarray(reference["target_dbm"][position], dtype=float)
        mask = np.asarray(reference["mask"][position], dtype=bool)
        scores.append(float(np.mean(np.abs(prediction[mask] - target[mask]))) if mask.any() else np.inf)
    order = np.argsort(np.asarray(scores))
    position, case_index = candidates[int(order[len(order) // 2])]
    return case_index, position


def _edges(centres: np.ndarray) -> np.ndarray:
    centres = np.asarray(centres, dtype=float)
    if centres.ndim != 1 or len(centres) < 2 or np.any(np.diff(centres) <= 0):
        raise ValueError("voxel axes must be strictly increasing and contain at least two points")
    mid = 0.5 * (centres[:-1] + centres[1:])
    return np.concatenate(([centres[0] - (mid[0] - centres[0])], mid,
                           [centres[-1] + (centres[-1] - mid[-1])]))


def _face_colors(field, air, cmap, norm, alpha=0.98):
    from matplotlib.colors import to_rgba

    colors = cmap(norm(np.clip(field, 0.0, 1.0)))
    colors[..., 3] = alpha
    colors[~air] = to_rgba("#555b61", 0.96)
    return colors


def _draw_panel(axis, probability, air, terrain, surface, x, y, z,
                view, indices, cmap, norm):
    """Render a 3-D outer box or one horizontal/vertical probability cut."""
    from matplotlib.colors import to_rgba

    z_reference = float(np.median(terrain))
    z_display = z - z_reference
    zi, xi, yi = indices
    xg, yg = np.meshgrid(x, y)

    if view == "3d":
        axis.plot_surface(
            xg, yg, np.full_like(xg, z_display[-1]),
            facecolors=_face_colors(probability[-1], air[-1], cmap, norm, 0.90),
            shade=False, linewidth=0.10, edgecolor=(0.12, 0.15, 0.18, 0.16),
            antialiased=False,
        )
        xside, zside = np.meshgrid(x, z_display)
        axis.plot_surface(
            xside, np.full_like(xside, y[0]), zside,
            facecolors=_face_colors(probability[:, 0, :], air[:, 0, :], cmap, norm, 0.84),
            shade=False, linewidth=0.10, edgecolor=(0.12, 0.15, 0.18, 0.14),
            antialiased=False,
        )
        yside, zside = np.meshgrid(y, z_display)
        axis.plot_surface(
            np.full_like(yside, x[-1]), yside, zside,
            facecolors=_face_colors(probability[:, :, -1], air[:, :, -1], cmap, norm, 0.84),
            shade=False, linewidth=0.10, edgecolor=(0.12, 0.15, 0.18, 0.14),
            antialiased=False,
        )
        buildings = (
            (z[:, None, None] > terrain[None, :, :] + 0.5)
            & (z[:, None, None] <= surface[None, :, :] + 0.5)
        )
        bz, by, bx = np.nonzero(buildings)
        if len(bx):
            stride = max(1, len(bx) // 1000)
            axis.scatter(x[bx[::stride]], y[by[::stride]], z_display[bz[::stride]],
                         s=3, color="#30363b", alpha=0.38, depthshade=False)
        terrain_rgba = np.tile(to_rgba("#788078", 0.48), (len(y), len(x), 1))
        axis.plot_surface(xg, yg, terrain - z_reference, facecolors=terrain_rgba,
                          shade=False, linewidth=0.0, antialiased=False)
    elif view == "z":
        axis.plot_surface(xg, yg, np.full_like(xg, z_display[zi]),
                          facecolors=_face_colors(probability[zi], air[zi], cmap, norm),
                          shade=False, linewidth=0.10, edgecolor=(0.12, 0.15, 0.18, 0.17),
                          antialiased=False)
    elif view == "x":
        yg, zg = np.meshgrid(y, z_display)
        axis.plot_surface(np.full_like(yg, x[xi]), yg, zg,
                          facecolors=_face_colors(probability[:, :, xi], air[:, :, xi], cmap, norm),
                          shade=False, linewidth=0.10, edgecolor=(0.12, 0.15, 0.18, 0.17),
                          antialiased=False)
    else:
        xg, zg = np.meshgrid(x, z_display)
        axis.plot_surface(xg, np.full_like(xg, y[yi]), zg,
                          facecolors=_face_colors(probability[:, yi, :], air[:, yi, :], cmap, norm),
                          shade=False, linewidth=0.10, edgecolor=(0.12, 0.15, 0.18, 0.17),
                          antialiased=False)

    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(float(y.min()), float(y.max()))
    axis.set_zlim(float(min(0.0, z_display.min())), float(z_display.max()))
    axis.set_xlabel("X (m)", labelpad=0, fontsize=6)
    axis.set_ylabel("Y (m)", labelpad=0, fontsize=6)
    axis.set_zlabel("Height (m)", labelpad=0, fontsize=6)
    axis.view_init(elev=24, azim=-55)
    axis.set_box_aspect((1.0, 1.0, 0.72))
    axis.tick_params(labelsize=5.5, pad=0)
    axis.grid(True, alpha=0.14)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prediction", type=_prediction_arg, action="append", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--case-index", type=int, help="global dataset case index; default selects median-error case in scene")
    parser.add_argument("--threshold-dbm", type=float, default=-100.0)
    parser.add_argument("--shadow-std-db", type=float, default=6.0)
    parser.add_argument("--coverage-probability", type=float, default=0.9)
    parser.add_argument("--slice-height-m", type=float, default=80.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"use a new output directory: {args.output_dir}")
    if not 0.0 < args.coverage_probability < 1.0:
        raise ValueError("coverage probability must be strictly between zero and one")

    data = _load_npz(args.dataset)
    required_dataset = {
        "x_m", "y_m", "z_m", "terrain_m", "surface_m", "air_mask",
        "station_xyz_m", "split", "scene_id", "coarse_dbm", "fidelity_dbm",
    }
    missing = sorted(required_dataset - set(data))
    if missing:
        raise ValueError(f"dataset is missing fields: {missing}")
    predictions = {}
    case_indices = None
    for name, path in args.prediction:
        if name in predictions:
            raise ValueError(f"duplicate prediction variant: {name}")
        archive = _load_npz(path)
        required = {"case_indices", "mean_dbm", "target_dbm", "detection_probability", "mask"}
        absent = sorted(required - set(archive))
        if absent:
            raise ValueError(f"{path} is missing fields: {absent}")
        current = np.asarray(archive["case_indices"], dtype=int)
        if case_indices is None:
            case_indices = current
        elif not np.array_equal(current, case_indices):
            raise ValueError(f"prediction case indices differ for variant {name}")
        predictions[name] = archive
    if "full_v3" not in predictions:
        raise ValueError("include --prediction full_v3=... to select a representative case")

    case_index, prediction_position = _select_case(
        data, predictions, args.scene_id, args.case_index
    )
    x = _case_axis(data, "x_m", case_index)
    y = _case_axis(data, "y_m", case_index)
    z = _case_axis(data, "z_m", case_index)
    terrain = _case_field(data, "terrain_m", case_index)
    surface = _case_field(data, "surface_m", case_index)
    air = _case_mask(data, case_index)
    if not np.array_equal(air, np.asarray(predictions["full_v3"]["mask"][prediction_position], dtype=bool)):
        raise ValueError("dataset air mask and evaluation mask are not aligned")

    fields: dict[str, np.ndarray] = {}
    fields["FSPL"] = normal_detection_probability(
        np.asarray(data["coarse_dbm"][case_index]), 0.0,
        args.threshold_dbm, args.shadow_std_db,
    )
    fields["GeometricTeacher"] = normal_detection_probability(
        np.asarray(data["fidelity_dbm"][case_index, 1]), 0.0,
        args.threshold_dbm, args.shadow_std_db,
    )
    for name in METHOD_ORDER:
        if name in predictions:
            fields[name] = np.asarray(
                predictions[name]["detection_probability"][prediction_position], dtype=float
            )
    for name in predictions:
        if name not in fields:
            fields[name] = np.asarray(
                predictions[name]["detection_probability"][prediction_position], dtype=float
            )
    expected = (len(z), len(y), len(x))
    for name, field in fields.items():
        if field.shape != expected or not np.isfinite(field).all():
            raise ValueError(f"{name} probability shape {field.shape} is invalid; expected {expected}")

    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    activate_publication_style()
    names = _ordered_methods(list(fields))
    cmap = colormaps["RdYlBu_r"]
    norm = Normalize(0.0, 1.0)
    z_reference = float(np.median(terrain))
    z_display = z - z_reference
    zi = int(np.argmin(np.abs(z_display - args.slice_height_m)))
    station = np.asarray(data["station_xyz_m"][case_index], dtype=float)
    xi = int(np.argmin(np.abs(x - station[0])))
    yi = int(np.argmin(np.abs(y - station[1])))
    xg, yg = np.meshgrid(x, y)
    pages = [names[start:start + 3] for start in range(0, len(names), 3)]
    views = (
        ("3d", "3-D field"),
        ("z", f"Z = {z_display[zi]:.0f} m AGL"),
        ("x", f"X = {x[xi]:.0f} m"),
        ("y", f"Y = {y[yi]:.0f} m"),
    )
    args.output_dir.mkdir(parents=True)
    files = []
    for page_index, page_names in enumerate(pages, start=1):
        fig = plt.figure(
            figsize=(14.4, 3.25 * len(page_names) + 0.8),
            layout="constrained",
        )
        grid = fig.add_gridspec(len(page_names), 5, width_ratios=[1.08, 1, 1, 1, 0.045])
        for row, name in enumerate(page_names):
            for column, (view, view_title) in enumerate(views):
                axis = fig.add_subplot(grid[row, column], projection="3d")
                _draw_panel(
                    axis, fields[name], air, terrain, surface, x, y, z,
                    view, (zi, xi, yi), cmap, norm,
                )
                method = METHOD_LABELS.get(name, name.replace("_", " "))
                axis.set_title(
                    f"{method} | {view_title}", fontsize=7.7,
                    fontweight="bold" if name == "full_v3" else "normal", pad=2,
                )
                if view == "z":
                    field_slice = fields[name][zi]
                    if field_slice.min() < args.coverage_probability < field_slice.max():
                        axis.contour(
                            xg, yg, field_slice,
                            levels=[args.coverage_probability],
                            zdir="z", offset=float(z_display[zi]),
                            colors="#20262b", linewidths=0.65,
                        )
        color_axis = fig.add_subplot(grid[:, 4])
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), cax=color_axis,
            label="Detection probability (0 to 1; red = high)",
        )
        fig.suptitle(
            f"3-D coverage field and orthogonal cuts | {args.scene_id} | "
            f"case {case_index} | page {page_index}/{len(pages)}",
            fontsize=10.5,
        )
        stem = args.output_dir / f"fig_all_methods_3d_coverage_p{page_index:02d}"
        files.extend(save_publication_figure(fig, stem))
        plt.close(fig)

    metadata = {
        "schema": "cmf_peno_rf_all_methods_3d_orthogonal_cuts_v1",
        "evidence": "model/reference detection-probability fields on one held-out simulated scene",
        "dataset": str(args.dataset.resolve()),
        "scene_id": args.scene_id,
        "dataset_case_index": int(case_index),
        "prediction_position": int(prediction_position),
        "station_xyz_m": station.tolist(),
        "methods": names,
        "views": [title for _, title in views],
        "pages": pages,
        "color_scale": "0 to 1, RdYlBu_r; maximum probability is red",
        "coverage_probability_threshold": float(args.coverage_probability),
        "slice_height_m_agl": float(z_display[zi]),
        "slice_x_m": float(x[xi]),
        "slice_y_m": float(y[yi]),
        "received_power_threshold_dbm": float(args.threshold_dbm),
        "shadow_std_db": float(args.shadow_std_db),
        "height_reference": "median terrain elevation in local scene",
        "files": files,
    }
    (args.output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
