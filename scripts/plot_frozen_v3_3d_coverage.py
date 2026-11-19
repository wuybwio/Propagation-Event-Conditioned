"""Create journal-ready 3-D power and conservative-coverage figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jiangxi_radio.frozen_v3_release import representative_case_index
from jiangxi_radio.paper_visualization import activate_publication_style, nearest_index, save_publication_figure


def _case(value: np.ndarray, index: int, case_ndim: int) -> np.ndarray:
    array = np.asarray(value)
    return array[index] if array.ndim == case_ndim else array


def _select_case(prediction, target, probability, mask, mode: str) -> int:
    if mode == "median":
        return representative_case_index(prediction, target, mask)
    if mode != "informative":
        raise ValueError(f"unknown case selection mode: {mode}")
    scores = []
    for index in range(len(prediction)):
        valid = mask[index].astype(bool)
        if not np.any(valid):
            scores.append(-np.inf)
            continue
        power = target[index][valid]
        prob = probability[index][valid]
        spread = float(np.percentile(power, 95) - np.percentile(power, 5))
        transition = float(np.mean((prob > 0.05) & (prob < 0.95)))
        low_mean = float(1.0 - np.mean(prob))
        scores.append(spread / 40.0 + 2.0 * transition + low_mean)
    if not np.isfinite(scores).any():
        raise ValueError("no informative case contains valid voxels")
    return int(np.argmax(scores))


def _load(path: Path, requested_case: int | None, selection: str = "informative") -> dict[str, object]:
    with np.load(path, allow_pickle=False) as source:
        required = {"predicted_dbm", "coarse_dbm", "target_dbm", "conservative_detection_probability", "air_mask", "x_m", "y_m", "z_m", "terrain_m", "surface_m", "station_xyz_m", "metadata_json"}
        missing = sorted(required.difference(source.files))
        if missing:
            raise ValueError(f"frozen archive is missing fields: {missing}")
        prediction = np.asarray(source["predicted_dbm"], dtype=float)
        target = np.asarray(source["target_dbm"], dtype=float)
        mask = np.asarray(source["air_mask"], dtype=bool)
        probability_all = np.asarray(source["conservative_detection_probability"], dtype=float)
        if requested_case is None:
            index = _select_case(prediction, target, probability_all, mask, selection)
        else:
            index = int(requested_case)
        if index < 0 or index >= len(prediction):
            raise IndexError(f"case offset {index} is outside [0, {len(prediction)})")
        result = {
            "index": index,
            "prediction": prediction[index], "target": target[index],
            "coarse": np.asarray(source["coarse_dbm"], dtype=float)[index],
            "probability": probability_all[index],
            "mask": mask[index], "station": np.asarray(source["station_xyz_m"], dtype=float)[index],
            "x": _case(source["x_m"], index, 2).astype(float),
            "y": _case(source["y_m"], index, 2).astype(float),
            "z": _case(source["z_m"], index, 2).astype(float),
            "terrain": _case(source["terrain_m"], index, 3).astype(float),
            "surface": _case(source["surface_m"], index, 3).astype(float),
            "metadata": json.loads(str(source["metadata_json"].item())),
        }
    return result


def _building_mask(z: np.ndarray, terrain: np.ndarray, surface: np.ndarray) -> np.ndarray:
    return (z[:, None, None] > terrain[None] + 0.5) & (z[:, None, None] <= surface[None] + 0.5)


def _style_3d(axis, x, y, z):
    axis.set_xlim(x.min(), x.max()); axis.set_ylim(y.min(), y.max()); axis.set_zlim(z.min(), z.max())
    axis.set_xlabel("X (m)", labelpad=2); axis.set_ylabel("Y (m)", labelpad=2); axis.set_zlabel("Height AGL (m)", labelpad=2)
    axis.view_init(elev=24, azim=-55); axis.set_box_aspect((1, 1, 0.70)); axis.tick_params(pad=0)


def _rgba(field, buildings, cmap, norm, alpha=0.96):
    from matplotlib.colors import to_rgba
    colors = cmap(norm(field)); colors[..., 3] = alpha; colors[buildings] = to_rgba("#484d55", 1.0)
    return colors


def _cube(axis, field, buildings, x, y, z, cmap, norm):
    xx, yy = np.meshgrid(x, y)
    axis.plot_surface(xx, yy, np.full_like(xx, z[-1]), facecolors=_rgba(field[-1], buildings[-1], cmap, norm, 0.80), shade=False, linewidth=0)
    xx, zz = np.meshgrid(x, z)
    axis.plot_surface(xx, np.full_like(xx, y[0]), zz, facecolors=_rgba(field[:, 0], buildings[:, 0], cmap, norm, 0.82), shade=False, linewidth=0)
    yy, zz = np.meshgrid(y, z)
    axis.plot_surface(np.full_like(yy, x[-1]), yy, zz, facecolors=_rgba(field[:, :, -1], buildings[:, :, -1], cmap, norm, 0.82), shade=False, linewidth=0)
    bz, by, bx = np.nonzero(buildings)
    if len(bx):
        stride = max(1, len(bx) // 1800)
        axis.scatter(x[bx[::stride]], y[by[::stride]], z[bz[::stride]], s=5, c="#353a42", alpha=0.56, depthshade=False)


def _slices(axis, field, buildings, x, y, z, kind, index, cmap, norm):
    if kind == "z":
        xx, yy = np.meshgrid(x, y); zz = np.full_like(xx, z[index]); colors = _rgba(field[index], buildings[index], cmap, norm)
    elif kind == "x":
        yy, zz = np.meshgrid(y, z); xx = np.full_like(yy, x[index]); colors = _rgba(field[:, :, index], buildings[:, :, index], cmap, norm)
    else:
        xx, zz = np.meshgrid(x, z); yy = np.full_like(xx, y[index]); colors = _rgba(field[:, index], buildings[:, index], cmap, norm)
    axis.plot_surface(xx, yy, zz, facecolors=colors, shade=False, linewidth=0.08, edgecolor=(0, 0, 0, 0.08))


def _comparison(case: dict[str, object], output: Path, slice_agl: float) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    x, y, z_abs = case["x"], case["y"], case["z"]
    ground = float(np.median(case["terrain"])); z = z_abs - ground
    buildings = _building_mask(z_abs, case["terrain"], case["surface"])
    zi = nearest_index(z, slice_agl); xi = nearest_index(x, case["station"][0]); yi = nearest_index(y, case["station"][1])
    valid_values = np.concatenate([case["target"][case["mask"]], case["prediction"][case["mask"]]])
    low, high = np.percentile(valid_values, [2, 98]); norm = Normalize(float(low), float(max(high, low + 1)))
    cmap = colormaps["turbo"]
    fig = plt.figure(figsize=(14.2, 9.5), layout="constrained")
    grid = fig.add_gridspec(3, 5, width_ratios=[1.12, 1, 1, 1, 0.05])
    panel = 0
    for row, (label, field) in enumerate((("FSPL", case["coarse"]), ("Multi-mechanism teacher", case["target"]), ("CMF-PENO-RF", case["prediction"]))):
        for col, (kind, index, title) in enumerate((("cube", 0, "3-D scene"), ("z", zi, f"Z={z[zi]:.0f} m AGL"), ("x", xi, f"X={x[xi]:.0f} m"), ("y", yi, f"Y={y[yi]:.0f} m"))):
            axis = fig.add_subplot(grid[row, col], projection="3d")
            if kind == "cube": _cube(axis, field, buildings, x, y, z, cmap, norm)
            else: _slices(axis, field, buildings, x, y, z, kind, index, cmap, norm)
            _style_3d(axis, x, y, z); axis.set_title(f"({chr(97 + panel)}) {label}: {title}", pad=4); panel += 1
    color_axis = fig.add_subplot(grid[:, 4]); fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_axis, label="Received power (dBm)")
    return save_publication_figure(fig, output / "figure_3d_power_comparison")


def _coverage(case: dict[str, object], output: Path, slice_agl: float, threshold: float) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    x, y, z_abs = case["x"], case["y"], case["z"]; ground = float(np.median(case["terrain"])); z = z_abs - ground
    buildings = _building_mask(z_abs, case["terrain"], case["surface"]); field = case["probability"]
    zi = nearest_index(z, slice_agl); xi = nearest_index(x, case["station"][0]); yi = nearest_index(y, case["station"][1])
    cmap = colormaps["viridis"]; norm = Normalize(0.0, 1.0)
    fig = plt.figure(figsize=(14.0, 3.65), layout="constrained"); grid = fig.add_gridspec(1, 5, width_ratios=[1.12, 1, 1, 1, 0.05])
    for col, (kind, index, title) in enumerate((("cube", 0, "3-D conservative coverage"), ("z", zi, f"Z={z[zi]:.0f} m AGL"), ("x", xi, f"X={x[xi]:.0f} m"), ("y", yi, f"Y={y[yi]:.0f} m"))):
        axis = fig.add_subplot(grid[0, col], projection="3d")
        shown = np.where(field >= threshold, field, np.nan)
        if kind == "cube": _cube(axis, shown, buildings, x, y, z, cmap, norm)
        else: _slices(axis, shown, buildings, x, y, z, kind, index, cmap, norm)
        _style_3d(axis, x, y, z); axis.set_title(f"({chr(97 + col)}) {title}", pad=4)
    color_axis = fig.add_subplot(grid[0, 4]); fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_axis, label="Conservative P(detection)")
    return save_publication_figure(fig, output / "figure_3d_conservative_coverage")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-offset", type=int)
    parser.add_argument("--case-selection", choices=("informative", "median"), default="informative")
    parser.add_argument("--slice-agl-m", type=float, default=80.0)
    parser.add_argument("--coverage-threshold", type=float, default=0.9)
    args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(f"use a new output directory: {args.output_dir}")
    activate_publication_style(); case = _load(args.archive, args.case_offset, args.case_selection); args.output_dir.mkdir(parents=True)
    files = _comparison(case, args.output_dir, args.slice_agl_m) + _coverage(case, args.output_dir, args.slice_agl_m, args.coverage_threshold)
    metadata = {"schema": "frozen_v3_3d_figures_v1", "archive": str(args.archive.resolve()), "case_offset": case["index"], "case_selection": args.case_selection, "coverage_threshold": args.coverage_threshold, "files": files, "building_geometry": "surface-minus-terrain voxel occupancy"}
    (args.output_dir / "figure_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
