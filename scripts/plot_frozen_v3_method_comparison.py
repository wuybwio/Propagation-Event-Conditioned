"""Plot a slice-level FSPL/FrozenV3/teacher comparison and error improvement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jiangxi_radio.paper_visualization import activate_publication_style, nearest_index, save_publication_figure
from plot_frozen_v3_3d_coverage import _load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-offset", type=int)
    parser.add_argument("--case-selection", choices=("informative", "median"), default="informative")
    parser.add_argument("--slice-agl-m", type=float, default=80.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"use a new output directory: {args.output_dir}")

    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib import colormaps

    activate_publication_style()
    case = _load(args.archive, args.case_offset, args.case_selection)
    x, y, z_abs = case["x"], case["y"], case["z"]
    ground = float(np.median(case["terrain"])); z = z_abs - ground
    buildings = (z_abs[:, None, None] > case["terrain"][None] + 0.5) & (z_abs[:, None, None] <= case["surface"][None] + 0.5)
    zi = nearest_index(z, args.slice_agl_m)
    fields = {"FSPL": case["coarse"], "FrozenV3": case["prediction"], "Teacher": case["target"]}
    errors = {name: np.abs(field - case["target"]) for name, field in fields.items() if name != "Teacher"}
    improvement = errors["FSPL"] - errors["FrozenV3"]
    valid = case["mask"][zi]
    power_values = np.concatenate([fields[name][zi][valid] for name in fields])
    power_norm = Normalize(float(np.percentile(power_values, 2)), float(np.percentile(power_values, 98)))
    error_values = np.concatenate([errors[name][zi][valid] for name in errors])
    error_norm = Normalize(0.0, float(max(np.percentile(error_values, 99), 1.0)))
    delta_bound = float(max(np.percentile(np.abs(improvement[zi][valid]), 98), 1.0))
    delta_norm = TwoSlopeNorm(vmin=-delta_bound, vcenter=0.0, vmax=delta_bound)
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.5), sharex=True, sharey=True, layout="constrained")
    extent = [x.min() / 1000.0, x.max() / 1000.0, y.min() / 1000.0, y.max() / 1000.0]
    for column, (name, field) in enumerate(fields.items()):
        image = axes[0, column].imshow(np.where(valid, field[zi], np.nan), origin="lower", extent=extent, aspect="auto", cmap="turbo", norm=power_norm)
        axes[0, column].contour(x / 1000.0, y / 1000.0, buildings[zi].astype(float), levels=[0.5], colors="#222222", linewidths=0.6)
        axes[0, column].set_title(name)
        axes[0, column].set_xlabel("X (km)")
        if column == 0: axes[0, column].set_ylabel(f"Y (km), Z={z[zi]:.0f} m")
        if name != "Teacher":
            error = np.where(valid, errors[name][zi], np.nan)
            axes[1, column].imshow(error, origin="lower", extent=extent, aspect="auto", cmap="magma", norm=error_norm)
            axes[1, column].set_title(f"|{name} - Teacher|")
        else:
            axes[1, column].imshow(np.where(valid, improvement[zi], np.nan), origin="lower", extent=extent, aspect="auto", cmap="RdBu_r", norm=delta_norm)
            axes[1, column].set_title("FSPL error − FrozenV3 error")
        axes[1, column].set_xlabel("X (km)")
        if column == 0: axes[1, column].set_ylabel("Absolute error / improvement (dB)")
        axes[0, column].grid(alpha=0.16); axes[1, column].grid(alpha=0.16)
    fig.colorbar(ScalarMappable(norm=power_norm, cmap="turbo"), ax=axes[0, :], shrink=0.83, label="Received power (dBm)")
    fig.colorbar(ScalarMappable(norm=error_norm, cmap="magma"), ax=axes[1, :2], shrink=0.83, label="Absolute error (dB)")
    fig.colorbar(ScalarMappable(norm=delta_norm, cmap="RdBu_r"), ax=axes[1, 2], shrink=0.83, label="Positive = FrozenV3 improves")
    args.output_dir.mkdir(parents=True)
    files = save_publication_figure(fig, args.output_dir / "figure_method_slice_comparison")
    metadata = {"schema": "frozen_v3_method_comparison_v1", "archive": str(args.archive.resolve()), "case_offset": case["index"], "case_selection": args.case_selection, "slice_agl_m": float(z[zi]), "files": files}
    (args.output_dir / "method_comparison_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
