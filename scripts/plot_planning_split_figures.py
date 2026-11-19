#!/usr/bin/env python3
"""Plot planning baselines and model ablations as separate figures.

The script deliberately preserves the budgets available in the input CSV.  In
particular, ExactReference is not interpolated beyond the budgets for which
exact enumeration was performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_LABELS = {
    "full_v3": "CMF-PENO-RF",
    "full": "CMF-PENO-RF",
    "deterministic": "Deterministic",
    "FSPL": "FSPL",
    "Random": "Random",
    "Uniform": "Uniform",
    "ReferenceGreedy": "Reference-field greedy",
    "ExactReference": "Exact reference (available budgets only)",
    "GeometricTeacher": "Geometry-derived teacher",
    "cnn_residual": "CNN residual",
    "fno_residual": "FNO residual",
    "no_multilevel": "No multi-fidelity",
    "no_events": "No propagation events",
    "no_local": "No local operator",
    "no_spectral": "No spectral operator",
    "no_event_consistency": "No event consistency",
    "no_residual": "No residual",
}

COLORS = {
    "full_v3": "#D55E00",
    "full": "#D55E00",
    "deterministic": "#0072B2",
    "FSPL": "#7A7A7A",
    "Random": "#CC79A7",
    "Uniform": "#E69F00",
    "ReferenceGreedy": "#56B4E9",
    "ExactReference": "#009E73",
    "GeometricTeacher": "#009E73",
    "cnn_residual": "#6A3D9A",
    "fno_residual": "#1B9E77",
    "no_multilevel": "#A6761D",
    "no_events": "#E7298A",
    "no_local": "#66A61E",
    "no_spectral": "#E6AB02",
    "no_event_consistency": "#A6761D",
    "no_residual": "#7570B3",
}

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
LINESTYLES = ["-", "--", "-.", ":"]

BASELINE_ORDER = [
    "full_v3", "full", "deterministic", "FSPL", "Random", "Uniform",
    "ReferenceGreedy", "ExactReference", "GeometricTeacher",
]
ABLATION_ORDER = [
    "full_v3", "full", "deterministic", "cnn_residual", "fno_residual",
    "no_multilevel", "no_events", "no_local", "no_spectral",
    "no_event_consistency", "no_residual",
]


def _read_data(path: Path, scope: str) -> tuple[pd.DataFrame, str, str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"planning CSV not found: {path}")
    data = pd.read_csv(path, encoding="utf-8-sig")
    method_col = "method" if "method" in data.columns else "policy" if "policy" in data.columns else None
    if method_col is None or "budget" not in data.columns:
        raise ValueError("planning CSV must contain method or policy and budget columns")
    metric = next(
        (column for column in (
            "mean_detection_probability_mean", "mean_detection_probability",
            "coverage_fraction_mean", "coverage_fraction",
        ) if column in data.columns),
        None,
    )
    if metric is None:
        raise ValueError("planning CSV has no detection-probability or coverage metric")
    if scope and "planning_scope" in data.columns:
        scoped = data[data["planning_scope"].astype(str) == scope].copy()
        if not scoped.empty:
            data = scoped
    data["budget"] = pd.to_numeric(data["budget"], errors="coerce")
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.dropna(subset=[method_col, "budget", metric]).copy()
    if data.empty:
        raise ValueError("no valid numeric planning rows remain after filtering")
    return data, method_col, "budget", metric


def _ordered(values: list[str], priority: list[str]) -> list[str]:
    present = set(values)
    return [name for name in priority if name in present]


def _aggregate(data: pd.DataFrame, method_col: str, metric: str, methods: list[str]) -> pd.DataFrame:
    subset = data[data[method_col].astype(str).isin(methods)].copy()
    if subset.empty:
        return subset
    # The summary CSV normally has one row per method and budget.  Grouping
    # here makes duplicate rows explicit and deterministic without inventing
    # budgets that are absent from the source file.
    return (
        subset.groupby([method_col, "budget"], as_index=False)[metric]
        .mean()
        .sort_values([method_col, "budget"])
    )


def _plot(data: pd.DataFrame, method_col: str, metric: str, methods: list[str],
          output: Path, title: str, audit: dict) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "savefig.bbox": "tight",
    })
    fig, axis = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
    aggregate = _aggregate(data, method_col, metric, methods)
    for index, method in enumerate(methods):
        group = aggregate[aggregate[method_col].astype(str) == method]
        if group.empty:
            continue
        label = METHOD_LABELS.get(method, method)
        exact = method == "ExactReference"
        if exact:
            budget_text = ", ".join(
                str(int(value)) if float(value).is_integer() else f"{float(value):g}"
                for value in group["budget"].tolist()
            )
            label = f"Exact reference (B={budget_text})"
        axis.plot(
            group["budget"].to_numpy(float), group[metric].to_numpy(float),
            color=COLORS.get(method, "#4D4D4D"),
            marker=MARKERS[index % len(MARKERS)],
            linestyle=LINESTYLES[index % len(LINESTYLES)],
            linewidth=2.2 if method in {"full_v3", "full"} else 1.35,
            markersize=4.8,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=label,
            zorder=5 if method in {"full_v3", "full"} else 3,
        )
        lower_col = metric.replace("_mean", "") + "_group_bootstrap_ci95_low"
        upper_col = metric.replace("_mean", "") + "_group_bootstrap_ci95_high"
        if lower_col in data.columns and upper_col in data.columns:
            ci = data[data[method_col].astype(str) == method].groupby("budget")[
                [lower_col, upper_col]
            ].mean().reindex(group["budget"])
            if ci.notna().all().all():
                axis.fill_between(
                    group["budget"].to_numpy(float),
                    ci[lower_col].to_numpy(float),
                    ci[upper_col].to_numpy(float),
                    color=COLORS.get(method, "#4D4D4D"), alpha=0.09,
                    linewidth=0, zorder=1,
                )
        audit.setdefault("series", []).append({
            "method": method,
            "label": label,
            "budget_values": [int(value) if float(value).is_integer() else float(value)
                              for value in group["budget"].tolist()],
            "point_count": int(len(group)),
        })
        if exact:
            audit["exact_reference_note"] = (
                "Exact reference is plotted only at budgets present in the CSV; "
                "no values are interpolated or extrapolated."
            )
    axis.set_xlabel("Station-action budget")
    ylabel = metric.replace("_mean", "").replace("_", " ").title()
    axis.set_ylabel(ylabel)
    axis.set_ylim(0.0, 1.02)
    axis.set_xticks(sorted(data["budget"].unique()))
    axis.grid(axis="y", alpha=0.14)
    axis.legend(frameon=False, ncol=1, loc="lower right")
    axis.set_title(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=400)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", default="geographical_group")
    args = parser.parse_args()

    data, method_col, _, metric = _read_data(args.planning_csv, args.scope)
    present = sorted(data[method_col].astype(str).unique())
    baseline_methods = _ordered(present, BASELINE_ORDER)
    ablation_methods = _ordered(present, ABLATION_ORDER)
    if not baseline_methods:
        raise ValueError(f"no planning-baseline methods found; available: {present}")
    if not ablation_methods:
        raise ValueError(f"no ablation methods found; available: {present}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "input_csv": str(args.planning_csv.resolve()),
        "scope": args.scope,
        "metric": metric,
        "available_methods": present,
        "baseline_methods": baseline_methods,
        "ablation_methods": ablation_methods,
        "series": [],
    }
    _plot(
        data, method_col, metric, baseline_methods,
        args.output_dir / "figure5a_planning_baselines",
        "Planning baselines across station-action budgets", audit,
    )
    # Keep the audit entries separate by figure while retaining one file.
    first_series = audit.pop("series")
    audit["planning_baselines_series"] = first_series
    audit["series"] = []
    _plot(
        data, method_col, metric, ablation_methods,
        args.output_dir / "figure5b_planning_ablations",
        "Model ablations across station-action budgets", audit,
    )
    audit["model_ablations_series"] = audit.pop("series")
    audit["notes"] = [
        "Each legend entry corresponds to at least one observed CSV series.",
        "Missing budgets are left missing; lines are not extended through absent observations.",
        "Exact reference is labelled as available-budget-only; the plotted x values are read directly from the CSV.",
    ]
    (args.output_dir / "figure5_split_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
