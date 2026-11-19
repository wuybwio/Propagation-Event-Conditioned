"""Describe existing ablations across metrics; do not select on test scores."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path("/mnt/e/project/guihua")
SOURCE = ROOT / "outputs/joint_ablation_20261106/analysis/group_metrics.csv"
OUT = ROOT / "outputs/multimetric_ablation_20261107"
METRICS = ["mae_db", "rmse_db", "probability_mse", "blind_iou",
           "balanced_accuracy", "false_coverage_rate"]
LABELS = {
    "full_v3": "Complete",
    "no_multilevel": "No multi-level supervision",
    "no_spectral": "No spectral operator",
    "no_multilevel_no_spectral": "Joint removal",
}


def number(value, metric):
    precision = 3 if metric.endswith("_db") else 4
    return "$" + format(value, f"+.{precision}f") + "$"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE)
    assert not data.duplicated(["variant", "geo_group"]).any()
    subset = data[data.variant.isin(LABELS)].copy()
    assert len(subset) == 20 and subset[METRICS].notna().all().all()
    subset["configuration"] = subset.variant.map(LABELS)
    subset.to_csv(OUT / "all_configuration_group_metrics.csv", index=False)
    baseline = subset[subset.variant.eq("no_multilevel")].set_index("geo_group")
    spectral = subset[subset.variant.eq("no_spectral")].set_index("geo_group")
    differences = spectral[METRICS] - baseline[METRICS]
    assert len(differences) == 5
    differences.to_csv(OUT / "no_spectral_minus_no_multilevel.csv")
    counts = {metric: {
        "positive": int((differences[metric] > 0).sum()),
        "negative": int((differences[metric] < 0).sum()),
        "zero": int((differences[metric] == 0).sum()),
    } for metric in METRICS}
    displayed = ["mae_db", "rmse_db", "blind_iou",
                 "balanced_accuracy", "false_coverage_rate"]
    rows = []
    for group, values in differences.iterrows():
        group_number = int(group.rsplit("_", 1)[-1])
        cells = [f"Group {group_number}"] + [number(values[m], m) for m in displayed]
        rows.append(" & ".join(cells) + r"\\")
    average = differences.mean()
    cells = ["Mean"] + [number(average[m], m) for m in displayed]
    table = r"""\begin{table}[htbp]
\centering
\caption{Geographic-group trade-offs between two reduced configurations.
Entries are no-spectral minus no-multi-level-supervision configuration.
Negative differences favour no-spectral for MAE, RMSE and false-coverage rate;
positive differences favour it for blind IoU and balanced accuracy.}
\label{tab:ablation-metric-tradeoffs}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrr}
\toprule
Test group & $\Delta$ MAE & $\Delta$ RMSE & $\Delta$ Blind IoU & $\Delta$ BA & $\Delta$ FCR\\
\midrule
""" + "\n".join(rows) + "\n" + r"\midrule" + "\n" + " & ".join(cells) + r"\\" + "\n" + r"""\botrule
\end{tabular*}
\footnotetext{Power errors are in dB; classification differences are
dimensionless. BA: balanced accuracy; FCR: false-coverage rate.
Each group averages four scene means after averaging three seeds.
Classification uses the same mean-only probability mapping and fixed
operating point in every configuration. These are descriptive contrasts on
previously inspected test groups, not configuration-selection criteria or
independent significance tests.}
\end{table}
"""
    (OUT / "group_metric_tradeoffs.tex").write_text(table)
    report = {
        "source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "contrast": "no_spectral minus no_multilevel", "direction_counts": counts,
        "group_mean_differences": average.to_dict(),
        "new_training_runs": 0, "new_predictions": False,
        "validation_selection_rule_changed": False,
        "scope": "Descriptive multi-metric audit; no claim of task-optimal selection",
    }
    (OUT / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
