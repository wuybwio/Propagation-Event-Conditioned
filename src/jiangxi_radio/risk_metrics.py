"""Risk summaries for coverage under trajectory and parameter perturbations."""

from __future__ import annotations

import numpy as np


def summarize_coverage_risk(
    coverage_ratios: np.ndarray,
    target_ratio: float = 0.90,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Return mean, pass probability, lower quantile and CVaR shortfall.

    ``alpha`` controls the lower-tail size used for the shortfall CVaR. A value
    of 0.05 summarizes the worst 5% perturbation scenarios.
    """
    values = np.asarray(coverage_ratios, dtype=float).reshape(-1)
    if len(values) == 0 or np.any(~np.isfinite(values)):
        raise ValueError("coverage_ratios must be a non-empty finite vector")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("coverage ratios must be in [0, 1]")
    if not 0.0 < target_ratio <= 1.0:
        raise ValueError("target_ratio must be in (0, 1]")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    quantile = float(np.quantile(values, 0.05, method="linear"))
    tail_count = max(1, int(np.ceil(alpha * len(values))))
    lower_tail = np.sort(values)[:tail_count]
    shortfall = np.maximum(0.0, target_ratio - lower_tail)
    return {
        "mean_coverage_ratio": float(np.mean(values)),
        "std_coverage_ratio": float(np.std(values)),
        "probability_meeting_target": float(np.mean(values >= target_ratio)),
        "q05_coverage_ratio": quantile,
        "cvar_shortfall": float(np.mean(shortfall)),
    }
