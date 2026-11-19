from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import NormalDist

import numpy as np


def detection_probability(
    mean_margin_db: float | np.ndarray,
    std_margin_db: float | np.ndarray,
) -> float | np.ndarray:
    mean = np.asarray(mean_margin_db, dtype=float)
    std = np.asarray(std_margin_db, dtype=float)
    if np.any(std <= 0) or np.any(~np.isfinite(std)):
        raise ValueError("standard deviation must be finite and positive")
    z = mean / (std * math.sqrt(2.0))
    erf_values = np.vectorize(math.erf, otypes=[float])(z)
    probability = 0.5 * (1.0 + erf_values)
    if probability.ndim == 0:
        return float(probability)
    return probability


def at_least_k_probability(probabilities: Iterable[float], k: int) -> float:
    probs = np.asarray(tuple(probabilities), dtype=float)
    if np.any(~np.isfinite(probs)) or np.any((probs < 0) | (probs > 1)):
        raise ValueError("station probabilities must be in [0, 1]")
    if k <= 0:
        return 1.0
    if k > len(probs):
        return 0.0

    exact = np.zeros(len(probs) + 1, dtype=float)
    exact[0] = 1.0
    for probability in probs:
        for successes in range(len(probs), 0, -1):
            exact[successes] = (
                exact[successes] * (1.0 - probability)
                + exact[successes - 1] * probability
            )
        exact[0] *= 1.0 - probability
    return float(exact[k:].sum())


def batch_at_least_k_probability(probabilities: np.ndarray, required_k: np.ndarray) -> np.ndarray:
    """Vectorized Poisson-binomial at-least-k probability for route points."""
    values = np.asarray(probabilities, dtype=float)
    ks = np.asarray(required_k, dtype=int)
    if values.ndim != 2 or ks.shape != (len(values),):
        raise ValueError("probabilities must be [points, stations] and required_k [points]")
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("station probabilities must be in [0, 1]")
    if np.any(ks < 1):
        raise ValueError("required_k must be positive")
    max_required_k = int(np.max(ks))
    exact = np.zeros((len(values), max_required_k + 1), dtype=float)
    exact[:, 0] = 1.0
    for index in range(values.shape[1]):
        probability = values[:, index]
        upper = min(index + 1, max_required_k)
        if upper < max_required_k:
            exact[:, 1 : upper + 1] = (
                exact[:, 1 : upper + 1] * (1.0 - probability[:, None])
                + exact[:, :upper] * probability[:, None]
            )
        else:
            previous_boundary = exact[:, max_required_k - 1].copy()
            previous_absorbed = exact[:, max_required_k].copy()
            exact[:, 1:max_required_k] = (
                exact[:, 1:max_required_k] * (1.0 - probability[:, None])
                + exact[:, : max_required_k - 1] * probability[:, None]
            )
            exact[:, max_required_k] = previous_absorbed + previous_boundary * probability
        exact[:, 0] *= 1.0 - probability
    result = np.zeros(len(values), dtype=float)
    for k in range(1, max_required_k + 1):
        result[ks == k] = exact[ks == k, k:].sum(axis=1)
    return result


def correlated_at_least_k_probability(
    probabilities: Iterable[float],
    k: int,
    *,
    correlation: float = 0.0,
    samples: int = 4096,
    seed: int = 0,
) -> float:
    """Estimate an at-least-``k`` probability with correlated shadowing.

    A one-factor Gaussian copula preserves each station's marginal detection
    probability while introducing an equicorrelation between station events.
    This is a practical uncertainty analysis for the sparse Nanchang data: it
    does not invent a fitted covariance matrix, and the deterministic seed
    keeps station-plan comparisons reproducible.  ``correlation=0`` falls back
    to the exact Poisson-binomial result.
    """
    probs = np.asarray(tuple(probabilities), dtype=float)
    if np.any(~np.isfinite(probs)) or np.any((probs < 0) | (probs > 1)):
        raise ValueError("station probabilities must be in [0, 1]")
    if k <= 0:
        return 1.0
    if k > len(probs):
        return 0.0
    if not 0.0 <= correlation < 1.0:
        raise ValueError("correlation must be in [0, 1)")
    if samples < 128:
        raise ValueError("samples must be at least 128")
    if correlation == 0.0:
        return at_least_k_probability(probs, k)
    # Gaussian copula thresholds give Bernoulli marginals exactly up to the
    # Monte-Carlo integration error.  Clip endpoints to avoid +/-inf values.
    normal = NormalDist()
    thresholds = np.asarray(
        [normal.inv_cdf(float(np.clip(p, 1e-12, 1.0 - 1e-12))) for p in probs],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(samples)
    idiosyncratic = rng.standard_normal((samples, len(probs)))
    latent = np.sqrt(correlation) * common[:, None] + np.sqrt(1.0 - correlation) * idiosyncratic
    successes = np.count_nonzero(latent <= thresholds[None, :], axis=1)
    return float(np.mean(successes >= int(k)))


def batch_correlated_at_least_k_probability(
    probabilities: np.ndarray,
    required_k: np.ndarray,
    *,
    correlation: float = 0.0,
    samples: int = 4096,
    seed: int = 0,
) -> np.ndarray:
    """Vectorized wrapper for correlated route-point coverage probabilities."""
    values = np.asarray(probabilities, dtype=float)
    ks = np.asarray(required_k, dtype=int)
    if values.ndim != 2 or ks.shape != (len(values),):
        raise ValueError("probabilities must be [points, stations] and required_k [points]")
    if correlation == 0.0:
        return batch_at_least_k_probability(values, ks)
    return np.asarray(
        [
            correlated_at_least_k_probability(
                row, int(k), correlation=correlation, samples=samples, seed=seed + index
            )
            for index, (row, k) in enumerate(zip(values, ks, strict=True))
        ],
        dtype=float,
    )
