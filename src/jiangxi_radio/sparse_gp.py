"""Small dependency-free sparse RBF Gaussian-process-style residual model.

This is intentionally a practical post-calibration model rather than a
replacement for the neural map.  It keeps at most ``max_inducing_points``
observations and solves the kernel ridge posterior in NumPy, providing a
smooth residual mean and a distance-aware uncertainty estimate.
"""

from __future__ import annotations

import numpy as np


class SparseRBFResidualGP:
    """RBF kernel ridge regression with a GP-inspired predictive deviation."""

    def __init__(
        self,
        *,
        length_scale: float = 0.01,
        noise: float = 0.1,
        max_inducing_points: int = 512,
    ) -> None:
        if length_scale <= 0 or noise <= 0 or max_inducing_points <= 0:
            raise ValueError("length_scale, noise, and max_inducing_points must be positive")
        self.length_scale = float(length_scale)
        self.noise = float(noise)
        self.max_inducing_points = int(max_inducing_points)
        self._points: np.ndarray | None = None
        self._alpha: np.ndarray | None = None
        self._inverse: np.ndarray | None = None
        self._scale: float = 1.0

    def fit(self, points: np.ndarray, residual: np.ndarray) -> "SparseRBFResidualGP":
        coordinates = np.asarray(points, dtype=float)
        values = np.asarray(residual, dtype=float).reshape(-1)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("points must have shape [n, 2]")
        if len(coordinates) != len(values) or len(values) == 0:
            raise ValueError("points and residual must have the same non-zero length")
        valid = np.isfinite(coordinates).all(axis=1) & np.isfinite(values)
        coordinates, values = coordinates[valid], values[valid]
        if len(values) == 0:
            raise ValueError("points and residual contain no finite rows")
        indices = np.linspace(0, len(values) - 1, min(len(values), self.max_inducing_points), dtype=int)
        self._points = coordinates[indices]
        targets = values[indices]
        self._scale = max(float(np.std(targets)), 1e-6)
        kernel = self._kernel(self._points, self._points)
        regularized = kernel + np.eye(len(kernel)) * self.noise**2
        self._inverse = np.linalg.pinv(regularized)
        self._alpha = self._inverse @ targets
        return self

    def _kernel(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        delta = left[:, None, :] - right[None, :, :]
        distance2 = np.sum(delta * delta, axis=-1)
        return np.exp(-0.5 * distance2 / self.length_scale**2)

    def predict(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._points is None or self._alpha is None or self._inverse is None:
            raise RuntimeError("fit must be called before predict")
        query = np.asarray(points, dtype=float)
        if query.ndim != 2 or query.shape[1] != 2:
            raise ValueError("points must have shape [n, 2]")
        kernel = self._kernel(query, self._points)
        mean = kernel @ self._alpha
        leverage = np.einsum("ij,jk,ik->i", kernel, self._inverse, kernel)
        variance = np.maximum(0.0, 1.0 - leverage) * self._scale**2
        return mean.astype(float), np.sqrt(variance).astype(float)
