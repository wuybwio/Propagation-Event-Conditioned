"""Physics-consistent late fusion of frozen Path and Boundary experts.

The experts predict absolute received power, but fusion is performed in the
residual domain relative to the same coarse physical field.  This keeps the
fused field tied to the propagation baseline and avoids averaging outputs
that may have different learned calibration offsets.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def _aligned_finite(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    converted = tuple(np.asarray(value, dtype=np.float32) for value in arrays)
    shape = converted[0].shape
    if any(value.shape != shape for value in converted):
        raise ValueError("coarse, expert predictions and gate must be aligned")
    if any(not np.isfinite(value).all() for value in converted):
        raise ValueError("fusion inputs must be finite")
    return converted


def fuse_residual_fields(
    coarse: np.ndarray,
    path: np.ndarray,
    boundary: np.ndarray,
    gate: np.ndarray,
) -> np.ndarray:
    """Fuse absolute dBm predictions through their coarse-field residuals.

    ``gate=0`` selects Path and ``gate=1`` selects Boundary.  Gate values are
    deliberately validated rather than silently clipped, because clipping can
    hide a broken standardization or checkpoint at inference time.
    """
    coarse, path, boundary = _aligned_finite(coarse, path, boundary)
    gate = np.asarray(gate, dtype=np.float32)
    if gate.shape != coarse.shape:
        raise ValueError("gate must have the same shape as the expert fields")
    if not np.isfinite(gate).all() or np.any((gate < 0) | (gate > 1)):
        raise ValueError("gate must be finite and in [0, 1]")
    path_residual = path - coarse
    boundary_residual = boundary - coarse
    fused = coarse + (1.0 - gate) * path_residual + gate * boundary_residual
    if not np.isfinite(fused).all():
        raise ValueError("fused field is non-finite")
    return fused.astype(np.float32, copy=False)


def build_gate_features(
    coarse_dbm: np.ndarray,
    path_dbm: np.ndarray,
    boundary_dbm: np.ndarray,
    physical_features: np.ndarray | None = None,
    residual_scale_db: float = 60.0,
) -> np.ndarray:
    """Build inference-available features with layout ``[N,C,Z,Y,X]``.

    The first three channels are normalized Path residual, Boundary residual,
    and their absolute disagreement.  ``physical_features`` can be a single
    ``[N,Z,Y,X]`` field or ``[N,C,Z,Y,X]`` fields generated without teacher
    labels (for example ``input_volume`` channels).
    """
    if not math.isfinite(residual_scale_db) or residual_scale_db <= 0:
        raise ValueError("residual_scale_db must be finite and positive")
    coarse, path, boundary = _aligned_finite(coarse_dbm, path_dbm, boundary_dbm)
    path_residual = (path - coarse) / residual_scale_db
    boundary_residual = (boundary - coarse) / residual_scale_db
    disagreement = np.abs(path - boundary) / residual_scale_db
    features = [path_residual, boundary_residual, disagreement]
    if physical_features is not None:
        physical = np.asarray(physical_features, dtype=np.float32)
        if physical.ndim == coarse.ndim:
            physical = physical[:, None, ...]
        if physical.ndim != coarse.ndim + 1 or physical.shape[0] != coarse.shape[0] \
                or physical.shape[2:] != coarse.shape[1:]:
            raise ValueError("physical_features must align with [N,Z,Y,X] fields")
        if not np.isfinite(physical).all():
            raise ValueError("physical_features must be finite")
        features.extend(physical[:, channel] for channel in range(physical.shape[1]))
    return np.stack(features, axis=1).astype(np.float32, copy=False)


def detection_probability_from_dbm(
    power_dbm: np.ndarray,
    threshold_dbm: float = -100.0,
    shadow_std_db: float = 6.0,
) -> np.ndarray:
    """Convert dBm to soft coverage probability using Gaussian shadowing."""
    if not math.isfinite(threshold_dbm) or not math.isfinite(shadow_std_db) \
            or shadow_std_db <= 0:
        raise ValueError("finite threshold and positive shadow standard deviation required")
    power = np.asarray(power_dbm, dtype=np.float32)
    if not np.isfinite(power).all():
        raise ValueError("power_dbm must be finite")
    scaled = ((power.astype(np.float64) - threshold_dbm)
              / (shadow_std_db * math.sqrt(2.0))).ravel()
    values = np.fromiter((math.erf(float(value)) for value in scaled),
                         dtype=np.float64, count=scaled.size)
    return (0.5 * (1.0 + values)).reshape(power.shape).astype(np.float32)


class GateMLP(nn.Module):
    """Small spatially independent gate applied independently per voxel."""

    def __init__(self, input_dim: int, hidden: tuple[int, int] = (32, 16)):
        super().__init__()
        if input_dim < 1 or len(hidden) != 2 or any(int(value) < 1 for value in hidden):
            raise ValueError("input_dim and hidden widths must be positive")
        first, second = map(int, hidden)
        self.input_dim = int(input_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, first),
            nn.LayerNorm(first),
            nn.SiLU(),
            nn.Linear(first, second),
            nn.SiLU(),
            nn.Linear(second, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim < 2 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected [...,{self.input_dim}] gate features, got {tuple(features.shape)}"
            )
        return self.network(features)


class ContextGate3D(nn.Module):
    """Spatially contextual gate for fields shaped ``[N,C,Z,Y,X]``.

    Unlike :class:`GateMLP`, this model can use neighboring voxels when
    deciding whether the Path or Boundary residual is more reliable.  The
    final output is a bounded Boundary weight with shape ``[N,1,Z,Y,X]``.
    """

    def __init__(self, input_dim: int, hidden: int = 32):
        super().__init__()
        if input_dim < 1 or hidden < 4:
            raise ValueError("input_dim and hidden width must be positive")
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        groups = 8 if hidden % 8 == 0 else 1
        self.network = nn.Sequential(
            nn.Conv3d(self.input_dim, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv3d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 5 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"expected [N,{self.input_dim},Z,Y,X] features, got {tuple(features.shape)}"
            )
        return self.network(features)


def gate_total_variation(gate: torch.Tensor) -> torch.Tensor:
    """Anisotropic TV penalty for a gate tensor ``[N,1,Z,Y,X]``."""
    if gate.ndim != 5 or gate.shape[1] != 1:
        raise ValueError("gate must have shape [N,1,Z,Y,X]")
    terms = []
    for axis in (2, 3, 4):
        left = [slice(None)] * 5
        right = [slice(None)] * 5
        left[axis] = slice(1, None)
        right[axis] = slice(None, -1)
        terms.append(torch.abs(gate[tuple(left)] - gate[tuple(right)]).mean())
    return torch.stack(terms).mean()
