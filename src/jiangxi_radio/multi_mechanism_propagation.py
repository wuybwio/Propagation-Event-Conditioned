"""Mechanism-aware engineering propagation components.

This module is intentionally explicit about its evidence level.  It provides a
fast, deterministic teacher for simulation experiments; it is not a full
electromagnetic ray tracer.  FSPL remains the reference term, while secondary
mechanisms add multi-edge diffraction, two-ray ground interaction, and a
terrain-roughness proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .environment_features import knife_edge_diffraction_loss_db


@dataclass(frozen=True)
class MultiMechanismScenario:
    """One reproducible propagation scenario used for sensitivity analysis."""

    name: str
    frequency_mhz: float = 2500.0
    reflection_coefficient: float = -0.65
    ground_roughness_m: float = 0.15
    environment_scale: float = 1.0
    max_reflection_gain_db: float = 6.0

    def __post_init__(self) -> None:
        values = (
            self.frequency_mhz,
            self.ground_roughness_m,
            self.environment_scale,
            self.max_reflection_gain_db,
        )
        if not self.name or not all(math.isfinite(float(v)) for v in values):
            raise ValueError("scenario fields must be finite")
        if self.frequency_mhz <= 0.0 or self.ground_roughness_m < 0.0:
            raise ValueError("frequency and roughness must be non-negative/positive")
        if not 0.0 <= abs(self.reflection_coefficient) <= 1.0:
            raise ValueError("reflection coefficient magnitude must be in [0,1]")
        if self.environment_scale < 0.0 or self.max_reflection_gain_db < 0.0:
            raise ValueError("environment scale and gain limit must be non-negative")


def default_multi_mechanism_scenarios() -> tuple[MultiMechanismScenario, ...]:
    """Return conservative low/nominal/high environmental scenarios."""

    return (
        MultiMechanismScenario(
            "low_clutter", frequency_mhz=1800.0, reflection_coefficient=-0.45,
            ground_roughness_m=0.05, environment_scale=0.6,
        ),
        MultiMechanismScenario(
            "nominal", frequency_mhz=2500.0, reflection_coefficient=-0.65,
            ground_roughness_m=0.15, environment_scale=1.0,
        ),
        MultiMechanismScenario(
            "high_clutter", frequency_mhz=3500.0, reflection_coefficient=-0.80,
            ground_roughness_m=0.35, environment_scale=1.4,
        ),
    )


def combine_multi_edge_losses_db(
    edge_loss_db: np.ndarray,
    *,
    secondary_weight: float = 0.35,
    tertiary_weight: float = 0.15,
    max_loss_db: float = 60.0,
) -> np.ndarray:
    """Combine ranked knife-edge losses into a bounded multi-edge envelope.

    The strongest edge is retained fully and secondary edges are down-weighted
    to avoid double-counting the same obstruction.  This is an auditable
    engineering envelope, not a claim of exact Epstein--Peterson diffraction.
    The last axis is interpreted as the candidate-edge axis.
    """

    values = np.asarray(edge_loss_db, dtype=float)
    if values.ndim < 1 or not np.isfinite(values).all():
        raise ValueError("edge losses must be finite and have an edge axis")
    if not 0.0 <= secondary_weight <= 1.0 or not 0.0 <= tertiary_weight <= 1.0:
        raise ValueError("secondary and tertiary weights must be in [0,1]")
    if not math.isfinite(max_loss_db) or max_loss_db <= 0.0:
        raise ValueError("max_loss_db must be positive")
    ranked = np.sort(np.clip(values, 0.0, max_loss_db), axis=-1)[..., ::-1]
    result = ranked[..., 0]
    if ranked.shape[-1] > 1:
        result = result + secondary_weight * ranked[..., 1]
    if ranked.shape[-1] > 2:
        result = result + tertiary_weight * ranked[..., 2]
    return np.clip(result, 0.0, max_loss_db)


def two_ray_reflection_gain_db(
    horizontal_distance_m: np.ndarray | float,
    tx_height_m: np.ndarray | float,
    rx_height_m: np.ndarray | float,
    *,
    frequency_mhz: float,
    reflection_coefficient: float = -0.65,
    ground_roughness_m: float = 0.15,
    max_gain_db: float = 6.0,
) -> np.ndarray:
    """Return bounded two-ray ground-interaction gain relative to direct path."""

    distance = np.asarray(horizontal_distance_m, dtype=float)
    tx = np.asarray(tx_height_m, dtype=float)
    rx = np.asarray(rx_height_m, dtype=float)
    distance, tx, rx = np.broadcast_arrays(distance, tx, rx)
    if not np.isfinite(distance).all() or not np.isfinite(tx).all() or not np.isfinite(rx).all():
        raise ValueError("two-ray inputs must be finite")
    if frequency_mhz <= 0.0 or ground_roughness_m < 0.0:
        raise ValueError("frequency and roughness must be valid")
    if abs(reflection_coefficient) > 1.0 or max_gain_db < 0.0:
        raise ValueError("reflection coefficient or gain bound is invalid")
    distance = np.maximum(distance, 1.0)
    wavelength = 299_792_458.0 / (float(frequency_mhz) * 1e6)
    direct = np.sqrt(distance**2 + (tx - rx) ** 2)
    reflected = np.sqrt(distance**2 + (tx + rx) ** 2)
    phase = 2.0 * np.pi * (reflected - direct) / wavelength
    grazing = np.arctan2(tx + rx, distance)
    roughness_factor = np.exp(
        -0.5 * (4.0 * np.pi * ground_roughness_m * np.sin(grazing) / wavelength) ** 2
    )
    amplitude = float(reflection_coefficient) * roughness_factor * direct / reflected
    combined = np.abs(1.0 + amplitude * np.exp(-1j * phase)) ** 2
    gain = 10.0 * np.log10(np.maximum(combined, 1e-8))
    return np.clip(gain, -30.0, float(max_gain_db)).astype(np.float32)


def terrain_roughness_loss_db(
    terrain_m: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    path_length_m: np.ndarray,
    *,
    scale: float = 1.0,
    db_per_km: float = 0.8,
) -> np.ndarray:
    """Estimate a transparent terrain-roughness loss from DEM gradients."""

    terrain = np.asarray(terrain_m, dtype=float)
    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    path = np.asarray(path_length_m, dtype=float)
    if terrain.shape != (len(y), len(x)) or not np.isfinite(terrain).all():
        raise ValueError("terrain and axes are not aligned/finite")
    if not np.isfinite(path).all() or scale < 0.0 or db_per_km < 0.0:
        raise ValueError("path and roughness parameters must be valid")
    gy, gx = np.gradient(terrain, y, x, edge_order=1)
    roughness = np.sqrt(gx * gx + gy * gy)
    local = float(np.nanmean(roughness))
    return (float(scale) * float(db_per_km) * path / 1000.0 * local / (1.0 + local)).astype(np.float32)


def synthesize_multi_mechanism_teacher(
    coarse_dbm: np.ndarray,
    edge_loss_db: np.ndarray,
    reflection_gain_db: np.ndarray,
    environment_loss_db: np.ndarray,
) -> dict[str, np.ndarray]:
    """Combine mechanism components without double-counting their signs."""

    coarse, edge, reflection, environment = (
        np.asarray(value, dtype=float)
        for value in (coarse_dbm, edge_loss_db, reflection_gain_db, environment_loss_db)
    )
    if not (coarse.shape == edge.shape == reflection.shape == environment.shape):
        raise ValueError("all propagation components must have the same shape")
    if not all(np.isfinite(value).all() for value in (coarse, edge, reflection, environment)):
        raise ValueError("propagation components must be finite")
    teacher = coarse - np.maximum(edge, 0.0) + reflection - np.maximum(environment, 0.0)
    return {
        "multi_edge_loss_db": np.maximum(edge, 0.0).astype(np.float32),
        "reflection_gain_db": reflection.astype(np.float32),
        "environment_loss_db": np.maximum(environment, 0.0).astype(np.float32),
        "teacher_dbm": teacher.astype(np.float32),
    }
