"""Layered propagation-model framework for province planning and local audit.

Mathematical budget used by the engineering main model (dB)::

    L_total = L_fspl + L_env + L_diffraction + L_antenna + X_shadow

where ``L_diffraction`` is either zero (FSPL ablation), a height-decayed grid
building prior, or the sampled Fresnel knife-edge envelope.  The grid prior and
knife-edge diffraction are mutually exclusive on one link to avoid double
counting the same buildings.

Tiering
-------
A. Province main model: auditable knife-edge Fresnel diffraction.
B. Local high-fidelity reference: externally supplied ray-tracing adapter.
C. Optimizer surrogate: neural radio-map trained on Tier-A labels, with Tier-B
   recheck on shortlisted candidates when available.

A local ray tracer without materials, measured antenna patterns and field data
is an inter-model reference, not ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping

import numpy as np

from .environment_features import (
    building_obstruction_loss_db,
    station_path_knife_edge_loss_db,
)


class EngineeringPropagationMode(str, Enum):
    """Ablation modes for the fast engineering model family."""

    FSPL = "fspl"
    GRID_BUILDING_PRIOR = "grid_building_prior"
    KNIFE_EDGE = "knife_edge"
    KNIFE_EDGE_UNCERTAINTY = "knife_edge_uncertainty"


@dataclass(frozen=True)
class PropagationScenario:
    """One physically plausible parameter scenario for sensitivity analysis."""

    name: str
    frequency_mhz: float
    shadow_std_db: float
    building_max_loss_db: float
    dem_offset_m: float = 0.0
    building_height_scale: float = 1.0
    estimated_height_buffer_m: float = 15.0

    def __post_init__(self) -> None:
        if self.frequency_mhz <= 0 or self.shadow_std_db <= 0:
            raise ValueError("frequency and shadow_std_db must be positive")
        if self.building_max_loss_db < 0 or self.building_height_scale <= 0:
            raise ValueError("building loss and height scale are invalid")
        if self.estimated_height_buffer_m < 0:
            raise ValueError("estimated_height_buffer_m must be non-negative")


def default_scenarios() -> tuple[PropagationScenario, ...]:
    """Return a conservative ensemble when weather/measurement data are absent."""
    return (
        PropagationScenario("700mhz_sigma4", 700.0, 4.0, 12.0, -10.0, 0.90, 8.0),
        PropagationScenario("2500mhz_sigma6", 2500.0, 6.0, 18.0, 0.0, 1.00, 15.0),
        PropagationScenario("4900mhz_sigma10", 4900.0, 10.0, 30.0, 10.0, 1.15, 25.0),
    )


def summarize_scenario_values(
    values: np.ndarray,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> dict[str, np.ndarray]:
    """Summarize scenario outputs without treating them as measurements."""
    array = np.asarray(values, dtype=float)
    if array.ndim < 1 or array.shape[0] < 2 or not np.isfinite(array).all():
        raise ValueError("values must contain at least two finite scenarios")
    if not quantiles or any(float(q) < 0.0 or float(q) > 1.0 for q in quantiles):
        raise ValueError("quantiles must be non-empty and lie in [0, 1]")
    return {f"p{int(q * 100):02d}": np.quantile(array, q, axis=0) for q in quantiles}


def _rankdata(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    ranks[order] = np.arange(1, len(array) + 1, dtype=float)
    # Average ranks for ties.
    sorted_vals = array[order]
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and sorted_vals[end] == sorted_vals[start]:
            end += 1
        if end - start > 1:
            avg = 0.5 * (start + 1 + end)
            ranks[order[start:end]] = avg
        start = end
    return ranks


def spearman_rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return Spearman rho for decision-score agreement."""
    x = np.asarray(left, dtype=float).reshape(-1)
    y = np.asarray(right, dtype=float).reshape(-1)
    if x.shape != y.shape or len(x) < 2:
        raise ValueError("spearman inputs must be aligned and contain >=2 values")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("spearman inputs must be finite")
    rx = _rankdata(x)
    ry = _rankdata(y)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def compare_with_reference(
    engineering_values: np.ndarray,
    reference_values: np.ndarray,
) -> dict[str, float]:
    """Compare the fast model with a local high-fidelity simulator output."""
    fast = np.asarray(engineering_values, dtype=float).reshape(-1)
    ref = np.asarray(reference_values, dtype=float).reshape(-1)
    if fast.shape != ref.shape or len(fast) == 0 or not np.isfinite(fast).all() or not np.isfinite(ref).all():
        raise ValueError("engineering and reference values must be aligned and finite")
    error = fast - ref
    correlation = (
        float(np.corrcoef(fast, ref)[0, 1])
        if len(fast) > 1 and float(np.std(fast)) and float(np.std(ref))
        else 0.0
    )
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
        "correlation": correlation,
        "spearman": spearman_rank_correlation(fast, ref),
    }


def compare_decision_metrics(
    engineering_scores: np.ndarray,
    reference_scores: np.ndarray,
    *,
    engineering_cost: float | None = None,
    reference_cost: float | None = None,
    engineering_coverage: float | None = None,
    reference_coverage: float | None = None,
    engineering_max_gap_km: float | None = None,
    reference_max_gap_km: float | None = None,
    engineering_runtime_s: float | None = None,
    reference_runtime_s: float | None = None,
) -> dict[str, float]:
    """Compare planning-relevant quantities, not only field RMSE."""
    metrics = compare_with_reference(engineering_scores, reference_scores)
    if engineering_cost is not None and reference_cost is not None:
        metrics["cost_delta"] = float(engineering_cost - reference_cost)
    if engineering_coverage is not None and reference_coverage is not None:
        metrics["coverage_delta"] = float(engineering_coverage - reference_coverage)
    if engineering_max_gap_km is not None and reference_max_gap_km is not None:
        metrics["max_gap_delta_km"] = float(engineering_max_gap_km - reference_max_gap_km)
    if (
        engineering_runtime_s is not None
        and reference_runtime_s is not None
        and engineering_runtime_s > 0
    ):
        metrics["runtime_speedup"] = float(reference_runtime_s / engineering_runtime_s)
    return metrics


def run_reference_adapter(
    points: np.ndarray,
    simulator: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Run an externally supplied local ray-tracing adapter with shape checks."""
    query = np.asarray(points, dtype=float)
    if query.ndim != 2 or query.shape[0] == 0 or not np.isfinite(query).all():
        raise ValueError("points must be a non-empty finite [n, d] array")
    values = np.asarray(simulator(query), dtype=float)
    if values.ndim == 0 or values.shape[0] != len(query) or not np.isfinite(values).all():
        raise ValueError("reference simulator must return one finite value per point")
    return values


def compute_additional_loss_db(
    mode: EngineeringPropagationMode | str,
    *,
    points: np.ndarray,
    station_coordinates: np.ndarray | None = None,
    building_features: np.ndarray | None = None,
    axes: tuple[np.ndarray, np.ndarray] | None = None,
    station_height_m: np.ndarray | float | None = None,
    flight_height_m: float = 120.0,
    frequency_mhz: float = 2500.0,
    max_loss_db: float = 18.0,
    samples_per_km: float = 2.0,
    max_path_distance_km: float = 20.0,
    effective_earth_radius_factor: float = 4.0 / 3.0,
    estimated_height_buffer_m: float = 15.0,
    terrain_elevation_m: np.ndarray | None = None,
    station_ground_elevation_m: np.ndarray | float = 0.0,
    target_ground_elevation_m: np.ndarray | float | None = None,
    scenarios: tuple[PropagationScenario, ...] | None = None,
) -> np.ndarray | dict[str, np.ndarray]:
    """Return additional link loss for one ablation mode.

    ``knife_edge_uncertainty`` returns a dict of scenario name -> loss matrix.
    All other modes return a single ``[n_points, n_stations]`` or
    ``[n_points]`` array.
    """
    selected = EngineeringPropagationMode(mode)
    point_count = len(np.asarray(points, dtype=float))
    if selected is EngineeringPropagationMode.FSPL:
        if station_coordinates is None:
            return np.zeros(point_count, dtype=np.float32)
        return np.zeros((point_count, len(station_coordinates)), dtype=np.float32)

    if building_features is None:
        raise ValueError(f"{selected.value} requires building_features")

    if selected is EngineeringPropagationMode.GRID_BUILDING_PRIOR:
        prior = building_obstruction_loss_db(building_features, max_loss_db=max_loss_db)
        return prior * np.exp(-float(flight_height_m) / 300.0)

    if station_coordinates is None or axes is None or station_height_m is None:
        raise ValueError("knife-edge modes require stations, axes and station heights")

    if selected is EngineeringPropagationMode.KNIFE_EDGE:
        return station_path_knife_edge_loss_db(
            points,
            station_coordinates,
            building_features,
            axes,
            station_height_m=station_height_m,
            flight_height_m=float(flight_height_m),
            frequency_mhz=float(frequency_mhz),
            max_loss_db=float(max_loss_db),
            samples_per_km=float(samples_per_km),
            max_path_distance_km=float(max_path_distance_km),
            effective_earth_radius_factor=float(effective_earth_radius_factor),
            estimated_height_buffer_m=float(estimated_height_buffer_m),
            terrain_elevation_m=terrain_elevation_m,
            station_ground_elevation_m=station_ground_elevation_m,
            target_ground_elevation_m=target_ground_elevation_m,
        )

    ensemble = scenarios if scenarios is not None else default_scenarios()
    output: dict[str, np.ndarray] = {}
    for scenario in ensemble:
        scaled = np.asarray(building_features, dtype=float).copy()
        # Channels 1..3 are normalized height statistics; rescale conservatively.
        scaled[:, 1:4] = np.clip(scaled[:, 1:4] * scenario.building_height_scale, 0.0, 1.0)
        terrain = None if terrain_elevation_m is None else np.asarray(terrain_elevation_m, dtype=float) + scenario.dem_offset_m
        target_ground = (
            None
            if target_ground_elevation_m is None
            else np.asarray(target_ground_elevation_m, dtype=float) + scenario.dem_offset_m
        )
        output[scenario.name] = station_path_knife_edge_loss_db(
            points,
            station_coordinates,
            scaled,
            axes,
            station_height_m=station_height_m,
            flight_height_m=float(flight_height_m),
            frequency_mhz=float(scenario.frequency_mhz),
            max_loss_db=float(scenario.building_max_loss_db),
            samples_per_km=float(samples_per_km),
            max_path_distance_km=float(max_path_distance_km),
            effective_earth_radius_factor=float(effective_earth_radius_factor),
            estimated_height_buffer_m=float(scenario.estimated_height_buffer_m),
            terrain_elevation_m=terrain,
            station_ground_elevation_m=station_ground_elevation_m,
            target_ground_elevation_m=target_ground,
        )
    return output


def evidence_metadata(mode: EngineeringPropagationMode | str) -> Mapping[str, str]:
    """Return paper-facing evidence labels for a selected model tier."""
    selected = EngineeringPropagationMode(mode)
    labels = {
        EngineeringPropagationMode.FSPL: {
            "model_tier": "A_ablation",
            "evidence_class": "physics_scenario_fspl",
            "claim_limit": "open-space baseline only; no building/terrain diffraction",
        },
        EngineeringPropagationMode.GRID_BUILDING_PRIOR: {
            "model_tier": "A_ablation",
            "evidence_class": "physics_scenario_grid_building_prior",
            "claim_limit": "receiver-grid building prior; not link-specific diffraction",
        },
        EngineeringPropagationMode.KNIFE_EDGE: {
            "model_tier": "A_main",
            "evidence_class": "physics_scenario_knife_edge_fresnel",
            "claim_limit": "engineering dominant-obstacle envelope; not multipath ray tracing",
        },
        EngineeringPropagationMode.KNIFE_EDGE_UNCERTAINTY: {
            "model_tier": "A_main_uncertainty",
            "evidence_class": "physics_scenario_knife_edge_ensemble",
            "claim_limit": "parameter ensemble under missing weather/materials; not measurements",
        },
    }
    return labels[selected]
