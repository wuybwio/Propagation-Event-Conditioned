from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .candidates import filter_candidate_locations, generate_gap_candidates
from .config import ScenarioConfig
from .metrics import RouteCoverageResult, evaluate_route_coverage
from .optimization import (
    OptimizationSolution,
    StationAction,
    apply_actions,
    greedy_probability_gap_optimization,
)
from .physics_baseline import compute_route_station_probabilities
from .routes import assign_endpoint_priorities, haversine_distance_km


@dataclass(frozen=True)
class RoutePlanningResult:
    route_id: str
    height_m: int
    priorities: np.ndarray
    required_k: np.ndarray
    baseline_metrics: RouteCoverageResult
    candidates: pd.DataFrame
    solution: OptimizationSolution
    optimized_metrics: RouteCoverageResult
    new_stations: pd.DataFrame
    existing_station_actions: pd.DataFrame


def _priority_to_required_k(priorities: np.ndarray, config: ScenarioConfig) -> np.ndarray:
    return np.asarray(
        [config.coverage.required_stations(str(level)) for level in priorities],
        dtype=int,
    )


def _priority_weights(priorities: np.ndarray) -> np.ndarray:
    mapping = {"ordinary": 1.0, "critical": 2.0, "high_risk": 3.0}
    return np.asarray([mapping[str(level)] for level in priorities], dtype=float)


def _single_station_probabilities(
    route: np.ndarray,
    flight_height_m: int,
    longitude: float,
    latitude: float,
    grade: int,
    config: ScenarioConfig,
) -> np.ndarray:
    station = pd.DataFrame(
        {
            "station_id": ["action-station"],
            "longitude": [longitude],
            "latitude": [latitude],
            "grade": [grade],
        }
    )
    return compute_route_station_probabilities(
        route,
        flight_height_m,
        station,
        config.radio,
        config.station_grades,
        environment=config.environment,
    )[:, 0]


def _build_existing_actions(
    route: np.ndarray,
    flight_height_m: int,
    stations: pd.DataFrame,
    candidates: pd.DataFrame,
    config: ScenarioConfig,
    priorities: np.ndarray,
    max_relocation_candidates_per_station: int = 12,
) -> tuple[list[StationAction], pd.DataFrame]:
    """Create one optional upgrade/relocation action per existing station.

    Relocation is restricted to the best physics-scored route-corridor candidate
    for that station.  This keeps the joint action space tractable and makes the
    proposed coordinate auditable in the output table.
    """
    weights = _priority_weights(priorities)
    actions: list[StationAction] = []
    rows: list[dict[str, object]] = []
    for station_index, station in enumerate(stations.itertuples(index=False)):
        station_id = str(station.station_id)
        current_grade = int(station.grade)
        current_probability = _single_station_probabilities(
            route,
            flight_height_m,
            float(station.longitude),
            float(station.latitude),
            current_grade,
            config,
        )
        selected_action = "retain"
        selected_cost = 0.0
        selected_lon = float(station.longitude)
        selected_lat = float(station.latitude)
        selected_grade = current_grade
        selected_target: str | None = None

        if current_grade > 2:
            upgraded_grade = current_grade - 1
            upgrade_probability = _single_station_probabilities(
                route,
                flight_height_m,
                float(station.longitude),
                float(station.latitude),
                upgraded_grade,
                config,
            )
            actions.append(
                StationAction(
                    action_id=f"upgrade:{station_id}",
                    kind="upgrade",
                    cost=config.cost.upgrade,
                    probabilities=upgrade_probability,
                    target_station_index=station_index,
                )
            )
        else:
            upgraded_grade = None

        if len(candidates):
            # Restrict relocation evaluation to nearby corridor candidates. New
            # station actions still use the complete candidate set. This keeps
            # the joint action space tractable for statewide runs while
            # preserving physically plausible relocation targets.
            nearby_candidates = sorted(
                candidates.itertuples(index=False),
                key=lambda candidate: haversine_distance_km(
                    float(station.longitude),
                    float(station.latitude),
                    float(candidate.longitude),
                    float(candidate.latitude),
                ),
            )[:max_relocation_candidates_per_station]
            scored_candidates: list[tuple[float, object, np.ndarray]] = []
            for candidate in nearby_candidates:
                candidate_probability = _single_station_probabilities(
                    route,
                    flight_height_m,
                    float(candidate.longitude),
                    float(candidate.latitude),
                    current_grade,
                    config,
                )
                gain = float(np.sum(weights * (candidate_probability - current_probability)))
                scored_candidates.append((gain, candidate, candidate_probability))
            gain, best_candidate, best_probability = max(scored_candidates, key=lambda item: item[0])
            if gain > 1e-9:
                selected_target = str(best_candidate.candidate_id)
                actions.append(
                    StationAction(
                        action_id=f"relocate:{station_id}:{selected_target}",
                        kind="relocate",
                        cost=config.cost.relocate,
                        probabilities=best_probability,
                        target_station_index=station_index,
                    )
                )

        rows.append(
            {
                "station_id": station_id,
                "route_id": "",
                "action": selected_action,
                "cost": selected_cost,
                "source_longitude": float(station.longitude),
                "source_latitude": float(station.latitude),
                "source_grade": current_grade,
                "target_longitude": selected_lon,
                "target_latitude": selected_lat,
                "target_grade": selected_grade,
                "candidate_id": selected_target or "",
                "available_upgrade_grade": upgraded_grade if current_grade > 2 else "",
            }
        )
    return actions, pd.DataFrame.from_records(rows)


def plan_route_network(
    route_id: str,
    route_points: np.ndarray,
    flight_height_m: int,
    stations: pd.DataFrame,
    config: ScenarioConfig,
    geographic_bounds: tuple[float, float, float, float],
    high_risk_radius_km: float,
    critical_radius_km: float,
    candidate_spacing_km: float,
    lateral_offsets_km: tuple[float, ...],
    minimum_station_distance_km: float,
    new_station_grade: int,
) -> RoutePlanningResult:
    route = np.asarray(route_points, dtype=float)
    priorities = assign_endpoint_priorities(
        route[:, 2],
        high_risk_radius_km=high_risk_radius_km,
        critical_radius_km=critical_radius_km,
    )
    required_k = _priority_to_required_k(priorities, config)
    baseline_probabilities = compute_route_station_probabilities(
        route,
        flight_height_m,
        stations,
        config.radio,
        config.station_grades,
        environment=config.environment,
    )
    baseline_metrics = evaluate_route_coverage(
        baseline_probabilities,
        required_k,
        route[:, 2],
        config.coverage.probability_threshold,
    )

    candidates = generate_gap_candidates(
        route,
        ~baseline_metrics.covered,
        route_id=route_id,
        height_m=flight_height_m,
        longitudinal_spacing_km=candidate_spacing_km,
        lateral_offsets_km=lateral_offsets_km,
    )
    candidates = filter_candidate_locations(
        candidates,
        geographic_bounds,
        stations,
        minimum_station_distance_km,
    )
    if len(candidates):
        candidates = (
            candidates.assign(
                _lon_key=candidates["longitude"].round(6),
                _lat_key=candidates["latitude"].round(6),
            )
            .drop_duplicates(["_lon_key", "_lat_key"])
            .drop(columns=["_lon_key", "_lat_key"])
            .reset_index(drop=True)
        )

    actions: list[StationAction] = []
    for candidate in candidates.itertuples(index=False):
        candidate_station = pd.DataFrame(
            {
                "station_id": [candidate.candidate_id],
                "longitude": [candidate.longitude],
                "latitude": [candidate.latitude],
                "grade": [new_station_grade],
            }
        )
        probabilities = compute_route_station_probabilities(
            route,
            flight_height_m,
            candidate_station,
            config.radio,
            config.station_grades,
            environment=config.environment,
        )[:, 0]
        actions.append(
            StationAction(
                action_id=f"new:{candidate.candidate_id}",
                kind="new",
                cost=config.cost.new_station,
                probabilities=probabilities,
            )
        )

    existing_actions, existing_action_rows = _build_existing_actions(
        route,
        flight_height_m,
        stations,
        candidates,
        config,
        priorities,
    )
    actions.extend(existing_actions)

    solution = greedy_probability_gap_optimization(
        baseline_probabilities,
        actions,
        required_k,
        config.coverage.probability_threshold,
        config.cost.budget,
        point_weights=_priority_weights(priorities),
    )
    selected_by_id = {action.action_id: action for action in actions}
    selected_actions = [selected_by_id[action_id] for action_id in solution.selected_action_ids]
    optimized_probabilities = apply_actions(baseline_probabilities, selected_actions)
    optimized_metrics = evaluate_route_coverage(
        optimized_probabilities,
        required_k,
        route[:, 2],
        config.coverage.probability_threshold,
    )

    selected_candidate_ids = {
        action_id.removeprefix("new:") for action_id in solution.selected_action_ids
    }
    new_stations = candidates.loc[
        candidates["candidate_id"].isin(selected_candidate_ids)
    ].copy()
    if len(new_stations):
        new_stations["grade"] = int(new_station_grade)
        new_stations["cost"] = float(config.cost.new_station)
    else:
        new_stations = new_stations.assign(
            grade=pd.Series(dtype=int),
            cost=pd.Series(dtype=float),
        )

    existing_action_rows["route_id"] = route_id
    for row_index, row in existing_action_rows.iterrows():
        station_id = str(row["station_id"])
        selected = [
            action
            for action in selected_actions
            if action.target_station_index == row_index
        ]
        if selected:
            action = selected[0]
            existing_action_rows.loc[row_index, "action"] = action.kind
            existing_action_rows.loc[row_index, "cost"] = float(action.cost)
            if action.kind == "upgrade":
                existing_action_rows.loc[row_index, "target_grade"] = int(stations.iloc[row_index]["grade"]) - 1
                existing_action_rows.loc[row_index, "available_upgrade_grade"] = int(stations.iloc[row_index]["grade"]) - 1
            elif action.kind == "relocate":
                candidate_id = action.action_id.rsplit(":", 1)[-1]
                candidate_row = candidates.loc[candidates["candidate_id"] == candidate_id]
                if len(candidate_row):
                    candidate = candidate_row.iloc[0]
                    existing_action_rows.loc[row_index, "candidate_id"] = candidate_id
                    existing_action_rows.loc[row_index, "target_longitude"] = float(candidate["longitude"])
                    existing_action_rows.loc[row_index, "target_latitude"] = float(candidate["latitude"])
        else:
            existing_action_rows.loc[row_index, "action"] = "retain"
            existing_action_rows.loc[row_index, "cost"] = 0.0

    return RoutePlanningResult(
        route_id=route_id,
        height_m=int(flight_height_m),
        priorities=priorities,
        required_k=required_k,
        baseline_metrics=baseline_metrics,
        candidates=candidates,
        solution=solution,
        optimized_metrics=optimized_metrics,
        new_stations=new_stations.reset_index(drop=True),
        existing_station_actions=existing_action_rows.reset_index(drop=True),
    )
