from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Literal

import numpy as np

from .detection import at_least_k_probability, batch_at_least_k_probability, batch_correlated_at_least_k_probability


ActionKind = Literal["upgrade", "relocate", "new"]


@dataclass(frozen=True)
class StationAction:
    action_id: str
    kind: ActionKind
    cost: float
    probabilities: np.ndarray
    target_station_index: int | None = None
    exclusive_group: str | None = None

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities, dtype=float)
        object.__setattr__(self, "probabilities", probabilities)
        if self.exclusive_group is None:
            parts = str(self.action_id).split(":")
            if self.kind == "new" and len(parts) >= 2:
                object.__setattr__(self, "exclusive_group", f"candidate:{parts[1]}")
            elif self.kind == "relocate" and len(parts) >= 3:
                object.__setattr__(self, "exclusive_group", f"candidate:{parts[2]}")
        if self.kind not in {"upgrade", "relocate", "new"}:
            raise ValueError(f"unsupported action kind: {self.kind}")
        if self.cost < 0:
            raise ValueError("action cost must be non-negative")
        if probabilities.ndim != 1 or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("action probabilities must be a vector in [0, 1]")
        if self.kind == "new" and self.target_station_index is not None:
            raise ValueError("new-station action must not target an existing station")
        if self.kind != "new" and self.target_station_index is None:
            raise ValueError("upgrade and relocate actions must target an existing station")
        if self.exclusive_group is not None and not str(self.exclusive_group):
            raise ValueError("exclusive_group must be a non-empty string when provided")


@dataclass(frozen=True)
class OptimizationSolution:
    selected_action_ids: tuple[str, ...]
    total_cost: float
    point_probabilities: np.ndarray
    total_probability_gap: float
    feasible: bool


def _validate_baseline(probabilities: np.ndarray) -> np.ndarray:
    baseline = np.asarray(probabilities, dtype=float)
    if baseline.ndim != 2 or baseline.shape[1] < 1:
        raise ValueError("baseline probabilities must have shape [route points, stations]")
    if np.any(~np.isfinite(baseline)) or np.any((baseline < 0) | (baseline > 1)):
        raise ValueError("baseline probabilities must be finite and in [0, 1]")
    return baseline


def apply_actions(
    baseline_probabilities: np.ndarray,
    actions: list[StationAction] | tuple[StationAction, ...],
) -> np.ndarray:
    updated = _validate_baseline(baseline_probabilities).copy()
    occupied_targets: set[int] = set()
    occupied_groups: set[str] = set()
    additions: list[np.ndarray] = []
    for action in actions:
        if len(action.probabilities) != len(updated):
            raise ValueError("action and baseline must contain the same route points")
        if action.kind == "new":
            additions.append(action.probabilities)
            if action.exclusive_group is not None:
                if str(action.exclusive_group) in occupied_groups:
                    raise ValueError("conflicting actions share an exclusive location group")
                occupied_groups.add(str(action.exclusive_group))
            continue
        target = int(action.target_station_index)
        if target < 0 or target >= updated.shape[1]:
            raise ValueError("action targets an unknown existing station")
        if target in occupied_targets:
            raise ValueError("conflicting actions target the same existing station")
        occupied_targets.add(target)
        if action.exclusive_group is not None:
            occupied_groups.add(str(action.exclusive_group))
        updated[:, target] = action.probabilities
    if additions:
        updated = np.column_stack([updated, *additions])
    return updated


def _joint_point_probabilities(
    station_probabilities: np.ndarray,
    required_k: np.ndarray,
    *,
    shadow_correlation: float = 0.0,
    correlation_samples: int = 4096,
    correlation_seed: int = 0,
) -> np.ndarray:
    return batch_correlated_at_least_k_probability(
        station_probabilities, required_k,
        correlation=shadow_correlation,
        samples=correlation_samples,
        seed=correlation_seed,
    )


def _evaluate_selection(
    baseline: np.ndarray,
    selected: list[StationAction],
    required_k: np.ndarray,
    threshold: float,
    weights: np.ndarray,
    *,
    shadow_correlation: float = 0.0,
    correlation_samples: int = 4096,
    correlation_seed: int = 0,
) -> tuple[np.ndarray, float, bool]:
    station_probabilities = apply_actions(baseline, selected)
    joint = _joint_point_probabilities(
        station_probabilities, required_k,
        shadow_correlation=shadow_correlation,
        correlation_samples=correlation_samples,
        correlation_seed=correlation_seed,
    )
    gap = float(np.sum(weights * np.maximum(0.0, threshold - joint)))
    return joint, gap, bool(np.all(joint >= threshold - 1e-12))


def _validate_exclusive_groups(actions: Sequence[StationAction]) -> None:
    groups = [
        str(action.exclusive_group)
        for action in actions
        if action.exclusive_group is not None
    ]
    duplicates = {group for group in groups if groups.count(group) > 1}
    if duplicates:
        raise ValueError(
            "conflicting actions share exclusive location groups: "
            + ", ".join(sorted(duplicates))
        )


def greedy_probability_gap_optimization(
    baseline_probabilities: np.ndarray,
    actions: list[StationAction] | tuple[StationAction, ...],
    required_k: np.ndarray,
    probability_threshold: float,
    budget: float,
    point_weights: np.ndarray | None = None,
    *,
    shadow_correlation: float = 0.0,
    correlation_samples: int = 4096,
    correlation_seed: int = 0,
) -> OptimizationSolution:
    baseline = _validate_baseline(baseline_probabilities)
    redundancy = np.asarray(required_k, dtype=int)
    if redundancy.shape != (len(baseline),) or np.any(redundancy < 1):
        raise ValueError("required_k must contain one positive value per route point")
    if not 0 < probability_threshold < 1:
        raise ValueError("probability threshold must be between 0 and 1")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    weights = (
        np.ones(len(baseline), dtype=float)
        if point_weights is None
        else np.asarray(point_weights, dtype=float)
    )
    if weights.shape != (len(baseline),) or np.any(weights < 0):
        raise ValueError("point_weights must be a non-negative vector over route points")

    remaining = list(actions)
    selected: list[StationAction] = []
    total_cost = 0.0
    current_joint, current_gap, feasible = _evaluate_selection(
        baseline, selected, redundancy, probability_threshold, weights,
        shadow_correlation=shadow_correlation,
        correlation_samples=correlation_samples,
        correlation_seed=correlation_seed,
    )

    while not feasible:
        ranked: list[tuple[int, float, float, int, StationAction, np.ndarray, float, bool]] = []
        occupied_targets = {
            int(action.target_station_index)
            for action in selected
            if action.target_station_index is not None
        }
        occupied_groups = {
            str(action.exclusive_group)
            for action in selected
            if action.exclusive_group is not None
        }
        for order, action in enumerate(remaining):
            if total_cost + action.cost > budget + 1e-12:
                continue
            if action.target_station_index in occupied_targets:
                continue
            if action.exclusive_group is not None and str(action.exclusive_group) in occupied_groups:
                continue
            try:
                trial_joint, trial_gap, trial_feasible = _evaluate_selection(
                    baseline,
                    [*selected, action],
                    redundancy,
                    probability_threshold,
                    weights,
                    shadow_correlation=shadow_correlation,
                    correlation_samples=correlation_samples,
                    correlation_seed=correlation_seed,
                )
            except ValueError:
                continue
            reduction = current_gap - trial_gap
            if reduction <= 1e-12:
                continue
            newly_covered = int(
                np.count_nonzero(
                    (current_joint < probability_threshold)
                    & (trial_joint >= probability_threshold)
                )
            )
            efficiency = reduction / max(action.cost, 1e-12)
            ranked.append(
                (
                    newly_covered,
                    efficiency,
                    reduction,
                    -order,
                    action,
                    trial_joint,
                    trial_gap,
                    trial_feasible,
                )
            )
        if not ranked:
            break
        best = max(ranked, key=lambda item: item[:4])
        action = best[4]
        selected.append(action)
        remaining.remove(action)
        total_cost += action.cost
        if action.exclusive_group is not None:
            occupied_groups.add(str(action.exclusive_group))
        current_joint, current_gap, feasible = best[5], best[6], best[7]

    if feasible:
        for action in sorted(selected, key=lambda item: item.cost, reverse=True):
            trial = [item for item in selected if item.action_id != action.action_id]
            trial_joint, trial_gap, trial_feasible = _evaluate_selection(
                baseline, trial, redundancy, probability_threshold, weights,
                shadow_correlation=shadow_correlation,
                correlation_samples=correlation_samples,
                correlation_seed=correlation_seed,
            )
            if trial_feasible:
                selected = trial
                total_cost -= action.cost
                current_joint, current_gap = trial_joint, trial_gap

    _validate_exclusive_groups(selected)
    return OptimizationSolution(
        selected_action_ids=tuple(action.action_id for action in selected),
        total_cost=float(total_cost),
        point_probabilities=current_joint,
        total_probability_gap=float(current_gap),
        feasible=feasible,
    )


def joint_greedy_probability_gap_optimization(
    baseline_probabilities: Sequence[np.ndarray],
    actions: Sequence[StationAction],
    required_k: Sequence[np.ndarray],
    probability_threshold: float,
    budget: float,
    point_weights: Sequence[np.ndarray] | None = None,
    *,
    shadow_correlation: float = 0.0,
    correlation_samples: int = 4096,
    correlation_seed: int = 0,
) -> OptimizationSolution:
    """Optimize one shared station network across multiple route scenarios.

    Each scenario contributes a block of route points. Actions must provide one
    concatenated probability vector over all blocks, so a selected action is
    paid for once and can improve several routes simultaneously.
    """
    if len(baseline_probabilities) == 0:
        raise ValueError("at least one baseline scenario is required")
    if len(baseline_probabilities) != len(required_k):
        raise ValueError("baseline and required_k must contain the same scenarios")
    baseline_blocks = [_validate_baseline(block) for block in baseline_probabilities]
    point_blocks = [len(block) for block in baseline_blocks]
    required_blocks = [np.asarray(block, dtype=int) for block in required_k]
    for index, (required, point_count) in enumerate(zip(required_blocks, point_blocks)):
        if required.shape != (point_count,):
            raise ValueError(f"required_k block {index} has an invalid shape")
    baseline = np.vstack(baseline_blocks)
    required = np.concatenate(required_blocks)
    if point_weights is None:
        weights = None
    else:
        if len(point_weights) != len(baseline_blocks):
            raise ValueError("point_weights must contain one block per scenario")
        weight_blocks = [np.asarray(block, dtype=float) for block in point_weights]
        for index, (weight, point_count) in enumerate(zip(weight_blocks, point_blocks)):
            if weight.shape != (point_count,):
                raise ValueError(f"point_weights block {index} has an invalid shape")
        weights = np.concatenate(weight_blocks)
    total_points = len(baseline)
    for action in actions:
        if len(action.probabilities) != total_points:
            raise ValueError("joint action probabilities must cover all scenarios")
    return greedy_probability_gap_optimization(
        baseline_probabilities=baseline,
        actions=list(actions),
        required_k=required,
        probability_threshold=probability_threshold,
        budget=budget,
        point_weights=weights,
        shadow_correlation=shadow_correlation,
        correlation_samples=correlation_samples,
        correlation_seed=correlation_seed,
    )
