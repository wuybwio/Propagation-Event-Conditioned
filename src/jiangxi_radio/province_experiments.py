"""Contracts for geographically grouped Jiangxi validation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .frozen_v3_release import (
    greedy_select_sites,
    planning_metrics,
    union_detection_probability,
)


ROLE_TO_SPLIT = {"train": 0, "validation": 1, "test": 2, "calibration": 3}


@dataclass(frozen=True)
class VariantSpec:
    name: str
    use_events: bool = True
    use_local: bool = True
    use_spectral: bool = True
    use_event_consistency: bool = True
    use_multilevel_supervision: bool = True
    use_uncertainty: bool = True


_VARIANTS = {
    "full_v3": VariantSpec("full_v3"),
    "no_multilevel": VariantSpec("no_multilevel", use_multilevel_supervision=False),
    "no_events": VariantSpec("no_events", use_events=False, use_event_consistency=False),
    "no_local": VariantSpec("no_local", use_local=False),
    "no_spectral": VariantSpec("no_spectral", use_spectral=False),
    "no_multilevel_no_spectral": VariantSpec(
        "no_multilevel_no_spectral",
        use_spectral=False,
        use_multilevel_supervision=False,
    ),
    "no_event_consistency": VariantSpec(
        "no_event_consistency", use_event_consistency=False
    ),
    "deterministic": VariantSpec("deterministic", use_uncertainty=False),
    "cnn_residual": VariantSpec(
        "cnn_residual", use_events=False, use_spectral=False,
        use_event_consistency=False,
    ),
    "fno_residual": VariantSpec(
        "fno_residual", use_events=False, use_local=False,
        use_event_consistency=False,
    ),
}


def variant_names() -> tuple[str, ...]:
    return tuple(_VARIANTS)


def variant_spec(name: str) -> VariantSpec:
    try:
        return _VARIANTS[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown province experiment variant {name!r}; expected {list(_VARIANTS)}"
        ) from exc


def _station_coordinates(stations: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    required = {"station_id", "longitude", "latitude"}
    missing = sorted(required - set(stations.columns))
    if missing:
        raise ValueError(f"station table is missing columns: {missing}")
    frame = stations.copy()
    frame["station_id"] = frame["station_id"].astype(str)
    for name in ("longitude", "latitude"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if frame["station_id"].duplicated().any():
        raise ValueError("station identifiers must be unique")
    if not np.isfinite(frame[["longitude", "latitude"]].to_numpy(float)).all():
        raise ValueError("station coordinates must be finite")
    frame = frame.drop_duplicates(["longitude", "latitude"], keep="first").reset_index(drop=True)
    latitude = frame["latitude"].to_numpy(float)
    longitude = frame["longitude"].to_numpy(float)
    mean_latitude = float(np.mean(latitude))
    coordinates = np.column_stack((
        (longitude - np.mean(longitude)) * 111.32 * np.cos(np.radians(mean_latitude)),
        (latitude - np.mean(latitude)) * 110.57,
    ))
    return frame, coordinates


def _farthest_initial_centres(points: np.ndarray, count: int) -> np.ndarray:
    first = int(np.argmax(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))
    selected = [first]
    while len(selected) < count:
        squared = np.sum((points[:, None, :] - points[selected][None, :, :]) ** 2, axis=2)
        distance = squared.min(axis=1)
        distance[selected] = -1.0
        selected.append(int(np.argmax(distance)))
    return points[selected].copy()


def _spatial_groups(points: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    centres = _farthest_initial_centres(points, count)
    labels = np.zeros(len(points), dtype=np.int64)
    for _ in range(100):
        distances = np.sum((points[:, None, :] - centres[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels) and _:
            break
        labels = new_labels
        for index in range(count):
            members = points[labels == index]
            if len(members):
                centres[index] = members.mean(axis=0)
    if len(set(labels.tolist())) != count:
        raise ValueError("spatial grouping produced an empty geographical group")
    order = sorted(range(count), key=lambda index: (-centres[index, 1], centres[index, 0]))
    remap = {old: new for new, old in enumerate(order)}
    labels = np.asarray([remap[int(value)] for value in labels], dtype=np.int64)
    return labels, centres[order]


def _representatives(points: np.ndarray, count: int) -> list[int]:
    if len(points) < count:
        raise ValueError("a geographical group contains fewer stations than requested scenes")
    centroid = points.mean(axis=0)
    chosen = [int(np.argmin(np.sum((points - centroid) ** 2, axis=1)))]
    while len(chosen) < count:
        distances = np.sum((points[:, None, :] - points[chosen][None, :, :]) ** 2, axis=2)
        nearest = distances.min(axis=1)
        nearest[chosen] = -1.0
        chosen.append(int(np.argmax(nearest)))
    return chosen


def select_spatial_scene_centres(
    stations: pd.DataFrame,
    *,
    group_count: int = 5,
    scenes_per_group: int = 4,
    seed: int = 2026,
) -> pd.DataFrame:
    """Select deterministic, geographically dispersed centres from real stations."""
    if group_count < 4 or scenes_per_group < 1:
        raise ValueError("at least four groups and one scene per group are required")
    frame, points = _station_coordinates(stations)
    if len(frame) < group_count * scenes_per_group:
        raise ValueError("not enough unique station coordinates for the requested design")
    # The algorithm is deterministic. The seed is recorded to make later
    # randomized extensions explicit rather than silently changing the design.
    labels, _ = _spatial_groups(points, group_count)
    rows: list[dict[str, object]] = []
    for group_index in range(group_count):
        members = np.flatnonzero(labels == group_index)
        local = _representatives(points[members], scenes_per_group)
        for scene_index, local_index in enumerate(local, start=1):
            source_index = int(members[local_index])
            source = frame.iloc[source_index]
            rows.append({
                "scene_id": f"region_{group_index + 1:02d}_scene_{scene_index:02d}",
                "geo_group": f"region_{group_index + 1:02d}",
                "source_station_id": str(source["station_id"]),
                "center_longitude": float(source["longitude"]),
                "center_latitude": float(source["latitude"]),
                "station_grade": source.get("grade", np.nan),
                "selection_seed": int(seed),
            })
    return pd.DataFrame(rows)


def _manifest_core(manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"scene_id", "geo_group"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"scene manifest is missing columns: {missing}")
    frame = manifest.copy()
    frame["scene_id"] = frame["scene_id"].astype(str)
    frame["geo_group"] = frame["geo_group"].astype(str)
    if frame["scene_id"].duplicated().any() or (frame["scene_id"].str.len() == 0).any():
        raise ValueError("scene identifiers must be nonempty and unique")
    if (frame["geo_group"].str.len() == 0).any():
        raise ValueError("geographical groups must be nonempty")
    return frame


def build_grouped_roles(manifest: pd.DataFrame) -> pd.DataFrame:
    """Build outer folds with group-disjoint train/validation/calibration/test roles."""
    frame = _manifest_core(manifest)
    groups = sorted(frame["geo_group"].unique())
    if len(groups) < 4:
        raise ValueError("four geographical groups are required for four disjoint roles")
    rows: list[pd.DataFrame] = []
    for index, test_group in enumerate(groups):
        validation_group = groups[(index + 1) % len(groups)]
        calibration_group = groups[(index + 2) % len(groups)]
        roles = {}
        for group in groups:
            roles[group] = (
                "test" if group == test_group else
                "validation" if group == validation_group else
                "calibration" if group == calibration_group else
                "train"
            )
        current = frame[["scene_id", "geo_group"]].copy()
        current.insert(0, "fold", test_group)
        current["role"] = current["geo_group"].map(roles)
        rows.append(current)
    return pd.concat(rows, ignore_index=True)


def _load_scene(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        data = {key: np.asarray(source[key]) for key in source.files}
    required = {
        "x_m", "y_m", "z_m", "terrain_m", "surface_m", "air_mask",
        "station_xyz_m", "coarse_dbm", "teacher_dbm", "metadata_json",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"{path} is missing station-volume fields: {missing}")
    metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if metadata.get("schema") != "station_volume_v1":
        raise ValueError(f"{path} is not a station_volume_v1 dataset")
    return data


def pack_grouped_scenes(
    entries: Mapping[str, str | Path],
    assignments: pd.DataFrame,
    *,
    fold: str,
) -> dict[str, np.ndarray]:
    """Pack scene datasets while assigning every scene exactly one geographic role."""
    frame = assignments.copy()
    if "fold" in frame.columns:
        frame = frame.loc[frame["fold"].astype(str) == str(fold)].copy()
    required = {"scene_id", "geo_group", "role"}
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"fold assignments are empty or missing columns: {missing}")
    frame["scene_id"] = frame["scene_id"].astype(str)
    if frame["scene_id"].duplicated().any():
        raise ValueError("a scene has more than one role in the selected fold")
    if set(frame["role"]) != set(ROLE_TO_SPLIT):
        raise ValueError("selected fold must contain train, validation, calibration and test scenes")
    if set(frame["scene_id"]) != set(entries):
        raise ValueError("dataset entries and fold assignments contain different scenes")

    repeated_names = ("x_m", "y_m", "z_m", "terrain_m", "surface_m", "air_mask")
    case_names = ("station_xyz_m", "coarse_dbm", "teacher_dbm")
    collected: dict[str, list[np.ndarray]] = {
        name: [] for name in (*repeated_names, *case_names)
    }
    scene_ids: list[np.ndarray] = []
    geo_groups: list[np.ndarray] = []
    source_indices: list[np.ndarray] = []
    split_values: list[np.ndarray] = []
    source_metadata: dict[str, object] = {}
    roles = frame.set_index("scene_id")
    common_shape = None
    for scene_id in sorted(entries):
        path = Path(entries[scene_id])
        data = _load_scene(path)
        cases = len(data["station_xyz_m"])
        shape = tuple(np.asarray(data["coarse_dbm"]).shape[1:])
        if common_shape is None:
            common_shape = shape
        elif shape != common_shape:
            raise ValueError("all province scenes must use the same voxel shape")
        case_ndim = {
            "x_m": 2, "y_m": 2, "z_m": 2,
            "terrain_m": 3, "surface_m": 3, "air_mask": 4,
        }
        for name in repeated_names:
            value = np.asarray(data[name])
            collected[name].append(
                value if value.ndim == case_ndim[name] else
                np.repeat(value[None], cases, axis=0)
            )
        for name in case_names:
            collected[name].append(np.asarray(data[name]))
        role = str(roles.loc[scene_id, "role"])
        scene_ids.append(np.full(cases, scene_id))
        geo_groups.append(np.full(cases, str(roles.loc[scene_id, "geo_group"])))
        source_indices.append(np.arange(cases, dtype=np.int64))
        split_values.append(np.full(cases, ROLE_TO_SPLIT[role], dtype=np.int64))
        source_metadata[scene_id] = json.loads(str(np.asarray(data["metadata_json"]).item()))
    result = {name: np.concatenate(values, axis=0) for name, values in collected.items()}
    result.update(
        split=np.concatenate(split_values),
        scene_id=np.concatenate(scene_ids),
        geo_group=np.concatenate(geo_groups),
        source_station_index=np.concatenate(source_indices),
    )
    metadata = {
        "schema": "station_volume_v1",
        "evidence": "simulation_teacher_agreement_not_measurement_accuracy",
        "measurement_validation": False,
        "multi_scene": True,
        "province_validation": True,
        "fold": str(fold),
        "split_policy": "geographical_group_roles",
        "split_ids": ROLE_TO_SPLIT,
        "calibration_split_id": ROLE_TO_SPLIT["calibration"],
        "source_scenes": source_metadata,
    }
    result["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    return result


def _farthest_first_sites(station_xyz_m: np.ndarray, budget: int) -> list[int]:
    points = np.asarray(station_xyz_m, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2 or not np.isfinite(points).all():
        raise ValueError("station coordinates must be a finite [sites, dimensions] array")
    if budget < 1 or budget > len(points):
        raise ValueError("planning budget is outside the candidate range")
    horizontal = points[:, :2]
    centroid = horizontal.mean(axis=0)
    selected = [int(np.argmin(np.sum((horizontal - centroid) ** 2, axis=1)))]
    while len(selected) < budget:
        distance = np.sum(
            (horizontal[:, None, :] - horizontal[selected][None, :, :]) ** 2,
            axis=2,
        ).min(axis=1)
        distance[selected] = -1.0
        selected.append(int(np.argmax(distance)))
    return selected


def exact_select_sites(
    candidate_probabilities: np.ndarray,
    budget: int,
    mask: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    """Solve the cardinality-constrained mean-union objective by enumeration."""
    probabilities = np.asarray(candidate_probabilities, dtype=float)
    valid = np.asarray(mask, dtype=bool)
    if probabilities.ndim < 2 or probabilities.shape[1:] != valid.shape:
        raise ValueError("candidate fields and exact-planning mask do not align")
    if budget < 1 or budget > len(probabilities) or not np.any(valid):
        raise ValueError("exact-planning budget or mask is invalid")
    best: tuple[int, ...] | None = None
    best_score = -np.inf
    best_union: np.ndarray | None = None
    for chosen in combinations(range(len(probabilities)), budget):
        union = union_detection_probability(probabilities[list(chosen)], axis=0)
        score = float(np.mean(union[valid]))
        if score > best_score + 1e-15:
            best, best_score, best_union = chosen, score, union
    assert best is not None and best_union is not None
    return list(best), best_union.astype(np.float32)


def evaluate_candidate_planning(
    selection_fields: Mapping[str, np.ndarray],
    reference_probabilities: np.ndarray,
    station_xyz_m: np.ndarray,
    mask: np.ndarray,
    *,
    budgets: list[int] | tuple[int, ...],
    probability_threshold: float = 0.9,
    random_repeats: int = 100,
    seed: int = 2026,
    exact_budget_max: int = 3,
) -> tuple[list[dict[str, object]], dict[str, list[int]]]:
    """Select on each method field and evaluate every selection on one reference field."""
    reference = np.asarray(reference_probabilities, dtype=float)
    stations = np.asarray(station_xyz_m, dtype=float)
    valid = np.asarray(mask, dtype=bool)
    if reference.ndim < 2 or reference.shape[0] != len(stations):
        raise ValueError("reference probabilities must have one field per candidate site")
    if reference.shape[1:] != valid.shape or not np.any(valid):
        raise ValueError("reference fields and planning mask do not align")
    if np.any(~np.isfinite(reference)) or np.any((reference < 0.0) | (reference > 1.0)):
        raise ValueError("reference probabilities must be finite and lie in [0, 1]")
    fields = {str(name): np.asarray(value, dtype=float) for name, value in selection_fields.items()}
    reserved = {"ReferenceGreedy", "ExactReference", "Random", "Uniform"}
    if reserved.intersection(fields):
        raise ValueError("reserved planning method name")
    for name, value in fields.items():
        if value.shape != reference.shape:
            raise ValueError(f"selection field {name!r} does not align with the reference")
    budget_values = sorted(set(int(value) for value in budgets))
    if not budget_values or budget_values[0] < 1 or budget_values[-1] > len(stations):
        raise ValueError("planning budgets are empty or outside the candidate range")
    if random_repeats < 1:
        raise ValueError("random_repeats must be positive")

    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    selections: dict[str, list[int]] = {}

    def record(method: str, budget: int, repeat: int, chosen: list[int]) -> None:
        union = union_detection_probability(reference[chosen], axis=0)
        metric = planning_metrics(union, valid, probability_threshold)
        records.append({
            "method": method,
            "budget": int(budget),
            "repeat": int(repeat),
            "selected_indices": ";".join(str(value) for value in chosen),
            **metric,
        })
        selections[f"{method}:{budget}:{repeat}"] = list(chosen)

    for budget in budget_values:
        for method, field in fields.items():
            chosen, _ = greedy_select_sites(field, budget, valid)
            record(method, budget, 0, chosen)
        reference_greedy, _ = greedy_select_sites(reference, budget, valid)
        record("ReferenceGreedy", budget, 0, reference_greedy)
        if budget <= exact_budget_max:
            exact, _ = exact_select_sites(reference, budget, valid)
            record("ExactReference", budget, 0, exact)
        record("Uniform", budget, 0, _farthest_first_sites(stations, budget))
        for repeat in range(random_repeats):
            chosen = sorted(rng.choice(len(stations), size=budget, replace=False).tolist())
            record("Random", budget, repeat, chosen)
    return records, selections
