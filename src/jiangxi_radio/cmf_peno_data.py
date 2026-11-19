"""Dataset construction and validation for CMF-PENO experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from .propagation_events import (
    EVENT_FEATURE_NAMES,
    EVENT_SCHEMA,
    RF_EVENT_SCHEMA,
    EVENT_TYPE_NAMES,
    build_propagation_events,
)
from .multi_mechanism_propagation import (
    MultiMechanismScenario,
    combine_multi_edge_losses_db,
    default_multi_mechanism_scenarios,
    synthesize_multi_mechanism_teacher,
    terrain_roughness_loss_db,
    two_ray_reflection_gain_db,
)

CMF_PENO_DATASET_SCHEMA = "cmf_peno_dataset_v1"
CMF_PENO_RF_DATASET_SCHEMA = "cmf_peno_rf_dataset_v2"


def _metadata(data: Mapping[str, np.ndarray]) -> dict:
    if "metadata_json" not in data:
        raise ValueError("base dataset is missing metadata_json")
    try:
        value = np.asarray(data["metadata_json"]).item()
        result = json.loads(str(value))
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("invalid base metadata_json") from error
    if result.get("schema") != "station_volume_v1":
        raise ValueError("base dataset must use station_volume_v1")
    if result.get("measurement_validation") is not False:
        raise ValueError("CMF-PENO builder currently accepts simulation-only data")
    return result


def _case_axis(data: Mapping[str, np.ndarray], name: str, index: int) -> np.ndarray:
    value = np.asarray(data[name])
    return np.asarray(value[index] if value.ndim == 2 else value, dtype=float)


def _case_field(data: Mapping[str, np.ndarray], name: str, index: int) -> np.ndarray:
    value = np.asarray(data[name])
    return np.asarray(value[index] if value.ndim == 3 else value, dtype=float)


def _bilinear_sample(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    sample_x: float,
    sample_y: float,
) -> float:
    """Sample one aligned 2-D field without using a scene-wide elevation."""

    ix = int(np.clip(np.searchsorted(x, sample_x, side="right") - 1, 0, len(x) - 2))
    iy = int(np.clip(np.searchsorted(y, sample_y, side="right") - 1, 0, len(y) - 2))
    ax = float((sample_x - x[ix]) / (x[ix + 1] - x[ix]))
    ay = float((sample_y - y[iy]) / (y[iy + 1] - y[iy]))
    return float(
        (1.0 - ax) * (1.0 - ay) * field[iy, ix]
        + ax * (1.0 - ay) * field[iy, ix + 1]
        + (1.0 - ax) * ay * field[iy + 1, ix]
        + ax * ay * field[iy + 1, ix + 1]
    )


def _validate_base(data: Mapping[str, np.ndarray]) -> tuple[int, tuple[int, int, int]]:
    required = {
        "x_m", "y_m", "z_m", "terrain_m", "surface_m", "air_mask",
        "station_xyz_m", "split", "coarse_dbm", "teacher_dbm", "metadata_json",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"base dataset is missing fields: {missing}")
    _metadata(data)
    stations = np.asarray(data["station_xyz_m"], dtype=float)
    if stations.ndim != 2 or stations.shape[1] != 3 or not np.isfinite(stations).all():
        raise ValueError("invalid station_xyz_m")
    cases = len(stations)
    z = _case_axis(data, "z_m", 0)
    y = _case_axis(data, "y_m", 0)
    x = _case_axis(data, "x_m", 0)
    shape = (len(z), len(y), len(x))
    for name in ("coarse_dbm", "teacher_dbm"):
        value = np.asarray(data[name])
        if value.shape != (cases, *shape) or not np.isfinite(value).all():
            raise ValueError(f"invalid {name}")
    split = np.asarray(data["split"])
    split_values = set(split.tolist())
    if split.shape != (cases,) or not {0, 1, 2}.issubset(split_values) or not split_values.issubset({0, 1, 2, 3}):
        raise ValueError("train/validation/test splits are required; optional calibration split is 3")
    for index in range(cases):
        current_shape = tuple(
            len(_case_axis(data, name, index)) for name in ("z_m", "y_m", "x_m")
        )
        if current_shape != shape:
            raise ValueError("all cases must use the same volume shape")
    return cases, shape


def _high_fidelity_field(
    base: Mapping[str, np.ndarray],
    high: Mapping[str, np.ndarray],
    cases: int,
    shape: tuple[int, int, int],
) -> np.ndarray:
    for name in ("high_fidelity_dbm", "teacher_dbm", "predicted_dbm"):
        if name in high:
            field = np.asarray(high[name], dtype=np.float32)
            break
    else:
        raise ValueError("high-fidelity data needs high_fidelity_dbm or teacher_dbm")
    if field.shape != (cases, *shape) or not np.isfinite(field).all():
        raise ValueError("high-fidelity field is not finite/aligned")
    for name in ("x_m", "y_m", "z_m"):
        if name not in high:
            raise ValueError(f"high-fidelity data is missing {name} axes")
        if not np.allclose(np.asarray(high[name]), np.asarray(base[name]), rtol=0.0, atol=1e-5):
            raise ValueError("high-fidelity axes do not match base axes")
    if "station_xyz_m" not in high or not np.allclose(
        np.asarray(high["station_xyz_m"]),
        np.asarray(base["station_xyz_m"]),
        rtol=0.0,
        atol=1e-4,
    ):
        raise ValueError("high-fidelity station order does not match base data")
    return field


def build_cmf_peno_dataset(
    base: Mapping[str, np.ndarray],
    *,
    high_fidelity: Mapping[str, np.ndarray] | None = None,
    samples: int = 32,
    max_events: int = 4,
    frequency_mhz: float = 2500.0,
    progress: bool = False,
) -> dict[str, np.ndarray]:
    """Enrich a station-volume dataset with events and fidelity levels."""
    cases, shape = _validate_base(base)
    event_features = []
    event_type = []
    event_mask = []
    counterfactual_gain = []
    for index in range(cases):
        result = build_propagation_events(
            _case_axis(base, "x_m", index),
            _case_axis(base, "y_m", index),
            _case_axis(base, "z_m", index),
            _case_field(base, "terrain_m", index),
            _case_field(base, "surface_m", index),
            np.asarray(base["station_xyz_m"][index], dtype=float),
            samples=samples,
            max_events=max_events,
            frequency_mhz=frequency_mhz,
        )
        event_features.append(result["event_features"])
        event_type.append(result["event_type"])
        event_mask.append(result["event_mask"])
        counterfactual_gain.append(result["counterfactual_gain_db"])
        if progress:
            print(f"event_case={index + 1}/{cases}", flush=True)

    fidelity_names = ["fspl", "geometric_teacher"]
    fidelity = [
        np.asarray(base["coarse_dbm"], dtype=np.float32),
        np.asarray(base["teacher_dbm"], dtype=np.float32),
    ]
    if high_fidelity is not None:
        fidelity.append(_high_fidelity_field(base, high_fidelity, cases, shape))
        fidelity_names.append("high_fidelity_teacher")
    fidelity_dbm = np.stack(fidelity, axis=1)

    base_meta = _metadata(base)
    metadata = {
        "schema": CMF_PENO_DATASET_SCHEMA,
        "evidence": "simulation_only",
        "measurement_validation": False,
        "base_schema": base_meta["schema"],
        "event_schema": EVENT_SCHEMA,
        "event_type_names": list(EVENT_TYPE_NAMES),
        "event_feature_names": list(EVENT_FEATURE_NAMES),
        "event_samples": int(samples),
        "max_events": int(max_events),
        "frequency_mhz": float(frequency_mhz),
        "fidelity_names": fidelity_names,
        "high_fidelity_available": high_fidelity is not None,
        "counterfactual_definition": "remove strongest sampled obstructing event",
        "unsupported_without_external_paths": [
            "reflection", "scattering", "transmission", "complex_phase",
        ],
        "base_metadata": base_meta,
    }
    output = {key: np.asarray(value) for key, value in base.items() if key != "metadata_json"}
    output.update(
        event_features=np.asarray(event_features, dtype=np.float32),
        event_type=np.asarray(event_type, dtype=np.uint8),
        event_mask=np.asarray(event_mask, dtype=bool),
        counterfactual_event_gain_db=np.asarray(counterfactual_gain, dtype=np.float32),
        fidelity_dbm=fidelity_dbm.astype(np.float32),
        fidelity_available=np.ones(len(fidelity_names), dtype=bool),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return output


def build_cmf_peno_rf_dataset(
    base: Mapping[str, np.ndarray],
    *,
    samples: int = 32,
    max_events: int = 4,
    frequency_mhz: float = 2500.0,
    scenarios: tuple[MultiMechanismScenario, ...] | None = None,
    use_multi_edge: bool = True,
    use_reflection: bool = True,
    progress: bool = False,
) -> dict[str, np.ndarray]:
    """Build a multi-mechanism teacher dataset from the existing volume data.

    The nominal event tensor is retained for the neural model.  Scenario
    teachers vary frequency, reflection and terrain-roughness assumptions and
    are stored separately for uncertainty/sensitivity experiments.
    """

    cases, shape = _validate_base(base)
    scenario_values = scenarios if scenarios is not None else default_multi_mechanism_scenarios()
    if not use_reflection:
        scenario_values = tuple(replace(value, reflection_coefficient=0.0) for value in scenario_values)
    if not scenario_values:
        raise ValueError("at least one propagation scenario is required")
    event_features, event_type, event_mask, counterfactual = [], [], [], []
    scenario_teachers, scenario_components = [], []
    for index in range(cases):
        x = _case_axis(base, "x_m", index)
        y = _case_axis(base, "y_m", index)
        z = _case_axis(base, "z_m", index)
        terrain = _case_field(base, "terrain_m", index)
        surface = _case_field(base, "surface_m", index)
        station = np.asarray(base["station_xyz_m"][index], dtype=float)
        events = build_propagation_events(
            x, y, z, terrain, surface, station,
            samples=samples, max_events=max_events, frequency_mhz=frequency_mhz,
        )
        event_features.append(events["event_features"])
        event_type.append(events["event_type"])
        event_mask.append(events["event_mask"])
        counterfactual.append(events["counterfactual_gain_db"])
        edge_values = np.where(
            events["event_mask"], events["event_features"][:, 11] * 60.0, 0.0
        )
        edge_loss = combine_multi_edge_losses_db(
            np.moveaxis(edge_values, 0, -1),
            secondary_weight=0.35 if use_multi_edge else 0.0,
            tertiary_weight=0.15 if use_multi_edge else 0.0,
            max_loss_db=60.0,
        )
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        horizontal = np.sqrt((xx - station[0]) ** 2 + (yy - station[1]) ** 2)
        path_length = np.sqrt(horizontal**2 + (zz - station[2]) ** 2)
        # Two-ray heights are AGL-like values; absolute DEM datum must not
        # change the reflection phase geometry.
        station_ground = _bilinear_sample(
            terrain, x, y, float(station[0]), float(station[1])
        )
        tx_agl = np.full_like(horizontal, max(station[2] - station_ground, 1.0))
        rx_agl = np.maximum(zz - terrain[None, :, :], 1.0)
        terrain_loss_base = terrain_roughness_loss_db(
            terrain, x, y, path_length, scale=1.0
        )
        scenario_fields, component_fields = [], []
        coarse = np.asarray(base["coarse_dbm"][index], dtype=float)
        for scenario in scenario_values:
            frequency_adjustment_db = 20.0 * np.log10(
                float(scenario.frequency_mhz) / float(frequency_mhz)
            )
            scenario_coarse = coarse - frequency_adjustment_db
            reflection = two_ray_reflection_gain_db(
                horizontal,
                tx_agl,
                rx_agl,
                frequency_mhz=scenario.frequency_mhz,
                reflection_coefficient=scenario.reflection_coefficient,
                ground_roughness_m=scenario.ground_roughness_m,
                max_gain_db=scenario.max_reflection_gain_db,
            )
            environment = terrain_loss_base * float(scenario.environment_scale)
            components = synthesize_multi_mechanism_teacher(
                scenario_coarse, edge_loss, reflection, environment
            )
            scenario_fields.append(components["teacher_dbm"])
            component_fields.append(np.stack([
                components["multi_edge_loss_db"],
                components["reflection_gain_db"],
                components["environment_loss_db"],
            ], axis=0))
        scenario_teachers.append(np.asarray(scenario_fields, dtype=np.float32))
        scenario_components.append(np.asarray(component_fields, dtype=np.float32))
        if progress:
            print(f"rf_case={index + 1}/{cases}", flush=True)
    scenario_teachers_array = np.asarray(scenario_teachers, dtype=np.float32)
    scenario_components_array = np.asarray(scenario_components, dtype=np.float32)
    nominal_index = next(
        (i for i, value in enumerate(scenario_values) if value.name == "nominal"),
        min(1, len(scenario_values) - 1),
    )
    base_event_array = np.asarray(event_features, dtype=np.float32)
    augmented_events = []
    for index, events in enumerate(base_event_array):
        rank = np.broadcast_to(
            (np.arange(max_events, dtype=np.float32) / max(float(max_events), 1.0))[:, None, None, None],
            events.shape[:1] + events.shape[2:],
        )
        valid = np.asarray(event_mask[index], dtype=np.float32)
        augmented_events.append(np.concatenate([
            events,
            rank[:, None, ...],
            valid[:, None, ...],
        ], axis=1))
    fidelity = np.stack([
        np.asarray(base["coarse_dbm"], dtype=np.float32),
        np.asarray(base["teacher_dbm"], dtype=np.float32),
        scenario_teachers_array[:, nominal_index],
    ], axis=1)
    base_meta = _metadata(base)
    metadata = {
        "schema": CMF_PENO_RF_DATASET_SCHEMA,
        "evidence": "simulation_teacher_agreement_not_measurement_accuracy",
        "measurement_validation": False,
        "base_schema": base_meta["schema"],
        "event_schema": RF_EVENT_SCHEMA,
        "event_type_names": list(EVENT_TYPE_NAMES),
        "event_feature_names": list(EVENT_FEATURE_NAMES) + [
            "event_rank", "event_valid",
        ],
        "model_input_event_feature_names": list(EVENT_FEATURE_NAMES) + [
            "event_rank", "event_valid",
        ],
        "event_feature_count": len(EVENT_FEATURE_NAMES) + 2,
        "event_samples": int(samples),
        "max_events": int(max_events),
        "base_frequency_mhz": float(frequency_mhz),
        "fidelity_names": ["fspl", "geometric_teacher", "multi_mechanism_teacher"],
        "scenario_names": [scenario.name for scenario in scenario_values],
        "scenario_parameters": [scenario.__dict__ for scenario in scenario_values],
        "nominal_scenario_index": int(nominal_index),
        "use_multi_edge": bool(use_multi_edge),
        "use_reflection": bool(use_reflection),
        "mechanism_channels": ["multi_edge_loss_db", "reflection_gain_db", "environment_loss_db"],
        "model_input_uses_teacher_components": False,
        "diagnostic_only_fields": [
            "rf_mechanism_db", "scenario_teacher_dbm", "scenario_mechanism_db",
        ],
        "teacher_definition": "FSPL - bounded multi-edge loss + two-ray reflection gain - terrain roughness proxy",
        "not_modelled": ["measured materials", "measured antenna patterns", "full ray tracing", "measured RSRP"],
        "base_metadata": base_meta,
    }
    output = {key: np.asarray(value) for key, value in base.items() if key != "metadata_json"}
    output.update(
        event_features=np.asarray(augmented_events, dtype=np.float32),
        event_type=np.asarray(event_type, dtype=np.uint8),
        event_mask=np.asarray(event_mask, dtype=bool),
        counterfactual_event_gain_db=np.asarray(counterfactual, dtype=np.float32),
        fidelity_dbm=fidelity.astype(np.float32),
        rf_mechanism_db=scenario_components_array[:, nominal_index].astype(np.float32),
        scenario_teacher_dbm=scenario_teachers_array,
        scenario_mechanism_db=scenario_components_array,
        fidelity_available=np.ones(3, dtype=bool),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return output


def load_cmf_peno_dataset(path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        data = {name: source[name] for name in source.files}
    try:
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("invalid CMF-PENO metadata") from error
    if metadata.get("schema") not in (CMF_PENO_DATASET_SCHEMA, CMF_PENO_RF_DATASET_SCHEMA):
        raise ValueError("requires cmf_peno_dataset_v1 or cmf_peno_rf_dataset_v2")
    if metadata.get("schema") == CMF_PENO_RF_DATASET_SCHEMA and metadata.get(
        "model_input_uses_teacher_components", True
    ):
        raise ValueError("RF dataset declares teacher components as model inputs")
    cases, shape = len(data["station_xyz_m"]), data["coarse_dbm"].shape[1:]
    max_events = int(metadata["max_events"])
    feature_count = int(data["event_features"].shape[2])
    if int(metadata.get("event_feature_count", feature_count)) != feature_count:
        raise ValueError("event feature count does not match metadata")
    if data["event_features"].shape != (cases, max_events, feature_count, *shape):
        raise ValueError("invalid event_features shape")
    if data["event_type"].shape != (cases, max_events, *shape):
        raise ValueError("invalid event_type shape")
    if data["event_mask"].shape != data["event_type"].shape:
        raise ValueError("event mask/type mismatch")
    if data["fidelity_dbm"].shape[0] != cases or data["fidelity_dbm"].shape[2:] != shape:
        raise ValueError("invalid fidelity_dbm shape")
    if not np.isfinite(data["event_features"]).all() or not np.isfinite(data["fidelity_dbm"]).all():
        raise ValueError("CMF-PENO data must be finite")
    return data


def load_cmf_peno_rf_dataset(path) -> dict[str, np.ndarray]:
    """Load and validate the mechanism-aware dataset."""

    data = load_cmf_peno_dataset(path)
    metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if metadata.get("schema") != CMF_PENO_RF_DATASET_SCHEMA:
        raise ValueError("requires cmf_peno_rf_dataset_v2; rebuild legacy RF v1 data")
    required = {"rf_mechanism_db", "scenario_teacher_dbm", "scenario_mechanism_db"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"RF dataset is missing fields: {missing}")
    cases = len(data["station_xyz_m"])
    shape = tuple(np.asarray(data["coarse_dbm"]).shape[1:])
    if np.asarray(data["rf_mechanism_db"]).shape != (cases, 3, *shape):
        raise ValueError("invalid rf_mechanism_db shape")
    if np.asarray(data["scenario_teacher_dbm"]).shape[:1] != (cases,) or np.asarray(
        data["scenario_teacher_dbm"]
    ).shape[2:] != shape:
        raise ValueError("invalid scenario_teacher_dbm shape")
    if not np.isfinite(data["rf_mechanism_db"]).all() or not np.isfinite(data["scenario_teacher_dbm"]).all():
        raise ValueError("RF dataset contains non-finite values")
    return data


def validate_aligned_reference_dataset(
    data: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
) -> None:
    """Reject cross-teacher evaluation unless every case is spatially aligned."""

    if np.asarray(data["coarse_dbm"]).shape != np.asarray(reference["coarse_dbm"]).shape:
        raise ValueError("reference volume shape does not match input dataset")
    if not np.array_equal(np.asarray(data["split"]), np.asarray(reference["split"])):
        raise ValueError("reference split does not match input dataset")
    if not np.allclose(
        np.asarray(data["station_xyz_m"]), np.asarray(reference["station_xyz_m"]),
        rtol=0.0, atol=1e-4,
    ):
        raise ValueError("reference station order or coordinates do not match")
    for name in ("x_m", "y_m", "z_m"):
        if np.asarray(data[name]).shape != np.asarray(reference[name]).shape or not np.allclose(
            np.asarray(data[name]), np.asarray(reference[name]), rtol=0.0, atol=1e-5,
        ):
            raise ValueError(f"reference {name} axes do not match")
    if np.asarray(data["air_mask"]).shape != np.asarray(reference["air_mask"]).shape or not np.array_equal(
        np.asarray(data["air_mask"]), np.asarray(reference["air_mask"]),
    ):
        raise ValueError("reference air mask does not match input dataset")
