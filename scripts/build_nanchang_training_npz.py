"""Build monitoring simulation tensors, or explicitly selected legacy LTE proxy diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jiangxi_radio.config import load_scenario
from jiangxi_radio.experiment_contract import MONITORING_SIMULATION, LEGACY_MIXED
from jiangxi_radio.environment_features import (
    EnvironmentFeatureConfig,
    OvertureBuildingConfig,
    build_overture_building_features,
    building_obstruction_loss_db,
    build_proxy_environment_features,
    station_path_knife_edge_loss_db,
)
from jiangxi_radio.physics_baseline import (
    compute_route_station_probabilities,
    compute_station_received_power_dbm,
)
from jiangxi_radio.physics_features import received_power_features
from jiangxi_radio.terrain import sample_srtm_elevation
from jiangxi_radio.radio_map_labels import _validate_heights
from jiangxi_radio.training_dataset import (
    _grid_axes,
    build_observation_targets,
    build_observation_signal_targets,
    gaussian_field,
    poisson_binomial_at_least,
)


def _aggregate_station_probabilities(values: np.ndarray) -> np.ndarray:
    tails = np.stack([poisson_binomial_at_least(row) for row in values], axis=0)
    return tails


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/nanchang_lte_clean/nanchang_lte_training_labels.csv")
    parser.add_argument("--stations", default="data/stations.csv")
    parser.add_argument("--routes", default="data/radio_map/route_points.csv")
    parser.add_argument("--config", default="configs/scenario.toml")
    parser.add_argument("--output", default="data/nanchang_lte_clean/monitoring_simulation_dataset.npz")
    parser.add_argument("--cell-size-deg", type=float, default=0.001)
    parser.add_argument("--experiment-domain", choices=(MONITORING_SIMULATION, LEGACY_MIXED),
                        default=MONITORING_SIMULATION,
                        help="Simulation labels use monitoring physics; legacy LTE mixing is diagnostic only.")
    parser.add_argument("--supervision-height-m", type=float, default=None,
                        help="Legacy proxy placement only; this is not a measured LTE altitude.")
    parser.add_argument("--rsrp-threshold-dbm", type=float, default=-95.0)
    parser.add_argument("--signal-min-dbm", type=float, default=-135.0)
    parser.add_argument("--signal-max-dbm", type=float, default=-40.0)
    parser.add_argument(
        "--signal-target-mode",
        choices=("weighted_mean", "strongest"),
        default="strongest",
        help="Legacy LTE target statistic only; does not establish monitoring-link equivalence.",
    )
    parser.add_argument(
        "--buildings-dir",
        default="",
        help="Optional directory of Overture tile_*.geojson files. Enables building-aware channels.",
    )
    parser.add_argument("--building-max-loss-db", type=float, default=18.0)
    parser.add_argument("--path-samples-per-km", type=float, default=2.0)
    parser.add_argument("--path-max-distance-km", type=float, default=15.0)
    parser.add_argument("--effective-earth-radius-factor", type=float, default=4.0 / 3.0)
    parser.add_argument("--dem-dir", default="", help="Directory containing SRTM .hgt or .hgt.gz tiles.")
    parser.add_argument(
        "--path-building-loss",
        action="store_true",
        help="Use station-to-grid path-specific building obstruction instead of one point loss.",
    )
    parser.add_argument(
        "--include-physics-features",
        action="store_true",
        help="Append height-specific multi-station propagation features to the environment tensor.",
    )
    args = parser.parse_args()
    if args.experiment_domain == MONITORING_SIMULATION and args.supervision_height_m is not None:
        raise ValueError("Simulation has no measured supervision height; remove --supervision-height-m")

    labels = pd.read_csv(args.labels)
    stations = pd.read_csv(args.stations, encoding="utf-8-sig")
    config = load_scenario(args.config)
    heights = _validate_heights(config.study.heights_m)
    if args.experiment_domain == LEGACY_MIXED and args.supervision_height_m not in heights:
        raise ValueError("supervision height must be included in scenario heights")

    lon_axis, lat_axis = _grid_axes(labels, args.cell_size_deg)
    lon_grid, lat_grid = np.meshgrid(lon_axis, lat_axis)
    points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    height_count, height_width = len(lat_axis), len(lon_axis)

    environment_flat, environment_meta = build_proxy_environment_features(
        points,
        EnvironmentFeatureConfig(include_height=False),
    )
    terrain_flat = np.zeros(len(points), dtype=np.float32)
    if args.dem_dir:
        terrain_flat = sample_srtm_elevation(points, args.dem_dir)
        if not np.any(terrain_flat):
            raise ValueError(
                "--dem-dir was provided but no valid SRTM elevations were sampled; "
                "check tile names and the DEM directory"
            )
        environment_flat[:, 0] = terrain_flat
        environment_meta = {
            **environment_meta,
            "terrain": {"source": "SRTM_1arcsec", "dem_dir": args.dem_dir},
        }
    building_loss = np.zeros(len(points), dtype=np.float32)
    building_flat = None
    if args.buildings_dir:
        building_flat, building_meta = build_overture_building_features(
            points,
            OvertureBuildingConfig(args.buildings_dir),
        )
        environment_flat = np.column_stack((environment_flat, building_flat))
        building_loss = building_obstruction_loss_db(
            building_flat, max_loss_db=args.building_max_loss_db
        )
        environment_meta = {
            **environment_meta,
            "building": building_meta,
            "building_loss_channel": "conservative normalized obstruction prior",
        }
    environment = np.repeat(
        environment_flat.T.reshape(1, environment_flat.shape[1], height_count, height_width),
        len(heights), axis=0,
    ).astype(np.float32)

    station_coordinates = stations[["longitude", "latitude"]].to_numpy(dtype=float)
    station_ground = sample_srtm_elevation(station_coordinates, args.dem_dir) if args.dem_dir else np.zeros(len(stations), dtype=np.float32)
    station_heights = stations["grade"].map(
        {grade: profile.antenna_height_m for grade, profile in config.station_grades.items()}
    ).to_numpy(dtype=float)
    station_field_flat = np.zeros((len(points), 3), dtype=np.float32)
    for grade_index, grade in enumerate((2, 3, 4)):
        centers = station_coordinates[stations["grade"].to_numpy(dtype=int) == grade]
        station_field_flat[:, grade_index] = gaussian_field(points, centers, scale_km=20.0)
    station_field = np.repeat(
        station_field_flat.T.reshape(1, 3, height_count, height_width),
        len(heights), axis=0,
    )

    if Path(args.routes).exists():
        route_points = pd.read_csv(args.routes)[["longitude", "latitude"]].to_numpy(dtype=float)
        route_flat = gaussian_field(points, route_points, scale_km=5.0)
        route_evidence = "route_points_gaussian_5km"
    else:
        route_flat = np.zeros(len(points), dtype=np.float32)
        route_evidence = "no_route_points"
    route_weight = np.repeat(
        route_flat.reshape(1, 1, height_count, height_width),
        len(heights), axis=0,
    )

    physical_prior = np.zeros((len(heights), 3, height_count, height_width), dtype=np.float32)
    physical_signal = np.zeros((len(heights), 1, height_count, height_width), dtype=np.float32)
    propagation_features = np.zeros((len(heights), 5, height_count, height_width), dtype=np.float32)
    for height_index, height in enumerate(heights):
        if args.path_building_loss:
            if building_flat is None:
                raise ValueError("--path-building-loss requires --buildings-dir")
            height_loss = station_path_knife_edge_loss_db(
                points,
                station_coordinates,
                building_flat,
                (lon_axis, lat_axis),
                station_height_m=station_heights,
                flight_height_m=float(height),
                max_loss_db=args.building_max_loss_db,
                samples_per_km=args.path_samples_per_km,
                max_path_distance_km=args.path_max_distance_km,
                frequency_mhz=float(config.radio.frequency_mhz),
                effective_earth_radius_factor=args.effective_earth_radius_factor,
                terrain_elevation_m=terrain_flat,
                station_ground_elevation_m=station_ground,
                target_ground_elevation_m=terrain_flat,
            )
        else:
            height_loss = building_loss * np.exp(-float(height) / 300.0)
        received_power = compute_station_received_power_dbm(
            points,
            float(height),
            stations,
            config.radio,
            config.station_grades,
            additional_loss_db=height_loss,
            environment=config.environment,
            station_ground_elevation_m=station_ground,
            target_ground_elevation_m=terrain_flat,
        )
        strongest_received = np.max(received_power, axis=1)
        if args.include_physics_features:
            propagation_features[height_index] = received_power_features(
                received_power,
                signal_min_dbm=args.signal_min_dbm,
                signal_max_dbm=args.signal_max_dbm,
                detection_threshold_dbm=args.rsrp_threshold_dbm,
            ).T.reshape(5, height_count, height_width)
        physical_signal[height_index, 0] = np.clip(
            (strongest_received - args.signal_min_dbm)
            / (args.signal_max_dbm - args.signal_min_dbm),
            0.0,
            1.0,
        ).reshape(height_count, height_width)
        per_station = compute_route_station_probabilities(
            points,
            float(height),
            stations,
            config.radio,
            config.station_grades,
            additional_loss_db=height_loss,
            environment=config.environment,
            station_ground_elevation_m=station_ground,
            target_ground_elevation_m=terrain_flat,
        )
        tails = _aggregate_station_probabilities(per_station)
        physical_prior[height_index] = tails.T.reshape(3, height_count, height_width)

    if args.include_physics_features:
        environment = np.concatenate((environment, propagation_features), axis=1)
        environment_meta = {
            **environment_meta,
            "propagation_features": "strongest, second strongest, top-3 mean, strong-station fraction, top-1/top-2 gap",
        }

    if args.experiment_domain == MONITORING_SIMULATION:
        # LTE coordinates define the study extent only; no LTE RSRP supervises this task.
        target = physical_prior.copy()
        signal_target = physical_signal.copy()
        mask = np.ones_like(target)
        signal_mask = np.ones_like(signal_target)
        evidence = "monitoring_physics_simulation"
        target_note = "Simulator emulation only; the generating physics is the exact target, not an independent baseline."
    else:
        target_flat, mask_flat = build_observation_targets(
            labels, lon_axis, lat_axis, rsrp_threshold_dbm=args.rsrp_threshold_dbm)
        signal_target_flat, signal_mask_flat = build_observation_signal_targets(
            labels, lon_axis, lat_axis, signal_min_dbm=args.signal_min_dbm,
            signal_max_dbm=args.signal_max_dbm, mode=args.signal_target_mode)
        target = np.repeat(target_flat[None, ...], len(heights), axis=0).astype(np.float32)
        mask = np.zeros_like(target)
        supervision_index = int(np.where(np.isclose(heights, args.supervision_height_m))[0][0])
        mask[supervision_index] = mask_flat
        signal_target = np.repeat(signal_target_flat[None, None, ...], len(heights), axis=0).astype(np.float32)
        signal_mask = np.zeros_like(signal_target)
        signal_mask[supervision_index, 0] = signal_mask_flat
        evidence = "LTE_derived_proxy_not_monitoring_truth"
        target_note = "Legacy mixed-domain diagnostic; assigned height is not an observation."

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        experiment_domain=np.asarray(args.experiment_domain),
        label_evidence=np.asarray(evidence),
        label_schema_version=np.asarray(2),
        heights_m=heights.astype(np.float32),
        longitude=lon_axis.astype(np.float32),
        latitude=lat_axis.astype(np.float32),
        terrain_elevation_m=terrain_flat.reshape(height_count, height_width),
        station_ground_elevation_m=station_ground.astype(np.float32),
        environment=environment,
        propagation_features=propagation_features,
        physical_prior=physical_prior[:, 0:1],
        physical_signal=physical_signal,
        physical_prior_redundancy=physical_prior,
        station_field=station_field,
        route_weight=route_weight,
        target_probability=target,
        target_mask=mask,
        target_signal=signal_target,
        target_signal_mask=signal_mask,
        signal_min_dbm=np.asarray(args.signal_min_dbm, dtype=np.float32),
        signal_max_dbm=np.asarray(args.signal_max_dbm, dtype=np.float32),
    )
    metadata = {
        "experiment_domain": args.experiment_domain,
        "environment": environment_meta,
        "route_evidence": route_evidence,
        "label_evidence": evidence,
        "supervision_height_m": args.supervision_height_m,
        "target_note": target_note,
        "signal_target": {"min_dbm": args.signal_min_dbm, "max_dbm": args.signal_max_dbm, "supervision_height_m": args.supervision_height_m, "mode": args.signal_target_mode},
        "physical_signal": "strongest simulated monitoring-link received power, not LTE RSRP",
        "measurement_validation": False,
        "building_aware": bool(args.buildings_dir),
        "terrain_aware": bool(args.dem_dir),
        "terrain_parameters": {"dem_dir": args.dem_dir, "source": "SRTM_1arcsec" if args.dem_dir else "proxy"},
        "path_building_loss": bool(args.path_building_loss),
        "path_building_loss_parameters": {
            "samples_per_km": args.path_samples_per_km,
            "max_distance_km": args.path_max_distance_km,
            "effective_earth_radius_factor": args.effective_earth_radius_factor,
            "frequency_mhz": float(config.radio.frequency_mhz),
            "model": "sampled_3d_knife_edge_fresnel",
        },
        "shape": {"height": len(heights), "channels": int(environment.shape[1]), "height_pixels": height_count, "width_pixels": height_width},
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print({"output": str(output), "environment_shape": environment.shape, "prior_shape": (len(heights), 1, height_count, height_width), "target_shape": target.shape, "supervised_pixels": int(mask.sum())})


if __name__ == "__main__":
    main()
