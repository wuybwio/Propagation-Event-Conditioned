"""Label-free propagation-event extraction from terrain and surface profiles.

The current geometry supports direct, terrain-diffraction, and building-
diffraction events. Reflection, scattering, and transmission identifiers are
reserved for externally generated high-fidelity path data; this module never
fabricates those mechanisms from building height alone.
"""

from __future__ import annotations

import math

import numpy as np

from .environment_features import knife_edge_diffraction_loss_db

DIRECT = 0
TERRAIN_DIFFRACTION = 1
BUILDING_DIFFRACTION = 2
REFLECTION = 3
SCATTERING = 4
TRANSMISSION = 5
EVENT_TYPE_NAMES = (
    "direct",
    "terrain_diffraction",
    "building_diffraction",
    "reflection",
    "scattering",
    "transmission",
)
EVENT_FEATURE_NAMES = (
    "path_fraction",
    "clearance_div_50m",
    "obstruction_height_div_50m",
    "fresnel_v_div_10",
    "d1_fraction",
    "d2_fraction",
    "building_excess_div_50m",
    "direction_x",
    "direction_y",
    "direction_z",
    "path_length_div_1000m",
    "knife_loss_div_60db",
)
EVENT_FEATURE_COUNT = len(EVENT_FEATURE_NAMES)
EVENT_SCHEMA = "propagation_events_v1"
RF_EVENT_SCHEMA = "propagation_events_rf_v1"


def _validate_geometry(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    terrain: np.ndarray,
    surface: np.ndarray,
    station: np.ndarray,
) -> tuple[np.ndarray, ...]:
    x, y, z = (np.asarray(value, dtype=float) for value in (x, y, z))
    terrain, surface, station = (
        np.asarray(value, dtype=float) for value in (terrain, surface, station)
    )
    if any(
        axis.ndim != 1
        or len(axis) < 2
        or not np.isfinite(axis).all()
        or not np.all(np.diff(axis) > 0.0)
        for axis in (x, y, z)
    ):
        raise ValueError("finite increasing x, y and z axes are required")
    if terrain.shape != (len(y), len(x)) or surface.shape != terrain.shape:
        raise ValueError("terrain and surface must align with y and x")
    if not np.isfinite(terrain).all() or not np.isfinite(surface).all():
        raise ValueError("terrain and surface must be finite")
    if np.any(surface < terrain - 1e-5):
        raise ValueError("surface cannot lie below terrain")
    if (
        station.shape != (3,)
        or not np.isfinite(station).all()
        or not x[0] <= station[0] <= x[-1]
        or not y[0] <= station[1] <= y[-1]
    ):
        raise ValueError("station must be finite and inside horizontal bounds")
    return x, y, z, terrain, surface, station


def _bilinear(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    sample_x: np.ndarray,
    sample_y: np.ndarray,
) -> np.ndarray:
    ix = np.clip(np.searchsorted(x, sample_x, side="right") - 1, 0, len(x) - 2)
    iy = np.clip(np.searchsorted(y, sample_y, side="right") - 1, 0, len(y) - 2)
    ax = (sample_x - x[ix]) / (x[ix + 1] - x[ix])
    ay = (sample_y - y[iy]) / (y[iy + 1] - y[iy])
    return (
        (1.0 - ax) * (1.0 - ay) * field[iy, ix]
        + ax * (1.0 - ay) * field[iy, ix + 1]
        + (1.0 - ax) * ay * field[iy + 1, ix]
        + ax * ay * field[iy + 1, ix + 1]
    )


def build_propagation_events(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    terrain: np.ndarray,
    surface: np.ndarray,
    station: np.ndarray,
    *,
    samples: int = 32,
    max_events: int = 4,
    frequency_mhz: float = 2500.0,
    effective_earth_radius_factor: float = 4.0 / 3.0,
    chunk_size: int = 2048,
) -> dict[str, np.ndarray]:
    """Build a fixed-size event set for every ``[Z,Y,X]`` query voxel.

    Slot zero is always the direct event. Remaining slots contain the largest
    positive line-of-sight obstructions ranked by knife-edge loss. The
    counterfactual target is the non-negative loss removed with the strongest
    obstruction event.
    """
    x, y, z, terrain, surface, station = _validate_geometry(
        x, y, z, terrain, surface, station
    )
    if not isinstance(samples, (int, np.integer)) or not 2 <= samples <= 1024:
        raise ValueError("samples must be an integer in [2, 1024]")
    if not isinstance(max_events, (int, np.integer)) or not 1 <= max_events <= 32:
        raise ValueError("max_events must be an integer in [1, 32]")
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if (
        not math.isfinite(frequency_mhz)
        or frequency_mhz <= 0.0
        or not math.isfinite(effective_earth_radius_factor)
        or effective_earth_radius_factor <= 0.0
    ):
        raise ValueError("frequency and effective Earth-radius factor must be positive")

    shape = (len(z), len(y), len(x))
    voxel_count = int(np.prod(shape))
    features = np.zeros(
        (max_events, EVENT_FEATURE_COUNT, voxel_count), dtype=np.float32
    )
    event_type = np.zeros((max_events, voxel_count), dtype=np.uint8)
    event_mask = np.zeros((max_events, voxel_count), dtype=bool)
    counterfactual_gain = np.zeros(voxel_count, dtype=np.float32)

    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    queries = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    vector = queries - station
    path_length = np.linalg.norm(vector, axis=1)
    safe_length = np.maximum(path_length, 1e-6)
    direction = vector / safe_length[:, None]
    direction[path_length < 1e-6] = 0.0
    event_mask[0] = True
    event_type[0] = DIRECT
    features[0, 7:10] = direction.T.astype(np.float32)
    features[0, 10] = (path_length / 1000.0).astype(np.float32)

    if max_events == 1:
        return {
            "event_features": features.reshape(max_events, EVENT_FEATURE_COUNT, *shape),
            "event_type": event_type.reshape(max_events, *shape),
            "event_mask": event_mask.reshape(max_events, *shape),
            "counterfactual_gain_db": counterfactual_gain.reshape(shape),
        }

    wavelength = 299_792_458.0 / (frequency_mhz * 1e6)
    fractions = np.arange(1, samples + 1, dtype=float) / (samples + 1.0)
    earth_radius = 6_371_000.0 * effective_earth_radius_factor
    for start in range(0, voxel_count, chunk_size):
        stop = min(start + chunk_size, voxel_count)
        query = queries[start:stop]
        count = len(query)
        delta = query - station
        horizontal = np.linalg.norm(delta[:, :2], axis=1)
        points = station + delta[:, None, :] * fractions[None, :, None]
        sample_x, sample_y = points[..., 0], points[..., 1]
        terrain_top = _bilinear(terrain, x, y, sample_x, sample_y)
        surface_top = _bilinear(surface, x, y, sample_x, sample_y)
        d1 = np.maximum(horizontal[:, None] * fractions[None, :], 0.01)
        d2 = np.maximum(horizontal[:, None] * (1.0 - fractions[None, :]), 0.01)
        curvature = d1 * d2 / (2.0 * earth_radius)
        line_z = points[..., 2] - curvature
        obstruction = surface_top - line_z
        clearance = -obstruction
        fresnel_v = obstruction * np.sqrt(
            2.0 * (d1 + d2) / (wavelength * d1 * d2)
        )
        knife_loss = np.asarray(knife_edge_diffraction_loss_db(fresnel_v), dtype=float)
        candidate = (obstruction > 0.0) & (horizontal[:, None] > 1e-6)
        knife_loss = np.where(candidate, knife_loss, -np.inf)
        building_excess = np.maximum(surface_top - terrain_top, 0.0)

        for local_index in range(count):
            valid = np.flatnonzero(candidate[local_index])
            if not len(valid):
                continue
            ranked = valid[np.argsort(knife_loss[local_index, valid])[::-1]]
            selected = ranked[: max_events - 1]
            counterfactual_gain[start + local_index] = float(
                np.clip(knife_loss[local_index, selected[0]], 0.0, 60.0)
            )
            for slot_offset, sample_index in enumerate(selected, start=1):
                output_index = start + local_index
                is_building = building_excess[local_index, sample_index] > 1.0
                event_type[slot_offset, output_index] = (
                    BUILDING_DIFFRACTION if is_building else TERRAIN_DIFFRACTION
                )
                event_mask[slot_offset, output_index] = True
                values = np.asarray(
                    [
                        fractions[sample_index],
                        clearance[local_index, sample_index] / 50.0,
                        obstruction[local_index, sample_index] / 50.0,
                        np.clip(fresnel_v[local_index, sample_index] / 10.0, -10.0, 10.0),
                        fractions[sample_index],
                        1.0 - fractions[sample_index],
                        building_excess[local_index, sample_index] / 50.0,
                        direction[start + local_index, 0],
                        direction[start + local_index, 1],
                        direction[start + local_index, 2],
                        path_length[start + local_index] / 1000.0,
                        np.clip(knife_loss[local_index, sample_index], 0.0, 60.0) / 60.0,
                    ],
                    dtype=np.float32,
                )
                features[slot_offset, :, output_index] = values

    return {
        "event_features": features.reshape(max_events, EVENT_FEATURE_COUNT, *shape),
        "event_type": event_type.reshape(max_events, *shape),
        "event_mask": event_mask.reshape(max_events, *shape),
        "counterfactual_gain_db": counterfactual_gain.reshape(shape),
    }
