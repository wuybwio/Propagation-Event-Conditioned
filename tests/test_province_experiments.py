import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from jiangxi_radio.field_evaluation import field_metrics
from jiangxi_radio.province_experiments import (
    build_grouped_roles,
    evaluate_candidate_planning,
    pack_grouped_scenes,
    select_spatial_scene_centres,
    variant_spec,
)


def _scene_payload(offset: float) -> dict[str, np.ndarray]:
    cases = 3
    shape = (4, 4, 4)
    return {
        "x_m": np.linspace(0, 30, shape[2], dtype=np.float32),
        "y_m": np.linspace(0, 30, shape[1], dtype=np.float32),
        "z_m": np.linspace(0, 30, shape[0], dtype=np.float32),
        "terrain_m": np.full(shape[1:], offset, dtype=np.float32),
        "surface_m": np.full(shape[1:], offset, dtype=np.float32),
        "air_mask": np.ones(shape, dtype=bool),
        "station_xyz_m": np.column_stack((np.arange(cases), np.arange(cases), np.full(cases, 20)))
        .astype(np.float32),
        "split": np.asarray([0, 1, 2], dtype=np.int64),
        "coarse_dbm": np.full((cases, *shape), -90 - offset, dtype=np.float32),
        "teacher_dbm": np.full((cases, *shape), -95 - offset, dtype=np.float32),
        "metadata_json": np.asarray(json.dumps({
            "schema": "station_volume_v1",
            "measurement_validation": False,
        })),
    }


class ProvinceExperimentTests(unittest.TestCase):
    def test_numpy_field_metrics_report_asymmetric_error_rates(self):
        metrics = field_metrics(
            np.asarray([-80.0, -80.0, -120.0, -120.0]),
            np.asarray([-80.0, -120.0, -80.0, -120.0]),
            np.ones(4, dtype=bool),
        )
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(metrics["false_coverage_rate"], 0.5)
        self.assertAlmostEqual(metrics["coverage_recall"], 0.5)

    def test_selects_unique_scene_centres_from_every_spatial_group(self):
        stations = pd.DataFrame({
            "station_id": [f"s{i}" for i in range(20)],
            "longitude": np.repeat([114.0, 115.0, 116.0, 117.0, 118.0], 4),
            "latitude": np.tile([25.0, 26.0, 28.0, 29.0], 5),
            "grade": 4,
        })
        scenes = select_spatial_scene_centres(
            stations, group_count=5, scenes_per_group=2, seed=2026
        )
        self.assertEqual(len(scenes), 10)
        self.assertEqual(scenes["geo_group"].nunique(), 5)
        self.assertTrue((scenes.groupby("geo_group").size() == 2).all())
        self.assertEqual(scenes["source_station_id"].nunique(), 10)

    def test_grouped_roles_keep_geographical_groups_disjoint(self):
        manifest = pd.DataFrame({
            "scene_id": [f"g{group}_s{scene}" for group in range(5) for scene in range(2)],
            "geo_group": [f"g{group}" for group in range(5) for _ in range(2)],
        })
        roles = build_grouped_roles(manifest)
        self.assertEqual(set(roles["role"]), {"train", "validation", "calibration", "test"})
        self.assertEqual(roles["fold"].nunique(), 5)
        for fold, frame in roles.groupby("fold"):
            per_group = frame.groupby("geo_group")["role"].nunique()
            self.assertTrue((per_group == 1).all())
            self.assertEqual(set(frame.loc[frame.role == "test", "geo_group"]), {fold})

    def test_packer_assigns_scene_level_roles_including_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = {}
            roles = ("train", "validation", "calibration", "test")
            records = []
            for index, role in enumerate(roles):
                scene = f"scene_{index}"
                path = root / f"{scene}.npz"
                np.savez_compressed(path, **_scene_payload(float(index)))
                entries[scene] = path
                records.append({"scene_id": scene, "geo_group": f"g{index}", "role": role})
            packed = pack_grouped_scenes(entries, pd.DataFrame(records), fold="g3")
            self.assertEqual(set(packed["split"].tolist()), {0, 1, 2, 3})
            for scene_index, split_id in enumerate((0, 1, 3, 2)):
                selected = packed["scene_id"] == f"scene_{scene_index}"
                self.assertTrue(np.all(packed["split"][selected] == split_id))
            metadata = json.loads(str(packed["metadata_json"].item()))
            self.assertEqual(metadata["split_policy"], "geographical_group_roles")
            self.assertEqual(metadata["calibration_split_id"], 3)

    def test_variant_registry_defines_full_and_structural_ablations(self):
        self.assertTrue(variant_spec("full_v3").use_events)
        self.assertFalse(variant_spec("no_events").use_events)
        self.assertFalse(variant_spec("no_local").use_local)
        self.assertFalse(variant_spec("no_spectral").use_spectral)
        self.assertFalse(variant_spec("no_event_consistency").use_event_consistency)
        self.assertFalse(variant_spec("deterministic").use_uncertainty)
        self.assertFalse(variant_spec("no_multilevel").use_multilevel_supervision)
        self.assertFalse(variant_spec("cnn_residual").use_events)
        self.assertFalse(variant_spec("cnn_residual").use_spectral)
        self.assertFalse(variant_spec("fno_residual").use_events)
        self.assertFalse(variant_spec("fno_residual").use_local)
        with self.assertRaises(ValueError):
            variant_spec("unknown")

    def test_planning_selects_on_method_field_but_evaluates_on_reference(self):
        reference = np.asarray([
            [[[0.1, 0.1]]],
            [[[0.9, 0.9]]],
        ])
        misleading = np.asarray([
            [[[0.99, 0.99]]],
            [[[0.01, 0.01]]],
        ])
        records, selections = evaluate_candidate_planning(
            {"MisleadingModel": misleading},
            reference,
            np.asarray([[0.0, 0.0, 20.0], [100.0, 0.0, 20.0]]),
            np.ones((1, 1, 2), dtype=bool),
            budgets=[1],
            random_repeats=2,
            seed=7,
        )
        frame = pd.DataFrame(records)
        model_score = frame.loc[frame.method == "MisleadingModel", "mean_detection_probability"].iloc[0]
        oracle_score = frame.loc[frame.method == "ExactReference", "mean_detection_probability"].iloc[0]
        self.assertLess(model_score, oracle_score)
        self.assertEqual(selections["MisleadingModel:1:0"], [0])
        self.assertEqual(selections["ReferenceGreedy:1:0"], [1])
        self.assertEqual(selections["ExactReference:1:0"], [1])


if __name__ == "__main__":
    unittest.main()
