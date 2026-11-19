import json
import unittest

import numpy as np


class CMFPENODataTests(unittest.TestCase):
    def base_data(self):
        axis = np.linspace(0.0, 30.0, 4).astype("float32")
        z = (axis + 20.0).astype("float32")
        terrain = np.zeros((4, 4), dtype="float32")
        surface = terrain.copy()
        surface[:, 2] = 35.0
        stations = np.array(
            [[0.0, 10.0, 20.0], [15.0, 10.0, 20.0], [30.0, 10.0, 20.0]],
            dtype="float32",
        )
        coarse = np.full((3, 4, 4, 4), -60.0, dtype="float32")
        teacher = coarse - 10.0
        metadata = {
            "schema": "station_volume_v1",
            "measurement_validation": False,
            "evidence": "simulation_only",
        }
        return {
            "x_m": axis,
            "y_m": axis,
            "z_m": z,
            "terrain_m": terrain,
            "surface_m": surface,
            "air_mask": np.ones((4, 4, 4), dtype=bool),
            "station_xyz_m": stations,
            "split": np.array([0, 1, 2], dtype="int64"),
            "coarse_dbm": coarse,
            "teacher_dbm": teacher,
            "metadata_json": np.asarray(json.dumps(metadata)),
        }

    def test_two_fidelity_dataset_contains_events_and_provenance(self):
        from jiangxi_radio.cmf_peno_data import build_cmf_peno_dataset

        result = build_cmf_peno_dataset(
            self.base_data(), samples=5, max_events=3, frequency_mhz=2500.0
        )
        self.assertEqual(result["event_features"].shape, (3, 3, 12, 4, 4, 4))
        self.assertEqual(result["event_type"].shape, (3, 3, 4, 4, 4))
        self.assertEqual(result["fidelity_dbm"].shape, (3, 2, 4, 4, 4))
        np.testing.assert_array_equal(result["fidelity_dbm"][:, 0], result["coarse_dbm"])
        np.testing.assert_array_equal(result["fidelity_dbm"][:, 1], result["teacher_dbm"])
        metadata = json.loads(str(result["metadata_json"]))
        self.assertEqual(metadata["schema"], "cmf_peno_dataset_v1")
        self.assertEqual(metadata["fidelity_names"], ["fspl", "geometric_teacher"])
        self.assertFalse(metadata["high_fidelity_available"])
        self.assertEqual(metadata["event_schema"], "propagation_events_v1")

    def test_optional_high_fidelity_becomes_third_level(self):
        from jiangxi_radio.cmf_peno_data import build_cmf_peno_dataset

        base = self.base_data()
        high = {
            "x_m": base["x_m"].copy(),
            "y_m": base["y_m"].copy(),
            "z_m": base["z_m"].copy(),
            "station_xyz_m": base["station_xyz_m"].copy(),
            "high_fidelity_dbm": base["teacher_dbm"] - 3.0,
        }
        result = build_cmf_peno_dataset(base, high_fidelity=high, samples=5, max_events=3)
        self.assertEqual(result["fidelity_dbm"].shape[1], 3)
        np.testing.assert_array_equal(result["fidelity_dbm"][:, 2], high["high_fidelity_dbm"])
        metadata = json.loads(str(result["metadata_json"]))
        self.assertTrue(metadata["high_fidelity_available"])
        self.assertEqual(metadata["fidelity_names"][-1], "high_fidelity_teacher")

    def test_misaligned_high_fidelity_is_rejected(self):
        from jiangxi_radio.cmf_peno_data import build_cmf_peno_dataset

        base = self.base_data()
        high = {
            "x_m": base["x_m"] + 1.0,
            "y_m": base["y_m"],
            "z_m": base["z_m"],
            "station_xyz_m": base["station_xyz_m"],
            "high_fidelity_dbm": base["teacher_dbm"],
        }
        with self.assertRaisesRegex(ValueError, "axes"):
            build_cmf_peno_dataset(base, high_fidelity=high, samples=5, max_events=3)

    def test_event_construction_does_not_depend_on_teacher_values(self):
        from jiangxi_radio.cmf_peno_data import build_cmf_peno_dataset

        first = self.base_data()
        second = self.base_data()
        second["teacher_dbm"][:] = -999.0
        a = build_cmf_peno_dataset(first, samples=5, max_events=3)
        b = build_cmf_peno_dataset(second, samples=5, max_events=3)
        np.testing.assert_array_equal(a["event_features"], b["event_features"])
        np.testing.assert_array_equal(a["event_type"], b["event_type"])
        np.testing.assert_array_equal(a["event_mask"], b["event_mask"])


if __name__ == "__main__":
    unittest.main()
