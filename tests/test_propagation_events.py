import unittest

import numpy as np


class PropagationEventTests(unittest.TestCase):
    def geometry(self):
        axis = np.linspace(0.0, 100.0, 11)
        z = np.array([20.0, 80.0])
        terrain = np.zeros((11, 11), dtype=float)
        surface = terrain.copy()
        station = np.array([0.0, 50.0, 20.0])
        return axis, axis, z, terrain, surface, station

    def test_clear_scene_contains_only_direct_events(self):
        from jiangxi_radio.propagation_events import (
            DIRECT,
            build_propagation_events,
        )

        result = build_propagation_events(
            *self.geometry(), samples=9, max_events=4, frequency_mhz=2500.0
        )
        self.assertEqual(result["event_features"].shape, (4, 12, 2, 11, 11))
        self.assertEqual(result["event_type"].shape, (4, 2, 11, 11))
        self.assertTrue(result["event_mask"][0].all())
        self.assertTrue((result["event_type"][0] == DIRECT).all())
        self.assertFalse(result["event_mask"][1:].any())
        np.testing.assert_array_equal(result["counterfactual_gain_db"], 0.0)

    def test_wall_creates_building_diffraction_and_positive_counterfactual_gain(self):
        from jiangxi_radio.propagation_events import (
            BUILDING_DIFFRACTION,
            build_propagation_events,
        )

        x, y, z, terrain, surface, station = self.geometry()
        surface[:, 5] = 50.0
        result = build_propagation_events(
            x, y, z, terrain, surface, station,
            samples=9, max_events=4, frequency_mhz=2500.0,
        )
        types = result["event_type"][:, 0, 5, 10]
        mask = result["event_mask"][:, 0, 5, 10]
        self.assertIn(BUILDING_DIFFRACTION, types[mask])
        self.assertGreater(float(result["counterfactual_gain_db"][0, 5, 10]), 0.0)

    def test_ridge_is_terrain_not_building_event(self):
        from jiangxi_radio.propagation_events import (
            BUILDING_DIFFRACTION,
            TERRAIN_DIFFRACTION,
            build_propagation_events,
        )

        x, y, z, terrain, surface, station = self.geometry()
        terrain[:, 5] = 50.0
        surface[:, 5] = 50.0
        result = build_propagation_events(
            x, y, z, terrain, surface, station,
            samples=9, max_events=4, frequency_mhz=2500.0,
        )
        types = result["event_type"][:, 0, 5, 10]
        mask = result["event_mask"][:, 0, 5, 10]
        self.assertIn(TERRAIN_DIFFRACTION, types[mask])
        self.assertNotIn(BUILDING_DIFFRACTION, types[mask])

    def test_translation_and_chunk_size_do_not_change_events(self):
        from jiangxi_radio.propagation_events import build_propagation_events

        x, y, z, terrain, surface, station = self.geometry()
        surface[:, 5] = 50.0
        first = build_propagation_events(
            x, y, z, terrain, surface, station,
            samples=9, max_events=4, chunk_size=7,
        )
        shift = np.array([1000.0, -200.0, 300.0])
        second = build_propagation_events(
            x + shift[0], y + shift[1], z + shift[2],
            terrain + shift[2], surface + shift[2], station + shift,
            samples=9, max_events=4, chunk_size=4096,
        )
        np.testing.assert_allclose(
            first["event_features"], second["event_features"], atol=1e-5
        )
        np.testing.assert_array_equal(first["event_type"], second["event_type"])
        np.testing.assert_array_equal(first["event_mask"], second["event_mask"])
        np.testing.assert_allclose(
            first["counterfactual_gain_db"],
            second["counterfactual_gain_db"],
            atol=1e-5,
        )

    def test_invalid_geometry_is_rejected(self):
        from jiangxi_radio.propagation_events import build_propagation_events

        x, y, z, terrain, surface, station = self.geometry()
        with self.assertRaises(ValueError):
            build_propagation_events(x[::-1], y, z, terrain, surface, station)
        with self.assertRaises(ValueError):
            build_propagation_events(x, y, z, terrain, surface, station, max_events=0)
        with self.assertRaises(ValueError):
            build_propagation_events(x, y, z, terrain, surface, station, samples=1.5)


if __name__ == "__main__":
    unittest.main()
