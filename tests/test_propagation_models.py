import unittest

import numpy as np

from jiangxi_radio.propagation_models import (
    EngineeringPropagationMode,
    compare_decision_metrics,
    compare_with_reference,
    compute_additional_loss_db,
    default_scenarios,
    evidence_metadata,
    run_reference_adapter,
    spearman_rank_correlation,
    summarize_scenario_values,
)


class PropagationModelTests(unittest.TestCase):
    def test_default_scenarios_are_finite_and_distinct(self):
        scenarios = default_scenarios()
        self.assertEqual(len(scenarios), 3)
        self.assertEqual(len({s.name for s in scenarios}), 3)

    def test_summary_and_reference_comparison(self):
        values = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        summary = summarize_scenario_values(values)
        self.assertEqual(summary["p50"].tolist(), [2.0, 3.0])
        metrics = compare_with_reference(values[0], values[1])
        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertIn("spearman", metrics)
        self.assertEqual(run_reference_adapter(np.zeros((2, 2)), lambda x: x.sum(axis=1)).shape, (2,))

    def test_spearman_and_decision_metrics(self):
        left = np.array([1.0, 2.0, 3.0, 4.0])
        right = np.array([2.0, 4.0, 6.0, 8.0])
        self.assertAlmostEqual(spearman_rank_correlation(left, right), 1.0)
        metrics = compare_decision_metrics(
            left,
            right,
            engineering_cost=10.0,
            reference_cost=12.0,
            engineering_coverage=0.9,
            reference_coverage=0.95,
            engineering_runtime_s=1.0,
            reference_runtime_s=10.0,
        )
        self.assertAlmostEqual(metrics["cost_delta"], -2.0)
        self.assertAlmostEqual(metrics["runtime_speedup"], 10.0)

    def test_ablation_modes_are_mutually_consistent(self):
        lon = np.array([0.0, 0.01, 0.02])
        lat = np.array([0.0])
        building = np.zeros((3, 5), dtype=float)
        building[1, 2] = 0.4  # 80 m when scale=100
        points = np.array([[0.02, 0.0]])
        stations = np.array([[0.0, 0.0]])
        axes = (lon, lat)
        fspl = compute_additional_loss_db(
            EngineeringPropagationMode.FSPL,
            points=points,
            station_coordinates=stations,
        )
        prior = compute_additional_loss_db(
            EngineeringPropagationMode.GRID_BUILDING_PRIOR,
            points=points,
            building_features=building[:1],
            flight_height_m=120.0,
            max_loss_db=18.0,
        )
        knife = compute_additional_loss_db(
            EngineeringPropagationMode.KNIFE_EDGE,
            points=points,
            station_coordinates=stations,
            building_features=building,
            axes=axes,
            station_height_m=30.0,
            flight_height_m=30.0,
            frequency_mhz=2500.0,
            samples_per_km=50.0,
            target_ground_elevation_m=0.0,
        )
        ensemble = compute_additional_loss_db(
            EngineeringPropagationMode.KNIFE_EDGE_UNCERTAINTY,
            points=points,
            station_coordinates=stations,
            building_features=building,
            axes=axes,
            station_height_m=30.0,
            flight_height_m=30.0,
            samples_per_km=50.0,
            target_ground_elevation_m=0.0,
        )
        self.assertEqual(fspl.shape, (1, 1))
        self.assertEqual(float(fspl[0, 0]), 0.0)
        self.assertEqual(prior.shape, (1,))
        self.assertGreaterEqual(float(knife[0, 0]), 0.0)
        self.assertEqual(len(ensemble), 3)
        self.assertEqual(evidence_metadata(EngineeringPropagationMode.KNIFE_EDGE)["model_tier"], "A_main")

    def test_reference_adapter_rejects_scalar_output(self):
        with self.assertRaises(ValueError):
            run_reference_adapter(np.zeros((2, 2)), lambda _points: np.array(1.0))

    def test_scenario_quantiles_are_validated(self):
        with self.assertRaises(ValueError):
            summarize_scenario_values(np.ones((2, 1)), quantiles=(1.2,))


if __name__ == "__main__":
    unittest.main()
