import unittest

import numpy as np

from jiangxi_radio.frozen_v3_release import (
    conformal_multiplier,
    greedy_select_sites,
    interval_metrics,
    representative_case_index,
    union_detection_probability,
    validation_bias_shift,
)


class FrozenV3ReleaseTests(unittest.TestCase):
    def test_validation_bias_shift_is_fitted_on_masked_validation_voxels(self):
        mean = np.array([0.0, 2.0, 100.0])
        target = np.array([1.0, 5.0, -100.0])
        mask = np.array([True, True, False])
        self.assertAlmostEqual(validation_bias_shift(mean, target, mask), 2.0)

    def test_conformal_multiplier_reaches_validation_target(self):
        mean = np.zeros(20)
        target = np.arange(20, dtype=float) / 10.0
        variance = np.ones(20)
        mask = np.ones(20, dtype=bool)
        q = conformal_multiplier(mean, variance, target, mask, 0.90)
        metrics = interval_metrics(mean, variance, target, mask, q)
        self.assertGreaterEqual(metrics["coverage"], 0.90)
        self.assertGreater(metrics["mean_width_db"], 0.0)

    def test_union_probability_is_independent_at_least_one_probability(self):
        values = np.array([[0.2, 0.5], [0.4, 0.25]])
        actual = union_detection_probability(values, axis=0)
        np.testing.assert_allclose(actual, np.array([0.52, 0.625]))

    def test_greedy_selection_finds_complementary_sites(self):
        candidates = np.array([[0.9, 0.1], [0.1, 0.9], [0.4, 0.4]])
        selected, union = greedy_select_sites(candidates, 2, np.ones(2, dtype=bool))
        self.assertEqual(selected, [0, 1])
        np.testing.assert_allclose(union, np.array([0.91, 0.91]))

    def test_representative_case_is_nearest_median_mae(self):
        prediction = np.array([[0.0, 0.0], [2.0, 2.0], [8.0, 8.0]])
        target = np.zeros_like(prediction)
        mask = np.ones_like(prediction, dtype=bool)
        self.assertEqual(representative_case_index(prediction, target, mask), 1)


if __name__ == "__main__":
    unittest.main()
