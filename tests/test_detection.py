import unittest

import numpy as np

from jiangxi_radio.detection import (
    at_least_k_probability,
    batch_at_least_k_probability,
    correlated_at_least_k_probability,
    detection_probability,
)


class DetectionProbabilityTests(unittest.TestCase):
    def test_batch_poisson_binomial_matches_scalar(self):
        values = np.array([[0.2, 0.4, 0.8], [0.9, 0.1, 0.3]])
        required = np.array([1, 2])
        actual = batch_at_least_k_probability(values, required)
        expected = np.array([
            at_least_k_probability(values[0], 1),
            at_least_k_probability(values[1], 2),
        ])
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_batch_poisson_binomial_matches_scalar_for_mixed_redundancy(self):
        values = np.array([
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
        ])
        required = np.array([1, 2, 3])
        actual = batch_at_least_k_probability(values, required)
        expected = np.array([
            at_least_k_probability(values[row], required[row])
            for row in range(len(values))
        ])
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_zero_mean_margin_has_half_detection_probability(self):
        self.assertAlmostEqual(detection_probability(0.0, 6.0), 0.5)

    def test_probability_increases_with_detection_margin(self):
        values = detection_probability(np.array([-6.0, 0.0, 6.0]), 6.0)
        self.assertTrue(np.all(np.diff(values) > 0))

    def test_three_equal_stations_have_half_probability_of_at_least_two(self):
        self.assertAlmostEqual(at_least_k_probability([0.5, 0.5, 0.5], 2), 0.5)

    def test_at_least_k_handles_boundaries(self):
        self.assertEqual(at_least_k_probability([0.2, 0.7], 0), 1.0)
        self.assertEqual(at_least_k_probability([0.2, 0.7], 3), 0.0)

    def test_rejects_invalid_station_probability(self):
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            at_least_k_probability([0.4, 1.2], 1)

    def test_correlated_probability_is_reproducible_and_lower_for_redundancy(self):
        values = [0.9, 0.9, 0.9]
        independent = at_least_k_probability(values, 2)
        actual = correlated_at_least_k_probability(values, 2, correlation=0.8, samples=1024, seed=11)
        repeat = correlated_at_least_k_probability(values, 2, correlation=0.8, samples=1024, seed=11)
        self.assertEqual(actual, repeat)
        self.assertLess(actual, independent)


if __name__ == "__main__":
    unittest.main()
