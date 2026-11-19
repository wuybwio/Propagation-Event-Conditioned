import unittest

import numpy as np

from jiangxi_radio.multi_mechanism_propagation import (
    combine_multi_edge_losses_db,
    synthesize_multi_mechanism_teacher,
    two_ray_reflection_gain_db,
)


class MultiMechanismPropagationTests(unittest.TestCase):
    def test_multi_edge_envelope_is_bounded_and_monotone(self):
        low = combine_multi_edge_losses_db(np.array([[2.0, 1.0, 0.0]]))
        high = combine_multi_edge_losses_db(np.array([[5.0, 1.0, 0.0]]))
        self.assertGreaterEqual(float(high[0]), float(low[0]))
        self.assertLessEqual(float(high[0]), 60.0)

    def test_two_ray_is_finite_and_bounded(self):
        value = two_ray_reflection_gain_db(
            np.array([100.0, 1000.0]), 30.0, np.array([20.0, 120.0]),
            frequency_mhz=2500.0,
        )
        self.assertTrue(np.isfinite(value).all())
        self.assertLessEqual(float(value.max()), 6.0)
        self.assertGreaterEqual(float(value.min()), -30.0)

    def test_teacher_signs_are_explicit(self):
        result = synthesize_multi_mechanism_teacher(
            np.array([-70.0]), np.array([10.0]), np.array([2.0]), np.array([1.0])
        )
        self.assertAlmostEqual(float(result["teacher_dbm"][0]), -79.0)


if __name__ == "__main__":
    unittest.main()
