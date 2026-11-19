import unittest

import numpy as np
import pandas as pd

from jiangxi_radio.calibration import (
    cross_validate_residual_calibrator,
    fit_residual_binning_calibrator,
    prepare_lte_measurements,
    spatial_block_kfold,
    spatial_block_split,
)


class CalibrationTests(unittest.TestCase):
    def test_prepare_lte_measurements_normalizes_positive_rsrp_magnitudes(self):
        source = pd.DataFrame(
            {
                "longitude": [115.7576, 115.7577, 115.7578],
                "latitude": [28.6545, 28.6546, 28.6547],
                "ltescrsrp": [58, -82, 250],
                "ltescrsrq": [22, 11, 8],
            }
        )

        cleaned = prepare_lte_measurements(source)

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned["rsrp_dbm"].tolist(), [-58.0, -82.0])
        self.assertEqual(cleaned["rsrq_db"].tolist(), [-22.0, -11.0])

    def test_spatial_block_split_keeps_blocks_entirely_in_one_partition(self):
        source = pd.DataFrame(
            {
                "longitude": [115.00, 115.01, 115.20, 115.21, 115.40, 115.41],
                "latitude": [28.00, 28.01, 28.20, 28.21, 28.40, 28.41],
                "rsrp_dbm": [-90.0] * 6,
            }
        )

        train, validation = spatial_block_split(
            source, block_size_degrees=0.1, validation_fraction=1 / 3, seed=7
        )

        train_blocks = set(zip((train.longitude // 0.1), (train.latitude // 0.1)))
        validation_blocks = set(
            zip((validation.longitude // 0.1), (validation.latitude // 0.1))
        )
        self.assertTrue(train_blocks)
        self.assertTrue(validation_blocks)
        self.assertFalse(train_blocks.intersection(validation_blocks))

    def test_residual_calibrator_does_not_leak_validation_residual(self):
        training = pd.DataFrame(
            {
                "longitude": [115.00, 115.01],
                "latitude": [28.00, 28.01],
                "rsrp_dbm": [-89.0, -91.0],
                "physical_prediction_dbm": [-100.0, -100.0],
            }
        )
        held_out = pd.DataFrame(
            {
                "longitude": [116.00],
                "latitude": [29.00],
                "rsrp_dbm": [-20.0],
                "physical_prediction_dbm": [-100.0],
            }
        )

        calibrator = fit_residual_binning_calibrator(training, bin_size_degrees=0.1)
        predicted = calibrator.predict_residual(held_out)

        self.assertTrue(np.allclose(predicted, [10.0]))

    def test_spatial_kfold_keeps_blocks_disjoint(self):
        source = pd.DataFrame({
            "longitude": [115.00, 115.01, 115.20, 115.21, 115.40, 115.41],
            "latitude": [28.00, 28.01, 28.20, 28.21, 28.40, 28.41],
            "rsrp_dbm": [-90.0] * 6,
            "physical_prediction_dbm": [-95.0] * 6,
        })
        folds = spatial_block_kfold(source, block_size_degrees=0.1, folds=3, seed=3)
        self.assertEqual(len(folds), 3)
        for train, validation in folds:
            train_blocks = set(zip(train.longitude // 0.1, train.latitude // 0.1))
            validation_blocks = set(zip(validation.longitude // 0.1, validation.latitude // 0.1))
            self.assertFalse(train_blocks.intersection(validation_blocks))

    def test_spatial_cv_reports_calibrated_error(self):
        source = pd.DataFrame({
            "longitude": [115.00, 115.01, 115.20, 115.21, 115.40, 115.41],
            "latitude": [28.00, 28.01, 28.20, 28.21, 28.40, 28.41],
            "rsrp_dbm": [-90.0, -91.0, -88.0, -89.0, -92.0, -93.0],
            "physical_prediction_dbm": [-95.0] * 6,
        })
        result = cross_validate_residual_calibrator(
            source, block_size_degrees=0.1, folds=3, seed=2
        )
        self.assertEqual(len(result), 3)
        self.assertTrue((result["calibrated_mae_db"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
