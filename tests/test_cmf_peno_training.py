import unittest

import torch


class CMFPENOTrainingTests(unittest.TestCase):
    def assert_no_effective_gradient(self, tensor):
        if tensor.grad is not None:
            torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor))

    def test_log_variance_parameterization_is_shared_by_model_and_nll(self):
        from jiangxi_radio.cmf_peno_uncertainty import variance_from_log_variance
        from jiangxi_radio.cmf_peno_training import heteroscedastic_nll

        expected_variance = torch.tensor([1.0, 4.0, 9.0])
        log_variance = expected_variance.log()
        mean = torch.zeros(3)
        target = expected_variance.sqrt()
        mask = torch.ones(3)

        torch.testing.assert_close(
            variance_from_log_variance(log_variance), expected_variance
        )
        expected_nll = 0.5 * (1.0 + log_variance).mean()
        torch.testing.assert_close(
            heteroscedastic_nll(mean, log_variance, target, mask), expected_nll
        )

    def test_log_variance_conversion_remains_finite_at_extreme_logits(self):
        from jiangxi_radio.cmf_peno_uncertainty import variance_from_log_variance

        result = variance_from_log_variance(torch.tensor([-1.0e6, 1.0e6]))
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.all(result > 0.0))

    def test_detached_nll_updates_variance_without_updating_mean(self):
        from jiangxi_radio.cmf_peno_training import heteroscedastic_nll

        mean = torch.tensor([0.0, 1.0], requires_grad=True)
        log_variance = torch.zeros(2, requires_grad=True)
        target = torch.tensor([2.0, -1.0])
        mask = torch.ones(2)

        loss = heteroscedastic_nll(
            mean, log_variance, target, mask, detach_mean=True
        )
        loss.backward()

        self.assert_no_effective_gradient(mean)
        self.assertIsNotNone(log_variance.grad)
        self.assertGreater(float(log_variance.grad.abs().sum()), 0.0)

    def test_multifidelity_loss_forwards_detached_nll_policy(self):
        from jiangxi_radio.cmf_peno_training import cmf_peno_loss

        shape = (1, 1, 1, 1, 2)
        mean = torch.zeros(shape, requires_grad=True)
        log_variance = torch.zeros(shape, requires_grad=True)
        output = {
            "stage1_dbm": torch.zeros(shape, requires_grad=True),
            "mean_dbm": mean,
            "log_variance": log_variance,
            "variance_db2": torch.ones(shape),
        }
        fidelity = torch.zeros((1, 3, 1, 1, 2))
        fidelity[:, -1:] = 2.0
        mask = torch.ones(shape)
        losses = cmf_peno_loss(
            output,
            fidelity,
            mask,
            coarse=fidelity[:, :1],
            stage_weight=0.0,
            mean_weight=0.0,
            nll_weight=1.0,
            upper_weight=0.0,
            counterfactual_weight=0.0,
            detach_mean_for_nll=True,
        )
        losses["total"].backward()
        self.assert_no_effective_gradient(mean)
        self.assertIsNotNone(log_variance.grad)

    def test_multifidelity_loss_has_stage_and_final_terms(self):
        from jiangxi_radio.cmf_peno_training import cmf_peno_loss

        shape = (1, 1, 4, 4, 4)
        output = {
            "stage1_dbm": torch.full(shape, -75.0, requires_grad=True),
            "mean_dbm": torch.full(shape, -70.0, requires_grad=True),
            "log_variance": torch.zeros(shape, requires_grad=True),
            "variance_db2": torch.ones(shape, requires_grad=True),
        }
        field_shape = (1, 4, 4, 4)
        fidelity = torch.stack(
            [torch.full(field_shape, -60.0), torch.full(field_shape, -72.0)], dim=1
        )
        mask = torch.ones(shape)
        result = cmf_peno_loss(output, fidelity, mask, coarse=fidelity[:, 0:1])
        self.assertIn("total", result)
        self.assertGreater(float(result["stage1"].detach()), 0.0)
        self.assertGreater(float(result["mean"].detach()), 0.0)
        self.assertGreater(float(result["nll"].detach()), 0.0)
        result["total"].backward()
        self.assertIsNotNone(output["mean_dbm"].grad)

    def test_deterministic_loss_still_supervises_final_mean(self):
        from jiangxi_radio.cmf_peno_training import cmf_peno_loss

        shape = (1, 1, 2, 2, 2)
        mean = torch.full(shape, -60.0, requires_grad=True)
        output = {
            "stage1_dbm": torch.full(shape, -72.0, requires_grad=True),
            "mean_dbm": mean,
            "log_variance": torch.zeros(shape, requires_grad=True),
            "variance_db2": torch.ones(shape),
        }
        field_shape = (1, 2, 2, 2)
        fidelity = torch.stack([
            torch.full(field_shape, -60.0),
            torch.full(field_shape, -72.0),
            torch.full(field_shape, -82.0),
        ], dim=1)
        mask = torch.ones(shape)
        result = cmf_peno_loss(
            output, fidelity, mask, coarse=fidelity[:, 0:1],
            mean_weight=1.0, nll_weight=0.0, upper_weight=0.0,
            counterfactual_weight=0.0,
        )
        self.assertGreater(float(result["mean"].detach()), 0.0)
        result["total"].backward()
        self.assertIsNotNone(mean.grad)
        self.assertGreater(float(mean.grad.abs().sum()), 0.0)

    def test_upper_bound_penalty_only_penalizes_power_above_coarse(self):
        from jiangxi_radio.cmf_peno_training import fspl_upper_bound_loss

        coarse = torch.full((1, 1, 2, 2, 2), -70.0)
        below = torch.full_like(coarse, -72.0)
        above = torch.full_like(coarse, -60.0)
        self.assertEqual(float(fspl_upper_bound_loss(below, coarse)), 0.0)
        self.assertGreater(float(fspl_upper_bound_loss(above, coarse)), 0.0)

    def test_counterfactual_loss_enforces_nonnegative_removed_obstruction_gain(self):
        from jiangxi_radio.cmf_peno_training import counterfactual_consistency_loss

        original = torch.full((1, 1, 2, 2, 2), -80.0)
        counterfactual = torch.full_like(original, -72.0)
        target_gain = torch.full_like(original, 8.0)
        result = counterfactual_consistency_loss(original, counterfactual, target_gain)
        self.assertAlmostEqual(float(result), 0.0, places=6)
        bad = counterfactual_consistency_loss(original, torch.full_like(original, -85.0), target_gain)
        self.assertGreater(float(bad), 0.0)

    def test_strong_obstruction_mask_rejects_weak_or_missing_events(self):
        from jiangxi_radio.cmf_peno_training import strong_obstruction_mask

        air = torch.ones((1, 1, 1, 2, 3))
        gain = torch.tensor([[[[[0.5, 1.0, 3.0], [0.0, 2.9, 4.0]]]]])
        result = strong_obstruction_mask(air, gain, minimum_gain_db=3.0)
        expected = torch.tensor([[[[[False, False, True], [False, False, True]]]]])
        torch.testing.assert_close(result, expected)

    def test_uncertainty_interval_metrics_are_finite(self):
        from jiangxi_radio.cmf_peno_training import uncertainty_interval_metrics

        mean = torch.tensor([-80.0, -70.0, -60.0])
        variance = torch.tensor([1.0, 4.0, 9.0])
        target = torch.tensor([-79.0, -74.0, -62.0])
        result = uncertainty_interval_metrics(mean, variance, target)
        self.assertTrue(0.0 <= result["coverage_95"] <= 1.0)
        self.assertTrue(result["mean_interval_width"] > 0.0)
        self.assertTrue(result["standardized_error_rmse"] >= 0.0)


if __name__ == "__main__":
    unittest.main()
