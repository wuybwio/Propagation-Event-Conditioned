import unittest

import torch

from jiangxi_radio.cmf_peno_v3 import (
    CMFPENOV3,
    low_high_residual_targets,
    multi_edge_counterfactual_gain_db,
    v3_mean_loss,
    v3_variance_loss,
)


class CMFPENOV3Tests(unittest.TestCase):
    def _inputs(self):
        torch.manual_seed(7)
        return (
            torch.randn(2, 8, 4, 6, 6),
            torch.randn(2, 1, 4, 6, 6),
            torch.randn(2, 4, 14, 4, 6, 6),
            torch.zeros(2, 4, 4, 6, 6, dtype=torch.long),
            torch.ones(2, 4, 4, 6, 6, dtype=torch.bool),
        )

    def test_fourier_kernel_is_learnable_and_preserves_shape(self):
        model = CMFPENOV3(
            input_channels=8, width=8, event_hidden=8,
            event_feature_count=14, modes=(2, 2, 2),
        )
        volume, coarse, events, event_type, event_mask = self._inputs()
        output = model(volume, coarse, events, event_type, event_mask)
        self.assertEqual(tuple(output["mean_dbm"].shape), (2, 1, 4, 6, 6))
        self.assertTrue(any("complex_weight" in name for name, _ in model.named_parameters()))

    def test_non_direct_event_contributions_are_non_positive(self):
        model = CMFPENOV3(
            input_channels=8, width=8, event_hidden=8,
            event_feature_count=14, modes=(2, 2, 2),
        )
        volume, coarse, events, event_type, event_mask = self._inputs()
        output = model(volume, coarse, events, event_type, event_mask)
        contribution = output["event_contribution"][:, 1:]
        self.assertTrue(bool((contribution <= 1e-6).all()))

    def test_removing_an_obstruction_cannot_reduce_stage1_power(self):
        model = CMFPENOV3(
            input_channels=8, width=8, event_hidden=8,
            event_feature_count=14, modes=(2, 2, 2),
        )
        volume, coarse, events, event_type, event_mask = self._inputs()
        event_type[:, 1:] = 2
        original = model(volume, coarse, events, event_type, event_mask)
        removed_mask = event_mask.clone()
        removed_mask[:, 1] = False
        removed = model(volume, coarse, events, event_type, removed_mask)
        self.assertTrue(bool((removed["stage1_dbm"] >= original["stage1_dbm"] - 1e-6).all()))

    def test_low_high_targets_reconstruct_residual(self):
        residual = torch.randn(2, 1, 4, 6, 6)
        low, high = low_high_residual_targets(residual, modes=(2, 2, 2))
        torch.testing.assert_close(low + high, residual, atol=1e-5, rtol=1e-5)

    def test_variance_loss_detaches_mean_features(self):
        model = CMFPENOV3(
            input_channels=8, width=8, event_hidden=8,
            event_feature_count=14, modes=(2, 2, 2),
        )
        volume, coarse, events, event_type, event_mask = self._inputs()
        output = model(volume, coarse, events, event_type, event_mask)
        targets = torch.randn(2, 3, 4, 6, 6)
        mask = torch.ones(2, 1, 4, 6, 6)
        loss = v3_variance_loss(output, targets, mask)
        loss.backward()
        self.assertIsNotNone(model.variance_head[0].weight.grad)
        self.assertIsNone(model.input_projection.weight.grad)

    def test_counterfactual_gain_recomputes_ranked_multi_edge_envelope(self):
        features = torch.zeros(1, 4, 14, 1, 1, 1)
        mask = torch.ones(1, 4, 1, 1, 1, dtype=torch.bool)
        features[0, 1, 11, 0, 0, 0] = 30.0 / 60.0
        features[0, 2, 11, 0, 0, 0] = 20.0 / 60.0
        features[0, 3, 11, 0, 0, 0] = 10.0 / 60.0
        gain = multi_edge_counterfactual_gain_db(features, mask)
        # Original: 30 + .35*20 + .15*10 = 38.5; remove strongest:
        # 20 + .35*10 = 23.5, hence a 15 dB gain.
        torch.testing.assert_close(gain, torch.tensor([[[[[15.0]]]]]))

    def test_mean_loss_skips_empty_counterfactual_selection(self):
        model = CMFPENOV3(
            input_channels=8, width=8, event_hidden=8,
            event_feature_count=14, modes=(2, 2, 2),
        )
        volume, coarse, events, event_type, event_mask = self._inputs()
        output = model(volume, coarse, events, event_type, event_mask)
        removed_mask = event_mask.clone()
        counterfactual_output = model(volume, coarse, events, event_type, removed_mask)
        fidelity = torch.stack([coarse[:, 0], output["mean_dbm"].detach()[:, 0]], dim=1)
        supervision = torch.ones_like(coarse)
        empty_counterfactual = torch.zeros_like(supervision, dtype=torch.bool)
        result = v3_mean_loss(
            output,
            fidelity,
            supervision,
            coarse=coarse,
            counterfactual_gain_db=torch.ones_like(supervision),
            counterfactual_output=counterfactual_output,
            counterfactual_mask=empty_counterfactual,
        )
        self.assertTrue(torch.isfinite(result["total"]))
        self.assertEqual(float(result["counterfactual"]), 0.0)

    def test_structural_ablation_flags_disable_named_outputs(self):
        volume, coarse, events, event_type, event_mask = self._inputs()
        event_type[:, 1:] = 2
        model = CMFPENOV3(
            input_channels=8, width=8, event_hidden=8,
            event_feature_count=14, modes=(2, 2, 2),
            use_events=False, use_local=False, use_spectral=False,
        )
        output = model(volume, coarse, events, event_type, event_mask)
        torch.testing.assert_close(
            output["event_contribution"],
            torch.zeros_like(output["event_contribution"]),
        )
        torch.testing.assert_close(
            output["local_residual_db"],
            torch.zeros_like(output["local_residual_db"]),
        )
        torch.testing.assert_close(
            output["global_residual_db"],
            torch.zeros_like(output["global_residual_db"]),
        )


if __name__ == "__main__":
    unittest.main()
