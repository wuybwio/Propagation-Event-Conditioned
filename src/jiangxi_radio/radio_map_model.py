from __future__ import annotations

try:
    import torch
    from torch import nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class _Block(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            groups = 4 if out_channels >= 4 else 1
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.GroupNorm(groups, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.GroupNorm(groups, out_channels),
                nn.SiLU(),
            )

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.layers(value)


    class PhysicsGuidedProbabilisticRadioMap(nn.Module):
        """P²-RMNet: a physical-prior residual U-Net with uncertainty output.

        ``physical_prior`` is a single-channel probability map. The network
        predicts a residual logit correction and conditional probabilities for
        at-least-1/2/3 detection, preserving the monotonic ordering by
        construction.
        """

        def __init__(
            self,
            environment_channels: int,
            base_channels: int = 16,
            residual_scale: float = 0.35,
        ) -> None:
            super().__init__()
            if environment_channels <= 0 or base_channels <= 0:
                raise ValueError("channel counts must be positive")
            if residual_scale <= 0:
                raise ValueError("residual_scale must be positive")
            self.residual_scale = float(residual_scale)
            self.height_embedding = nn.Sequential(
                nn.Linear(1, base_channels), nn.SiLU(), nn.Linear(base_channels, base_channels)
            )
            in_channels = environment_channels + 1 + 3 + 1 + base_channels
            self.enc1 = _Block(in_channels, base_channels)
            self.enc2 = _Block(base_channels, base_channels * 2)
            self.bottleneck = _Block(base_channels * 2, base_channels * 4)
            self.dec2 = _Block(base_channels * 4 + base_channels * 2, base_channels * 2)
            self.dec1 = _Block(base_channels * 2 + base_channels, base_channels)
            # RadioUNet++-inspired coarse-to-fine refinement.  The zero
            # initialization makes this an identity adapter for old
            # probability-only checkpoints while new training can learn local
            # building-edge and street-canyon corrections.
            self.coarse_signal_head = nn.Conv2d(base_channels, 1, 1)
            self.refinement_gate = nn.Conv2d(base_channels + 1, base_channels, 1)
            self.physical_signal_gate = nn.Conv2d(1, base_channels, 1)
            nn.init.zeros_(self.coarse_signal_head.weight)
            nn.init.zeros_(self.coarse_signal_head.bias)
            nn.init.zeros_(self.refinement_gate.weight)
            nn.init.zeros_(self.refinement_gate.bias)
            nn.init.zeros_(self.physical_signal_gate.weight)
            nn.init.zeros_(self.physical_signal_gate.bias)
            self.residual_head = nn.Conv2d(base_channels, 1, 1)
            # Auxiliary received-signal head.  Training this head in dBm space
            # gives the coverage model a physically interpretable objective;
            # the probability heads remain the planning interface.
            self.signal_head = nn.Conv2d(base_channels, 1, 1)
            # New training starts as the physical signal baseline and learns a
            # bounded correction.  This prevents the auxiliary signal head
            # from discarding the interpretable propagation estimate.
            nn.init.zeros_(self.signal_head.weight)
            nn.init.zeros_(self.signal_head.bias)
            self.probability_head = nn.Conv2d(base_channels, 3, 1)
            self.log_variance_head = nn.Conv2d(base_channels, 3, 1)

        @staticmethod
        def _check_inputs(
            environment: torch.Tensor,
            physical_prior: torch.Tensor,
            station_field: torch.Tensor,
            route_weight: torch.Tensor,
            height_m: torch.Tensor,
        ) -> None:
            if physical_prior.shape[1] != 1:
                raise ValueError("physical_prior must have one channel")
            if station_field.shape[1] != 3 or route_weight.shape[1] != 1:
                raise ValueError("station_field must have 3 channels and route_weight one channel")
            spatial = environment.shape[-2:]
            if any(value.shape[-2:] != spatial for value in (physical_prior, station_field, route_weight)):
                raise ValueError("all spatial input sizes must match")
            batch = environment.shape[0]
            if any(value.shape[0] != batch for value in (physical_prior, station_field, route_weight)):
                raise ValueError("all inputs must share the same batch dimension")
            if height_m.reshape(-1).shape[0] != batch:
                raise ValueError("height_m must contain one value per batch item")

        def forward(
            self,
            *,
            environment: torch.Tensor,
            physical_prior: torch.Tensor,
            station_field: torch.Tensor,
            route_weight: torch.Tensor,
            height_m: torch.Tensor,
            physical_signal: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            baseline_provided = physical_signal is not None
            if physical_signal is None:
                physical_signal = torch.zeros_like(environment[:, :1])
            self._check_inputs(environment, physical_prior, station_field, route_weight, height_m)
            if physical_signal.shape != environment[:, :1].shape:
                raise ValueError("physical_signal must have shape [batch, 1, row, col]")
            height = height_m.reshape(-1, 1).to(dtype=environment.dtype) / 1000.0
            height_map = self.height_embedding(height).unsqueeze(-1).unsqueeze(-1)
            height_map = height_map.expand(-1, -1, environment.shape[-2], environment.shape[-1])
            value = torch.cat((environment, physical_prior, station_field, route_weight, height_map), dim=1)
            skip1 = self.enc1(value)
            skip2 = self.enc2(F.max_pool2d(skip1, 2))
            latent = self.bottleneck(F.max_pool2d(skip2, 2))
            decoded = F.interpolate(latent, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
            decoded = self.dec2(torch.cat((decoded, skip2), dim=1))
            decoded = F.interpolate(decoded, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
            decoded = self.dec1(torch.cat((decoded, skip1), dim=1))
            coarse_signal = torch.sigmoid(self.coarse_signal_head(decoded))
            refined = (
                decoded
                + self.refinement_gate(torch.cat((decoded, coarse_signal), dim=1))
                + self.physical_signal_gate(physical_signal)
            )
            residual = self.residual_head(refined)
            if baseline_provided:
                signal_residual = self.residual_scale * torch.tanh(self.signal_head(refined))
                signal_prediction = (physical_signal + signal_residual).clamp(0.0, 1.0)
            else:
                # Preserve the absolute-signal behavior of probability-only
                # checkpoints that predate the physical signal baseline.
                signal_prediction = torch.sigmoid(self.signal_head(refined))
                signal_residual = signal_prediction - physical_signal
            prior = physical_prior.clamp(1e-5, 1.0 - 1e-5)
            prior_logit = torch.log(prior) - torch.log1p(-prior)
            raw = self.probability_head(refined)
            p1 = torch.sigmoid(prior_logit + residual + raw[:, 0:1])
            p2 = p1 * torch.sigmoid(raw[:, 1:2])
            p3 = p2 * torch.sigmoid(raw[:, 2:3])
            uncertainty = F.softplus(self.log_variance_head(refined)) + 1e-4
            return {
                "probability": torch.cat((p1, p2, p3), dim=1),
                "uncertainty": uncertainty,
                "residual": residual,
                "signal_residual": signal_residual,
                "physical_signal": physical_signal,
                "signal_prediction": signal_prediction,
            }

else:

    class PhysicsGuidedProbabilisticRadioMap:  # pragma: no cover
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "PyTorch is required for PhysicsGuidedProbabilisticRadioMap; "
                "install a CUDA-enabled PyTorch wheel for training."
            )
