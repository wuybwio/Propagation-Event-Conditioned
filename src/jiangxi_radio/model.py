"""Optional PyTorch probabilistic multi-height radio-map surrogate."""

from __future__ import annotations

try:  # Keep the physical baseline usable in a NumPy-only environment.
    import torch
    from torch import nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by the base runtime test
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class _DoubleConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.GroupNorm(4 if out_channels >= 4 else 1, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.GroupNorm(4 if out_channels >= 4 else 1, out_channels),
                nn.SiLU(),
            )

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.layers(value)


    class ProbabilisticRadioUNet(nn.Module):
        """Physics-guided surrogate for P(at least 1/2/3 detections).

        Inputs are environmental raster channels, three grade-specific station
        fields, a route-priority raster, and the requested flight height.  The
        cumulative probability head enforces P1 >= P2 >= P3 by construction.
        """

        def __init__(self, environment_channels: int, base_channels: int = 16) -> None:
            super().__init__()
            if environment_channels <= 0 or base_channels <= 0:
                raise ValueError("channel counts must be positive")
            # The height embedding is expanded to one spatial channel per
            # embedding feature before it is concatenated with the rasters.
            input_channels = environment_channels + 3 + 1 + base_channels
            self.height_embedding = nn.Sequential(
                nn.Linear(1, base_channels), nn.SiLU(), nn.Linear(base_channels, base_channels)
            )
            self.enc1 = _DoubleConv(input_channels, base_channels)
            self.pool = nn.MaxPool2d(2)
            self.enc2 = _DoubleConv(base_channels, base_channels * 2)
            self.bottleneck = _DoubleConv(base_channels * 2, base_channels * 4)
            self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
            self.dec2 = _DoubleConv(base_channels * 4, base_channels * 2)
            self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
            self.dec1 = _DoubleConv(base_channels * 2, base_channels)
            self.probability_head = nn.Conv2d(base_channels, 3, 1)
            self.log_variance_head = nn.Conv2d(base_channels, 3, 1)

        def forward(
            self,
            *,
            environment: torch.Tensor,
            station_field: torch.Tensor,
            route_weight: torch.Tensor,
            height_m: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            if station_field.shape[1] != 3 or route_weight.shape[1] != 1:
                raise ValueError("station_field must have 3 channels and route_weight one channel")
            if environment.shape[-2:] != station_field.shape[-2:] or route_weight.shape[-2:] != station_field.shape[-2:]:
                raise ValueError("all spatial input sizes must match")
            height = height_m.reshape(-1, 1).to(dtype=environment.dtype) / 1000.0
            height_map = self.height_embedding(height).unsqueeze(-1).unsqueeze(-1)
            height_map = height_map.expand(-1, -1, environment.shape[-2], environment.shape[-1])
            value = torch.cat((environment, station_field, route_weight, height_map), dim=1)
            skip1 = self.enc1(value)
            skip2 = self.enc2(self.pool(skip1))
            latent = self.bottleneck(self.pool(skip2))
            decoded = self.up2(latent)
            decoded = self.dec2(torch.cat((decoded, skip2), dim=1))
            decoded = self.up1(decoded)
            decoded = self.dec1(torch.cat((decoded, skip1), dim=1))
            raw_probability = self.probability_head(decoded)
            p1 = torch.sigmoid(raw_probability[:, 0:1])
            p2 = p1 * torch.sigmoid(raw_probability[:, 1:2])
            p3 = p2 * torch.sigmoid(raw_probability[:, 2:3])
            probability = torch.cat((p1, p2, p3), dim=1)
            uncertainty = F.softplus(self.log_variance_head(decoded)) + 1e-4
            return {"probability": probability, "uncertainty": uncertainty}

else:

    class ProbabilisticRadioUNet:  # pragma: no cover - simple dependency guard
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "PyTorch is required for ProbabilisticRadioUNet; install the CUDA wheel "
                "for GPU training or use the NumPy physical baseline."
            )
