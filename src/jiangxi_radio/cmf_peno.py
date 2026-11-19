"""Causal Multi-Fidelity Propagation-Event Neural Operator.

The model is deliberately compact enough for the existing 16x24x24 station
volumes. It is a mechanism-conditioned surrogate, not a replacement for a
full electromagnetic solver.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .cmf_peno_uncertainty import variance_from_log_variance


def _normal_cdf(value: np.ndarray) -> np.ndarray:
    from math import erf, sqrt

    flat = np.asarray(value, dtype=float).ravel()
    return (0.5 * (1.0 + np.fromiter((erf(float(v)) for v in flat), float, len(flat)))).reshape(
        np.asarray(value).shape
    )


def combined_detection_probability(
    mean_dbm: np.ndarray,
    variance_db2: np.ndarray,
    threshold_dbm: float = -100.0,
    shadow_std_db: float = 6.0,
) -> np.ndarray:
    """Combine model variance and log-normal shadowing in a soft detector."""
    mean = np.asarray(mean_dbm, dtype=float)
    variance = np.asarray(variance_db2, dtype=float)
    if mean.shape != variance.shape or not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise ValueError("mean and variance must be aligned and finite")
    if np.any(variance < 0.0) or not math.isfinite(threshold_dbm) or not math.isfinite(shadow_std_db) or shadow_std_db <= 0.0:
        raise ValueError("variance must be non-negative and shadow std positive")
    denominator = np.sqrt(shadow_std_db**2 + variance)
    return _normal_cdf((mean - threshold_dbm) / denominator)


def remove_strongest_obstruction(
    event_features: torch.Tensor,
    event_type: torch.Tensor,
    event_mask: torch.Tensor,
) -> torch.Tensor:
    """Return a counterfactual mask with the strongest non-direct event removed."""
    if event_features.ndim != 6 or event_type.ndim != 5 or event_mask.shape != event_type.shape:
        raise ValueError("expected event features [N,K,F,Z,Y,X] and aligned masks")
    if event_features.shape[:2] != event_type.shape[:2] or event_features.shape[3:] != event_type.shape[2:]:
        raise ValueError("event feature and type shapes are not aligned")
    result = event_mask.clone()
    if event_features.shape[1] <= 1:
        return result
    strength = event_features[:, :, 11].abs().masked_fill(~event_mask, -torch.inf)
    strength[:, 0] = -torch.inf
    index = strength.argmax(dim=1, keepdim=True)
    any_event = event_mask[:, 1:].any(dim=1, keepdim=True)
    result.scatter_(1, index, False)
    return torch.where(any_event, result, event_mask)


class SpectralCorrectionStage(nn.Module):
    """Low-mode 3-D Fourier correction with a local pointwise residual."""

    def __init__(
        self,
        channels: int,
        width: int,
        modes: tuple[int, int, int],
        gate_init: float = 0.1,
    ):
        super().__init__()
        if any(int(mode) < 1 for mode in modes):
            raise ValueError("spectral modes must be positive")
        if not math.isfinite(gate_init) or not 0.0 < gate_init < 1.0:
            raise ValueError("spectral gate initialization must be in (0,1)")
        self.width = int(width)
        self.modes = tuple(int(mode) for mode in modes)
        self.in_projection = nn.Conv3d(channels, width, 1)
        self.local = nn.Conv3d(width, width, 3, padding=1)
        self.out_projection = nn.Conv3d(width, 2, 1)
        self.gate_logit = nn.Parameter(torch.tensor(math.log(gate_init / (1.0 - gate_init))))

    @property
    def gate_value(self) -> float:
        return float(torch.sigmoid(self.gate_logit).detach())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        projected = self.in_projection(value)
        local = F.silu(self.local(projected))
        spectral = torch.fft.rfftn(projected, dim=(-3, -2, -1), norm="ortho")
        filtered = torch.zeros_like(spectral)
        mz, my, mx = self.modes
        filtered[..., :mz, :my, :mx] = spectral[..., :mz, :my, :mx]
        filtered[..., -mz:, :my, :mx] = spectral[..., -mz:, :my, :mx]
        long_range = torch.fft.irfftn(filtered, s=projected.shape[-3:], dim=(-3, -2, -1), norm="ortho")
        return self.out_projection(F.silu(local + torch.sigmoid(self.gate_logit) * long_range))


class CMFPENO(nn.Module):
    """Event-set encoder followed by a staged 3-D neural operator."""

    def __init__(
        self,
        input_channels: int,
        width: int = 24,
        event_hidden: int = 32,
        event_types: int = 6,
        event_feature_count: int = 12,
        mechanism_channels: int = 0,
        modes: tuple[int, int, int] = (4, 4, 4),
        use_spectral: bool = True,
        use_events: bool = True,
        use_uncertainty: bool = True,
        spectral_gate_init: float = 0.1,
        residual_db: float = 60.0,
    ):
        super().__init__()
        if (
            input_channels < 1
            or width < 4
            or event_hidden < 4
            or event_types < 1
            or event_feature_count < 1
            or mechanism_channels < 0
        ):
            raise ValueError("model widths and input channels must be positive")
        if not math.isfinite(residual_db) or residual_db <= 0.0:
            raise ValueError("residual_db must be positive")
        self.input_channels = int(input_channels)
        self.width = int(width)
        self.event_hidden = int(event_hidden)
        self.event_types = int(event_types)
        self.event_feature_count = int(event_feature_count)
        self.mechanism_channels = int(mechanism_channels)
        self.residual_db = float(residual_db)
        self.use_spectral = bool(use_spectral)
        self.use_events = bool(use_events)
        self.use_uncertainty = bool(use_uncertainty)
        self.spectral_gate_init = float(spectral_gate_init)
        self.event_encoder = nn.Sequential(
            nn.Linear(self.event_feature_count, event_hidden), nn.LayerNorm(event_hidden), nn.SiLU(),
            nn.Linear(event_hidden, event_hidden), nn.SiLU(),
        )
        self.type_embedding = nn.Embedding(event_types, event_hidden)
        self.event_to_volume = nn.Conv3d(event_hidden, width, 1)
        self.input_projection = nn.Conv3d(
            input_channels + width + 1 + self.mechanism_channels, width, 1
        )
        self.stage1 = nn.Sequential(
            nn.Conv3d(width, width, 3, padding=1), nn.GroupNorm(4, width), nn.SiLU(),
            nn.Conv3d(width, width, 3, padding=1), nn.SiLU(),
        )
        self.stage1_head = nn.Conv3d(width, 1, 1)
        self.correction = SpectralCorrectionStage(width, width, modes, gate_init=spectral_gate_init) if use_spectral else nn.Sequential(
            nn.Conv3d(width, width, 3, padding=1), nn.SiLU(), nn.Conv3d(width, 2, 1)
        )
        self.final_head = nn.Conv3d(2, 2, 1)
        self.log_variance_bias = nn.Parameter(torch.tensor(math.log(9.0)))
        for layer in (self.stage1_head, self.final_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @property
    def spectral_gate_value(self) -> float:
        if isinstance(self.correction, SpectralCorrectionStage):
            return self.correction.gate_value
        return 0.0

    def _event_context(
        self,
        event_features: torch.Tensor,
        event_type: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if event_features.ndim != 6 or event_type.ndim != 5 or event_mask.shape != event_type.shape:
            raise ValueError("invalid propagation event tensors")
        n, k, feature_count, z, y, x = event_features.shape
        if feature_count != self.event_feature_count or event_type.shape != (n, k, z, y, x):
            raise ValueError(
                f"event tensors must use [N,K,{self.event_feature_count},Z,Y,X]"
            )
        flat = event_features.permute(0, 1, 3, 4, 5, 2).reshape(-1, feature_count)
        embedded = self.event_encoder(flat).reshape(n, k, z, y, x, self.event_hidden)
        types = self.type_embedding(event_type.long().clamp(0, self.event_types - 1))
        types = types.permute(0, 1, 2, 3, 4, 5)
        embedded = embedded + types
        scores = embedded.mean(dim=-1).masked_fill(~event_mask, -torch.inf)
        attention = torch.softmax(scores, dim=1)
        attention = torch.where(event_mask, attention, torch.zeros_like(attention))
        denominator = attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        attention = attention / denominator
        context = (embedded * attention[..., None]).sum(dim=1)
        return context.permute(0, 4, 1, 2, 3), attention

    def forward(
        self,
        volume: torch.Tensor,
        coarse_dbm: torch.Tensor,
        event_features: torch.Tensor,
        event_type: torch.Tensor,
        event_mask: torch.Tensor,
        mechanism_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if volume.ndim != 5 or volume.shape[1] != self.input_channels:
            raise ValueError("volume must have shape [N,input_channels,Z,Y,X]")
        if coarse_dbm.shape != volume[:, :1].shape:
            raise ValueError("coarse_dbm must align with volume")
        if self.mechanism_channels:
            if mechanism_context is None or mechanism_context.ndim != 5:
                raise ValueError("mechanism_context is required for RF models")
            if mechanism_context.shape != (
                volume.shape[0], self.mechanism_channels, *volume.shape[2:]
            ):
                raise ValueError("mechanism_context does not align with volume")
        elif mechanism_context is not None:
            raise ValueError("mechanism_context is not enabled for this model")
        if self.use_events:
            context, attention = self._event_context(event_features, event_type, event_mask)
            context = self.event_to_volume(context)
        else:
            context = torch.zeros(
                volume.shape[0], self.width, *volume.shape[2:],
                dtype=volume.dtype, device=volume.device,
            )
            attention = torch.zeros(event_mask.shape, dtype=volume.dtype, device=volume.device)
        feature_values = [volume, coarse_dbm / 60.0, context]
        if self.mechanism_channels:
            feature_values.append(mechanism_context)
        features = torch.cat(feature_values, dim=1)
        hidden = self.input_projection(features)
        stage_hidden = self.stage1(hidden)
        stage1 = coarse_dbm + self.residual_db * torch.tanh(self.stage1_head(stage_hidden))
        correction = self.correction(stage_hidden)
        correction = self.final_head(correction)
        mean_residual = self.residual_db * torch.tanh(correction[:, :1])
        mean_dbm = stage1 + mean_residual
        if self.use_uncertainty:
            log_variance = correction[:, 1:2] + self.log_variance_bias
            variance_db2 = variance_from_log_variance(log_variance)
        else:
            log_variance = torch.zeros_like(mean_dbm)
            variance_db2 = torch.zeros_like(mean_dbm)
        return {
            "mean_dbm": mean_dbm,
            "stage1_dbm": stage1,
            "log_variance": log_variance,
            "variance_db2": variance_db2,
            "event_attention": attention,
        }
