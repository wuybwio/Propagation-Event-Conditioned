"""CMF-PENO-RF v3: factorized events, learnable Fourier correction, and uncertainty."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .cmf_peno_uncertainty import variance_from_log_variance


def _frequency_mask(shape: tuple[int, int, int], modes: tuple[int, int, int], device) -> torch.Tensor:
    z, y, x = shape
    mz, my, mx = (min(int(value), size) for value, size in zip(modes, (z, y, x // 2 + 1)))
    z_keep = torch.zeros(z, dtype=torch.bool, device=device)
    y_keep = torch.zeros(y, dtype=torch.bool, device=device)
    z_keep[:mz] = True
    y_keep[:my] = True
    if mz:
        z_keep[max(z - mz, mz):] = True
    if my:
        y_keep[max(y - my, my):] = True
    mask = z_keep[:, None, None] & y_keep[None, :, None]
    mask = mask.expand(z, y, x // 2 + 1).clone()
    if mx < x // 2 + 1:
        mask[..., mx:] = False
    return mask


def low_high_residual_targets(
    residual: torch.Tensor,
    modes: tuple[int, int, int] = (4, 6, 6),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a residual field into a fixed low-frequency and complementary high-frequency field."""
    if residual.ndim != 5:
        raise ValueError("residual must have shape [N,1,Z,Y,X]")
    spectrum = torch.fft.rfftn(residual, dim=(-3, -2, -1), norm="ortho")
    mask = _frequency_mask(tuple(residual.shape[-3:]), modes, residual.device)
    low = torch.fft.irfftn(spectrum * mask, s=residual.shape[-3:], dim=(-3, -2, -1), norm="ortho")
    return low, residual - low


def _ranked_multi_edge_envelope(loss_db: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = torch.where(mask, loss_db.clamp(0.0, 60.0), torch.zeros_like(loss_db))
    ranked = torch.sort(values, dim=1, descending=True).values
    weights = torch.zeros(ranked.shape[1], dtype=ranked.dtype, device=ranked.device)
    for index, weight in enumerate((1.0, 0.35, 0.15)):
        if index < len(weights):
            weights[index] = weight
    return (ranked * weights.view(1, -1, 1, 1, 1)).sum(dim=1, keepdim=True).clamp(0.0, 60.0)


def multi_edge_counterfactual_gain_db(
    event_features: torch.Tensor,
    event_mask: torch.Tensor,
) -> torch.Tensor:
    """Recompute the multi-edge envelope after removing its strongest obstruction."""
    if event_features.ndim != 6 or event_mask.ndim != 5:
        raise ValueError("event tensors must have [N,K,F,Z,Y,X] and [N,K,Z,Y,X]")
    if event_features.shape[:2] != event_mask.shape[:2] or event_features.shape[3:] != event_mask.shape[2:]:
        raise ValueError("event feature and mask shapes do not align")
    obstruction_mask = event_mask.clone()
    obstruction_mask[:, 0] = False
    loss_db = event_features[:, :, 11].abs() * 60.0
    original = _ranked_multi_edge_envelope(loss_db, obstruction_mask)
    strongest = loss_db.masked_fill(~obstruction_mask, -torch.inf).argmax(dim=1, keepdim=True)
    removed = obstruction_mask.clone()
    removed.scatter_(1, strongest, False)
    any_obstruction = obstruction_mask.any(dim=1, keepdim=True)
    counterfactual = _ranked_multi_edge_envelope(loss_db, removed)
    return torch.where(any_obstruction, (original - counterfactual).clamp_min(0.0), torch.zeros_like(original))


class LearnableFourier3D(nn.Module):
    """A 3-D Fourier layer with learnable complex mode mixing and a local skip."""

    def __init__(self, channels: int, modes: tuple[int, int, int]):
        super().__init__()
        if channels < 1 or any(int(value) < 1 for value in modes):
            raise ValueError("channels and Fourier modes must be positive")
        self.channels = int(channels)
        self.modes = tuple(int(value) for value in modes)
        scale = 1.0 / max(channels, 1)
        self.complex_weight = nn.Parameter(
            scale * torch.randn(channels, channels, *self.modes, 2)
        )
        self.local = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5 or value.shape[1] != self.channels:
            raise ValueError("Fourier input must have shape [N,channels,Z,Y,X]")
        n, _, z, y, x = value.shape
        x_ft = torch.fft.rfftn(value, dim=(-3, -2, -1), norm="ortho")
        out_ft = torch.zeros_like(x_ft)
        mz = min(self.modes[0], z)
        my = min(self.modes[1], y)
        mx = min(self.modes[2], x // 2 + 1)
        weight = torch.view_as_complex(self.complex_weight).to(x_ft.dtype)
        z_ranges = (slice(0, mz), slice(max(z - mz, mz), z))
        y_ranges = (slice(0, my), slice(max(y - my, my), y))
        for z_slice in z_ranges:
            for y_slice in y_ranges:
                source = x_ft[:, :, z_slice, y_slice, :mx]
                if source.shape[2] == 0 or source.shape[3] == 0:
                    continue
                kernel = weight[:, :, :source.shape[2], :source.shape[3], :mx]
                out_ft[:, :, z_slice, y_slice, :mx] = torch.einsum(
                    "bizyx,iozyx->bozyx", source, kernel
                )
        global_value = torch.fft.irfftn(
            out_ft, s=(z, y, x), dim=(-3, -2, -1), norm="ortho"
        )
        return F.silu(global_value + self.local(value))


class CMFPENOV3(nn.Module):
    """Mechanism-aware local/global residual operator with a detached variance head."""

    def __init__(
        self,
        input_channels: int,
        width: int = 24,
        event_hidden: int = 32,
        event_types: int = 6,
        event_feature_count: int = 14,
        modes: tuple[int, int, int] = (4, 6, 6),
        residual_db: float = 60.0,
        max_event_loss_db: float = 60.0,
        use_events: bool = True,
        use_local: bool = True,
        use_spectral: bool = True,
    ):
        super().__init__()
        if input_channels < 1 or width < 4 or event_hidden < 4 or event_feature_count < 1:
            raise ValueError("model dimensions must be positive")
        if residual_db <= 0.0 or max_event_loss_db <= 0.0:
            raise ValueError("residual and event bounds must be positive")
        self.input_channels = int(input_channels)
        self.width = int(width)
        self.event_hidden = int(event_hidden)
        self.event_feature_count = int(event_feature_count)
        self.residual_db = float(residual_db)
        self.max_event_loss_db = float(max_event_loss_db)
        self.use_events = bool(use_events)
        self.use_local = bool(use_local)
        self.use_spectral = bool(use_spectral)
        self.event_encoder = nn.Sequential(
            nn.Linear(event_feature_count, event_hidden),
            nn.LayerNorm(event_hidden),
            nn.SiLU(),
            nn.Linear(event_hidden, event_hidden),
            nn.SiLU(),
        )
        self.type_embedding = nn.Embedding(event_types, event_hidden)
        self.event_attention = nn.Linear(event_hidden, 1)
        self.event_contribution_head = nn.Linear(event_hidden, 1)
        self.event_to_volume = nn.Conv3d(event_hidden, width, 1)
        self.input_projection = nn.Conv3d(input_channels + width + 1, width, 1)
        self.local_encoder = nn.Sequential(
            nn.Conv3d(width, width, 3, padding=1),
            nn.GroupNorm(4, width),
            nn.SiLU(),
            nn.Conv3d(width, width, 3, padding=1),
            nn.SiLU(),
        )
        self.fourier = LearnableFourier3D(width, modes)
        self.local_head = nn.Conv3d(width, 1, 1)
        self.global_head = nn.Conv3d(width, 1, 1)
        self.variance_head = nn.Sequential(
            nn.Conv3d(width * 2, width, 1), nn.SiLU(), nn.Conv3d(width, 1, 1)
        )
        self.log_variance_bias = nn.Parameter(torch.tensor(math.log(9.0)))
        for layer in (self.local_head, self.global_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.event_contribution_head.weight)
        nn.init.constant_(self.event_contribution_head.bias, -4.0)
        if not self.use_events:
            for module in (
                self.event_encoder, self.type_embedding, self.event_attention,
                self.event_contribution_head, self.event_to_volume,
            ):
                module.requires_grad_(False)
        if not self.use_local:
            self.local_encoder.requires_grad_(False)
            self.local_head.requires_grad_(False)
        if not self.use_spectral:
            self.fourier.requires_grad_(False)
            self.global_head.requires_grad_(False)

    def _events(self, event_features, event_type, event_mask):
        if event_features.ndim != 6 or event_type.ndim != 5 or event_mask.shape != event_type.shape:
            raise ValueError("invalid event tensor shapes")
        n, k, feature_count, z, y, x = event_features.shape
        if feature_count != self.event_feature_count:
            raise ValueError("event feature count mismatch")
        if not self.use_events:
            context = torch.zeros(
                n, self.width, z, y, x,
                dtype=event_features.dtype, device=event_features.device,
            )
            attention = torch.zeros_like(event_mask, dtype=event_features.dtype)
            contribution = torch.zeros_like(event_mask, dtype=event_features.dtype)
            return context, attention, contribution
        flat = event_features.permute(0, 1, 3, 4, 5, 2).reshape(-1, feature_count)
        hidden = self.event_encoder(flat).reshape(n, k, z, y, x, self.event_hidden)
        hidden = hidden + self.type_embedding(event_type.long().clamp_min(0))
        attention_logits = self.event_attention(hidden).squeeze(-1).masked_fill(~event_mask, -torch.inf)
        attention = torch.softmax(attention_logits, dim=1)
        attention = torch.where(event_mask, attention, torch.zeros_like(attention))
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        context = (hidden * attention[..., None]).sum(dim=1)
        context = self.event_to_volume(context.permute(0, 4, 1, 2, 3))
        raw = self.event_contribution_head(hidden).squeeze(-1)
        non_direct = event_type != 0
        contribution = -self.max_event_loss_db * torch.sigmoid(raw)
        contribution = contribution * event_mask * non_direct
        return context, attention, contribution

    def forward(self, volume, coarse_dbm, event_features, event_type, event_mask):
        if volume.ndim != 5 or coarse_dbm.shape != volume[:, :1].shape:
            raise ValueError("volume/coarse shapes are invalid")
        context, attention, event_contribution = self._events(
            event_features, event_type, event_mask
        )
        hidden = self.input_projection(torch.cat([volume, coarse_dbm / 60.0, context], dim=1))
        local_hidden = self.local_encoder(hidden) if self.use_local else hidden
        global_hidden = self.fourier(local_hidden) if self.use_spectral else torch.zeros_like(local_hidden)
        local_residual = (
            self.residual_db * torch.tanh(self.local_head(local_hidden))
            if self.use_local else torch.zeros_like(coarse_dbm)
        )
        global_residual = (
            self.residual_db * torch.tanh(self.global_head(global_hidden))
            if self.use_spectral else torch.zeros_like(coarse_dbm)
        )
        event_sum = event_contribution.sum(dim=1, keepdim=True)
        stage1 = coarse_dbm + event_sum
        mean_dbm = stage1 + local_residual + global_residual
        variance_input = torch.cat([local_hidden, global_hidden], dim=1).detach()
        log_variance = self.variance_head(variance_input) + self.log_variance_bias
        variance_db2 = variance_from_log_variance(log_variance)
        return {
            "mean_dbm": mean_dbm,
            "stage1_dbm": stage1,
            "local_residual_db": local_residual,
            "global_residual_db": global_residual,
            "event_contribution": event_contribution,
            "event_attention": attention,
            "log_variance": log_variance,
            "variance_db2": variance_db2,
        }


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0
    if not torch.any(valid):
        raise ValueError("empty supervision mask")
    return value[valid].mean()


def v3_variance_loss(
    output: dict[str, torch.Tensor],
    scenario_targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Fit detached aleatoric variance to the distribution of mechanism scenarios."""
    if scenario_targets.ndim != 5 or scenario_targets.shape[0] != output["mean_dbm"].shape[0]:
        raise ValueError("scenario_targets must have shape [N,S,Z,Y,X]")
    mean = output["mean_dbm"].detach()
    variance = output["variance_db2"]
    expanded_mask = mask.expand_as(mean)
    nll = 0.5 * ((scenario_targets.unsqueeze(1) - mean.unsqueeze(1)).square() / variance.unsqueeze(1) + torch.log(variance.unsqueeze(1)))
    return _masked_mean(nll, expanded_mask.unsqueeze(1).expand_as(nll))


def v3_mean_loss(
    output: dict[str, torch.Tensor],
    fidelity_dbm: torch.Tensor,
    mask: torch.Tensor,
    *,
    coarse: torch.Tensor,
    counterfactual_gain_db: torch.Tensor | None = None,
    counterfactual_output: dict[str, torch.Tensor] | None = None,
    counterfactual_mask: torch.Tensor | None = None,
    stage_weight: float = 0.5,
    mean_weight: float = 1.0,
    frequency_weight: float = 0.5,
    orthogonal_weight: float = 0.01,
    counterfactual_weight: float = 0.2,
    upper_weight: float = 0.02,
    allowed_gain_db: float = 7.0,
    modes: tuple[int, int, int] = (4, 6, 6),
) -> dict[str, torch.Tensor]:
    target_stage = fidelity_dbm[:, 1:2]
    target = fidelity_dbm[:, -1:]
    residual = (target - target_stage) * mask
    low_target, high_target = low_high_residual_targets(residual, modes)
    stage = _masked_mean(F.smooth_l1_loss(output["stage1_dbm"] / 20.0, target_stage / 20.0, reduction="none"), mask)
    mean = _masked_mean(F.smooth_l1_loss(output["mean_dbm"] / 20.0, target / 20.0, reduction="none"), mask)
    low = _masked_mean(F.smooth_l1_loss(output["global_residual_db"] / 20.0, low_target / 20.0, reduction="none"), mask)
    high = _masked_mean(F.smooth_l1_loss(output["local_residual_db"] / 20.0, high_target / 20.0, reduction="none"), mask)
    local_flat = (output["local_residual_db"] * mask).flatten(1)
    global_flat = (output["global_residual_db"] * mask).flatten(1)
    orth = (
        (local_flat * global_flat).sum(dim=1).abs()
        / (local_flat.norm(dim=1) * global_flat.norm(dim=1)).clamp_min(1e-6)
    ).mean()
    upper = _masked_mean(F.relu(output["mean_dbm"] - coarse - allowed_gain_db).square(), mask)
    counterfactual = torch.zeros((), dtype=mean.dtype, device=mean.device)
    if counterfactual_output is not None and counterfactual_gain_db is not None:
        selected = mask if counterfactual_mask is None else counterfactual_mask
        delta = counterfactual_output["mean_dbm"] - output["mean_dbm"]
        target_gain = counterfactual_gain_db
        # A batch can contain no voxel with a sufficiently strong obstruction.
        # This disables only the event-consistency term; the ordinary field
        # supervision above still requires a non-empty air mask.
        if torch.any(selected > 0):
            counterfactual = _masked_mean(
                F.smooth_l1_loss(delta, target_gain, reduction="none") + F.relu(-delta).square(),
                selected,
            )
    total = (
        stage_weight * stage
        + mean_weight * mean
        + frequency_weight * (low + high)
        + orthogonal_weight * orth
        + counterfactual_weight * counterfactual
        + upper_weight * upper
    )
    return {
        "total": total,
        "stage1": stage,
        "mean": mean,
        "low": low,
        "high": high,
        "orthogonal": orth,
        "counterfactual": counterfactual,
        "upper": upper,
    }
