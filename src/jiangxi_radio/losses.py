"""Losses for probability-map surrogate training (optional PyTorch module)."""

from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for probabilistic map losses")


def masked_gaussian_nll(
    prediction: "torch.Tensor",
    target: "torch.Tensor",
    uncertainty: "torch.Tensor",
    mask: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """Heteroscedastic NLL with positive standard deviation parameterization."""
    _require_torch()
    # Keep the heteroscedastic head numerically bounded.  Without an upper
    # bound the model can drive the variance toward zero on a few pixels,
    # making the validation NLL explode and destabilising the probability
    # head.  The targets are normalized to [0, 1], so this range is wide
    # enough to represent useful uncertainty while remaining well-conditioned.
    standard_deviation = uncertainty.clamp(min=0.05, max=1.0)
    variance = standard_deviation.square()
    value = 0.5 * ((prediction - target).square() / variance + variance.log())
    if mask is not None:
        value = value * mask.to(dtype=value.dtype)
        return value.sum() / mask.sum().clamp_min(1)
    return value.mean()


def probabilistic_map_loss(
    output: dict[str, "torch.Tensor"],
    target_probability: "torch.Tensor",
    *,
    mask: "torch.Tensor | None" = None,
    route_weight: "torch.Tensor | None" = None,
    physical_weight: float = 1.0,
    brier_weight: float = 0.5,
    monotonic_weight: float = 0.2,
) -> "torch.Tensor":
    """Combine uncertainty-aware physical loss, Brier calibration, and monotonicity."""
    _require_torch()
    probability = output["probability"]
    uncertainty = output["uncertainty"]
    weight = route_weight if route_weight is not None else None
    nll = masked_gaussian_nll(probability, target_probability, uncertainty, mask)
    brier_value = (probability - target_probability).square()
    brier_weight_tensor = torch.ones_like(brier_value)
    if weight is not None:
        brier_weight_tensor = brier_weight_tensor * weight
    if mask is not None:
        brier_weight_tensor = brier_weight_tensor * mask.to(dtype=brier_value.dtype)
    brier = (brier_value * brier_weight_tensor).sum() / brier_weight_tensor.sum().clamp_min(1.0)
    monotonic = torch.relu(probability[:, 1:] - probability[:, :-1]).mean()
    return physical_weight * nll + brier_weight * brier + monotonic_weight * monotonic


def masked_signal_huber_loss(
    prediction: "torch.Tensor",
    target: "torch.Tensor",
    mask: "torch.Tensor",
    *,
    signal_min_dbm: float = -135.0,
    signal_max_dbm: float = -40.0,
    delta_db: float = 6.0,
) -> "torch.Tensor":
    """Huber loss for normalized signal tensors, evaluated in physical dBm."""
    _require_torch()
    scale = float(signal_max_dbm - signal_min_dbm)
    if scale <= 0 or delta_db <= 0:
        raise ValueError("signal range and delta_db must be positive")
    prediction_dbm = prediction * scale + signal_min_dbm
    target_dbm = target * scale + signal_min_dbm
    value = torch.nn.functional.huber_loss(
        prediction_dbm, target_dbm, reduction="none", delta=float(delta_db)
    )
    weight = mask.to(dtype=value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1)


def masked_signal_gradient_loss(
    prediction: "torch.Tensor",
    target: "torch.Tensor",
    mask: "torch.Tensor",
    *,
    signal_min_dbm: float = -135.0,
    signal_max_dbm: float = -40.0,
) -> "torch.Tensor":
    """Match local dBm gradients to preserve building/street-canyon detail.

    The finite differences are evaluated only where both neighboring pixels
    are supervised. This auxiliary term is deliberately separate from the
    signal Huber loss so it can be ablated without changing the target.
    """
    _require_torch()
    scale = float(signal_max_dbm - signal_min_dbm)
    if scale <= 0:
        raise ValueError("signal range must be positive")
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target and mask must have the same shape")
    prediction_dbm = prediction * scale + signal_min_dbm
    target_dbm = target * scale + signal_min_dbm
    values = []
    for dim in (-2, -1):
        pred_diff = torch.diff(prediction_dbm, dim=dim)
        target_diff = torch.diff(target_dbm, dim=dim)
        if dim == -2:
            pair_mask = mask[..., 1:, :] * mask[..., :-1, :]
        else:
            pair_mask = mask[..., :, 1:] * mask[..., :, :-1]
        values.append((pred_diff - target_diff).abs() * pair_mask.to(pred_diff.dtype))
    numerator = sum(value.sum() for value in values)
    # Use supervised-pair count rather than total pair count when possible.
    supervised = sum(
        float((mask[..., 1:, :] * mask[..., :-1, :]).sum())
        if dim == -2
        else float((mask[..., :, 1:] * mask[..., :, :-1]).sum())
        for dim in (-2, -1)
    )
    return numerator / max(supervised, 1.0)


def cross_height_consistency_loss(
    signal_prediction: "torch.Tensor",
    heights_m: "torch.Tensor",
    mask: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """Penalize curvature of the predicted signal profile across heights.

    The loss uses a non-uniform finite difference, so the requested heights do
    not need to be equally spaced.  A linear signal-vs-height profile has zero
    loss.  The optional mask is reduced to a spatial weight shared across
    adjacent height triples; unlabeled pixels are otherwise still useful for
    the physics-guided extrapolation constraint.
    """
    _require_torch()
    if signal_prediction.ndim != 4:
        raise ValueError("signal_prediction must have shape [height, channel, row, col]")
    # Work in kilometres so the regularization weight is numerically useful
    # for flight heights expressed in metres.
    heights = heights_m.reshape(-1).to(dtype=signal_prediction.dtype, device=signal_prediction.device) / 1000.0
    if signal_prediction.shape[0] != heights.numel():
        raise ValueError("heights_m must match the signal height dimension")
    if heights.numel() < 3:
        return signal_prediction.new_zeros(())
    if not bool(torch.all(torch.diff(heights) > 0)):
        raise ValueError("heights_m must be strictly increasing")
    first = torch.diff(signal_prediction, dim=0) / torch.diff(heights).reshape(-1, 1, 1, 1)
    curvature = torch.diff(first, dim=0)
    value = curvature.abs()
    if mask is not None:
        valid = mask.to(dtype=value.dtype)
        if valid.shape[0] != signal_prediction.shape[0]:
            raise ValueError("mask must match signal height dimension")
        valid = valid[2:] + valid[1:-1] + valid[:-2]
        valid = (valid > 0).to(dtype=value.dtype)
        loss = (value * valid).sum() / valid.sum().clamp_min(1)
    else:
        loss = value.mean()
    # Interpolation of a mathematically linear profile can leave a tiny
    # floating-point residue (especially with float32 tensors).  Treat that
    # residue as exact zero so the regularizer has a stable zero point.
    return torch.where(loss.abs() < 1e-6, torch.zeros_like(loss), loss)
