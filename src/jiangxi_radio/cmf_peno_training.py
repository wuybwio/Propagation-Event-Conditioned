"""Losses and uncertainty metrics for CMF-PENO."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .cmf_peno_uncertainty import variance_from_log_variance


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.shape != mask.shape:
        raise ValueError("value and mask must align")
    valid = mask > 0
    if not torch.any(valid):
        raise ValueError("non-empty supervision mask required")
    return value[valid].mean()


def fspl_upper_bound_loss(
    mean_dbm: torch.Tensor,
    coarse_dbm: torch.Tensor,
    mask: torch.Tensor | None = None,
    margin_db: float = 0.5,
) -> torch.Tensor:
    if mean_dbm.shape != coarse_dbm.shape:
        raise ValueError("mean and coarse fields must align")
    if mask is None:
        mask = torch.ones_like(mean_dbm)
    penalty = F.relu(mean_dbm - coarse_dbm - float(margin_db)).square()
    return _masked_mean(penalty, mask)


def mechanism_upper_bound_loss(
    mean_dbm: torch.Tensor,
    upper_reference_dbm: torch.Tensor,
    mask: torch.Tensor | None = None,
    margin_db: float = 0.5,
) -> torch.Tensor:
    """Penalize predictions above a mechanism-aware constructive bound.

    Unlike the old FSPL-only bound, this permits a bounded reflection gain.
    ``upper_reference_dbm`` must already include the allowed positive mechanism
    contribution and must not include the target field.
    """

    if mean_dbm.shape != upper_reference_dbm.shape:
        raise ValueError("mean and mechanism upper reference must align")
    if mask is None:
        mask = torch.ones_like(mean_dbm)
    penalty = F.relu(mean_dbm - upper_reference_dbm - float(margin_db)).square()
    return _masked_mean(penalty, mask)


def counterfactual_consistency_loss(
    original_dbm: torch.Tensor,
    counterfactual_dbm: torch.Tensor,
    target_gain_db: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if original_dbm.shape != counterfactual_dbm.shape or original_dbm.shape != target_gain_db.shape:
        raise ValueError("counterfactual fields must align")
    if mask is None:
        mask = torch.ones_like(original_dbm)
    predicted_gain = counterfactual_dbm - original_dbm
    fit = F.smooth_l1_loss(predicted_gain, target_gain_db, reduction="none")
    monotonic = F.relu(-predicted_gain).square()
    return _masked_mean(fit + 0.25 * monotonic, mask)


def strong_obstruction_mask(
    air_mask: torch.Tensor,
    counterfactual_gain_db: torch.Tensor,
    minimum_gain_db: float = 3.0,
) -> torch.Tensor:
    if air_mask.shape != counterfactual_gain_db.shape:
        raise ValueError("air mask and counterfactual gain must align")
    if not math.isfinite(minimum_gain_db) or minimum_gain_db < 0.0:
        raise ValueError("minimum gain threshold must be finite and non-negative")
    return air_mask.bool() & (counterfactual_gain_db >= float(minimum_gain_db))


def heteroscedastic_nll(
    mean_dbm: torch.Tensor,
    log_variance: torch.Tensor,
    target_dbm: torch.Tensor,
    mask: torch.Tensor,
    *,
    detach_mean: bool = False,
) -> torch.Tensor:
    if mean_dbm.shape != log_variance.shape or mean_dbm.shape != target_dbm.shape:
        raise ValueError("NLL fields must align")
    variance = variance_from_log_variance(log_variance)
    nll_mean = mean_dbm.detach() if detach_mean else mean_dbm
    value = 0.5 * (
        (target_dbm - nll_mean).square() / variance + torch.log(variance)
    )
    return _masked_mean(value, mask)


def cmf_peno_loss(
    output: dict[str, torch.Tensor],
    fidelity_dbm: torch.Tensor,
    mask: torch.Tensor,
    *,
    coarse: torch.Tensor,
    upper_reference: torch.Tensor | None = None,
    stage_weight: float = 0.5,
    mean_weight: float = 1.0,
    nll_weight: float = 1.0,
    upper_weight: float = 0.02,
    counterfactual_gain_db: torch.Tensor | None = None,
    counterfactual_output: torch.Tensor | None = None,
    counterfactual_mask: torch.Tensor | None = None,
    counterfactual_weight: float = 0.1,
    detach_mean_for_nll: bool = False,
) -> dict[str, torch.Tensor]:
    # Dataset contract: [N,F,Z,Y,X]; model fields: [N,1,Z,Y,X].
    if fidelity_dbm.ndim != 5 or fidelity_dbm.shape[2:] != output["mean_dbm"].shape[2:]:
        raise ValueError("fidelity_dbm must have shape [N,F,Z,Y,X]")
    if output["mean_dbm"].shape != mask.shape or coarse.shape != mask.shape:
        raise ValueError("output, coarse and mask must align")
    if fidelity_dbm.shape[0] != mask.shape[0]:
        raise ValueError("fidelity and mask case counts must align")
    if any(value < 0.0 for value in (stage_weight, mean_weight, nll_weight, upper_weight, counterfactual_weight)):
        raise ValueError("loss weights must be non-negative")
    stage_target = fidelity_dbm[:, 1:2]
    stage = F.smooth_l1_loss(
        output["stage1_dbm"] / 20.0,
        stage_target / 20.0,
        reduction="none",
    )
    stage = _masked_mean(stage, mask)
    target = fidelity_dbm[:, -1:]
    mean = F.smooth_l1_loss(
        output["mean_dbm"] / 20.0,
        target / 20.0,
        reduction="none",
    )
    mean = _masked_mean(mean, mask)
    nll = heteroscedastic_nll(
        output["mean_dbm"], output["log_variance"], target, mask,
        detach_mean=detach_mean_for_nll,
    )
    upper = (
        fspl_upper_bound_loss(output["mean_dbm"], coarse, mask)
        if upper_reference is None
        else mechanism_upper_bound_loss(output["mean_dbm"], upper_reference, mask)
    )
    counterfactual = torch.zeros((), dtype=nll.dtype, device=nll.device)
    if counterfactual_gain_db is not None and counterfactual_output is not None:
        selected = mask if counterfactual_mask is None else counterfactual_mask
        if selected.shape != mask.shape:
            raise ValueError("counterfactual mask must align with supervision mask")
        if torch.any(selected > 0):
            counterfactual = counterfactual_consistency_loss(
                output["mean_dbm"], counterfactual_output, counterfactual_gain_db, selected
            )
    total = (
        stage_weight * stage
        + mean_weight * mean
        + nll_weight * nll
        + upper_weight * upper
        + counterfactual_weight * counterfactual
    )
    return {
        "total": total,
        "stage1": stage,
        "mean": mean,
        "nll": nll,
        "upper": upper,
        "counterfactual": counterfactual,
    }


def uncertainty_interval_metrics(
    mean_dbm: torch.Tensor,
    variance_db2: torch.Tensor,
    target_dbm: torch.Tensor,
    z_value: float = 1.96,
) -> dict[str, float]:
    if mean_dbm.shape != variance_db2.shape or mean_dbm.shape != target_dbm.shape:
        raise ValueError("uncertainty fields must align")
    if not torch.isfinite(mean_dbm).all() or not torch.isfinite(variance_db2).all() or not torch.isfinite(target_dbm).all():
        raise ValueError("uncertainty values must be finite")
    if torch.any(variance_db2 < 0.0) or not math.isfinite(z_value) or z_value <= 0.0:
        raise ValueError("variance and z value must be valid")
    std = variance_db2.clamp_min(1e-6).sqrt()
    error = target_dbm - mean_dbm
    covered = (error.abs() <= z_value * std).float()
    standardized = error / std
    return {
        "coverage_95": float(covered.mean()),
        "mean_interval_width": float((2.0 * z_value * std).mean()),
        "standardized_error_rmse": float(standardized.square().mean().sqrt()),
        "mean_std_db": float(std.mean()),
    }
