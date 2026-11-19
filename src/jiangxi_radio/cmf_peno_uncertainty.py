"""Shared uncertainty parameterization for CMF-PENO.

The model head emits a log variance.  Both the heteroscedastic objective and
the reported predictive variance must use the same transform; otherwise the
network is optimized for one uncertainty variable and evaluated as another.
"""

from __future__ import annotations

import torch


def variance_from_log_variance(log_variance: torch.Tensor) -> torch.Tensor:
    """Convert log variance to strictly positive variance in dB^2.

    ``exp(log_variance)`` is used because the training objective is the
    Gaussian negative log-likelihood in log-variance form.  Clamping the log
    variance before exponentiation prevents overflow while preserving the
    intended parameterization.
    """

    return torch.exp(log_variance.clamp(min=-20.0, max=20.0)).clamp_min(1e-6)
