"""Keep measurement evidence separate from monitoring-scenario simulations."""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Mapping, Any

import numpy as np


MONITORING_SIMULATION = "monitoring_simulation"
LEGACY_MIXED = "legacy_mixed_lte_proxy"


def experiment_domain(payload: Mapping[str, Any]) -> str:
    return str(np.asarray(payload["experiment_domain"]).item()) if "experiment_domain" in payload else "unknown_legacy"


def require_monitoring_contract(payload: Mapping[str, Any], *, allow_legacy: bool = False) -> str:
    domain = experiment_domain(payload)
    if domain != MONITORING_SIMULATION:
        message = (
            f"Experiment domain is {domain!r}. LTE-derived or unmarked labels are not "
            "fixed-monitoring-station ground truth. Use scripts/evaluate_lte_rsrp_baselines.py "
            "for LTE observations, or rebuild monitoring_simulation data. "
            "--allow-legacy-proxy is only for historical diagnostics."
        )
        if not allow_legacy:
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=2)
    return domain


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_evaluation_dataset(data: Mapping[str, Any], checkpoint: dict, path: str | Path,
                                *, allow_legacy: bool = False) -> str:
    domain = require_monitoring_contract(data, allow_legacy=allow_legacy)
    checkpoint_domain = require_monitoring_contract(checkpoint, allow_legacy=allow_legacy)
    if domain != checkpoint_domain:
        raise ValueError("Dataset and checkpoint experiment domains differ")
    expected = checkpoint.get("dataset_sha256")
    if expected and expected != file_sha256(path):
        raise ValueError("Dataset differs from the checkpoint training dataset; split evaluation is invalid")
    return domain
