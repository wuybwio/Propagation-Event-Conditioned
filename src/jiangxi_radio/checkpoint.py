"""Checkpoint helpers shared by training, evaluation, and inference scripts."""

from __future__ import annotations

from typing import Any


def load_p2rmnet_model(checkpoint: dict[str, Any], environment_channels: int):
    """Instantiate a P2-RMNet checkpoint with backward-compatible loading.

    Signal supervision was added after the first probability-only checkpoints.
    Missing ``signal_head`` parameters are therefore allowed, while any other
    missing or unexpected parameters still fail loudly.
    """
    from jiangxi_radio.radio_map_model import PhysicsGuidedProbabilisticRadioMap

    state = checkpoint["model"]
    inferred = int(state["enc1.layers.0.weight"].shape[0])
    base_channels = int(checkpoint.get("base_channels", inferred))
    model_channels = int(checkpoint.get("environment_channels", environment_channels))
    if model_channels != environment_channels:
        raise ValueError(
            f"checkpoint expects {model_channels} environment channels, "
            f"but dataset/grid provides {environment_channels}"
        )
    model = PhysicsGuidedProbabilisticRadioMap(
        environment_channels=environment_channels,
        base_channels=base_channels,
        residual_scale=float(checkpoint.get("residual_scale", 0.35)),
    )
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = {
        "signal_head.weight", "signal_head.bias",
        "coarse_signal_head.weight", "coarse_signal_head.bias",
        "refinement_gate.weight", "refinement_gate.bias",
        "physical_signal_gate.weight", "physical_signal_gate.bias",
    }
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if not missing.issubset(allowed_missing) or unexpected:
        raise RuntimeError(
            "incompatible P2-RMNet checkpoint: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return model
