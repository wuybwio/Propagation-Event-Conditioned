from __future__ import annotations

import math


def free_space_path_loss_db(frequency_mhz: float, distance_m: float) -> float:
    if frequency_mhz <= 0 or distance_m <= 0:
        raise ValueError("frequency and distance must be positive")
    distance_km = distance_m / 1000.0
    return 32.45 + 20.0 * math.log10(frequency_mhz) + 20.0 * math.log10(distance_km)


def combine_powers_dbm(*powers_dbm: float) -> float:
    if not powers_dbm:
        raise ValueError("at least one power value is required")
    total_mw = sum(10.0 ** (power / 10.0) for power in powers_dbm)
    return 10.0 * math.log10(total_mw)


def received_power_dbm(
    transmitter_power_dbm: float,
    transmitter_gain_dbi: float,
    receiver_gain_dbi: float,
    path_loss_db: float,
    additional_loss_db: float = 0.0,
) -> float:
    return (
        transmitter_power_dbm
        + transmitter_gain_dbi
        + receiver_gain_dbi
        - path_loss_db
        - additional_loss_db
    )


def detection_margin_db(
    received_dbm: float,
    noise_dbm: float,
    interference_dbm: float,
    required_snr_db: float,
    sensitivity_dbm: float | None = None,
) -> float:
    noise_interference_dbm = combine_powers_dbm(noise_dbm, interference_dbm)
    sinr_threshold_dbm = noise_interference_dbm + required_snr_db
    effective_threshold_dbm = (
        sinr_threshold_dbm
        if sensitivity_dbm is None
        else max(sensitivity_dbm, sinr_threshold_dbm)
    )
    return received_dbm - effective_threshold_dbm
