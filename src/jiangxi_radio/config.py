from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True)
class StudyConfig:
    heights_m: tuple[int, ...]
    primary_heights_m: tuple[int, ...]
    route_sample_km: float


@dataclass(frozen=True)
class CoverageConfig:
    probability_threshold: float
    ordinary_k: int
    critical_k: int
    high_risk_k: int

    def required_stations(self, level: str) -> int:
        values = {
            "ordinary": self.ordinary_k,
            "critical": self.critical_k,
            "high_risk": self.high_risk_k,
        }
        try:
            return values[level]
        except KeyError as exc:
            raise ValueError(f"unknown route priority level: {level}") from exc


@dataclass(frozen=True)
class CostConfig:
    budget: float
    upgrade: float
    relocate: float
    new_station: float


@dataclass(frozen=True)
class RadioConfig:
    frequency_mhz: float
    emitter_power_dbm: float
    emitter_gain_dbi: float
    noise_power_dbm: float
    interference_power_dbm: float
    required_snr_db: float
    shadow_std_db: float


@dataclass(frozen=True)
class StationGradeProfile:
    receiver_gain_dbi: float
    antenna_height_m: float
    sensitivity_dbm: float
    required_sinr_db: float


@dataclass(frozen=True)
class EnvironmentConfig:
    """Optional, calibratable proxy for height-dependent environmental loss."""

    enabled: bool = False
    base_loss_db: float = 0.0
    clutter_loss_db: float = 0.0
    height_scale_m: float = 300.0
    reference_height_m: float = 0.0

    def loss_db(self, flight_height_m: float) -> float:
        if not self.enabled:
            return 0.0
        if self.height_scale_m <= 0:
            raise ValueError("environment height_scale_m must be positive")
        height_delta = max(0.0, float(flight_height_m) - self.reference_height_m)
        return float(self.base_loss_db + self.clutter_loss_db * math.exp(-height_delta / self.height_scale_m))


@dataclass(frozen=True)
class ScenarioConfig:
    study: StudyConfig
    coverage: CoverageConfig
    cost: CostConfig
    radio: RadioConfig
    station_grades: dict[int, StationGradeProfile]
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"missing [{name}] configuration section")
    return section


def load_scenario(path: str | Path) -> ScenarioConfig:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)

    study_raw = _section(raw, "study")
    coverage_raw = _section(raw, "coverage")
    cost_raw = _section(raw, "cost")
    radio_raw = _section(raw, "radio")
    environment_raw = raw.get("environment", {})
    if not isinstance(environment_raw, dict):
        raise ValueError("[environment] must be a table when provided")

    study = StudyConfig(
        heights_m=tuple(int(value) for value in study_raw["heights_m"]),
        primary_heights_m=tuple(
            int(value) for value in study_raw["primary_heights_m"]
        ),
        route_sample_km=float(study_raw["route_sample_km"]),
    )
    if not set(study.primary_heights_m).issubset(study.heights_m):
        raise ValueError("primary heights must be included in evaluation heights")
    if study.route_sample_km <= 0:
        raise ValueError("route_sample_km must be positive")

    coverage = CoverageConfig(
        probability_threshold=float(coverage_raw["probability_threshold"]),
        ordinary_k=int(coverage_raw["ordinary_k"]),
        critical_k=int(coverage_raw["critical_k"]),
        high_risk_k=int(coverage_raw["high_risk_k"]),
    )
    if not 0 < coverage.probability_threshold < 1:
        raise ValueError("probability_threshold must be between 0 and 1")
    if not (1 <= coverage.ordinary_k <= coverage.critical_k <= coverage.high_risk_k):
        raise ValueError("coverage redundancy levels must be ordered positive integers")

    cost = CostConfig(**{name: float(cost_raw[name]) for name in (
        "budget", "upgrade", "relocate", "new_station"
    )})
    if min(cost.budget, cost.upgrade, cost.relocate, cost.new_station) < 0:
        raise ValueError("cost values must be non-negative")

    radio = RadioConfig(**{name: float(radio_raw[name]) for name in (
        "frequency_mhz",
        "emitter_power_dbm",
        "emitter_gain_dbi",
        "noise_power_dbm",
        "interference_power_dbm",
        "required_snr_db",
        "shadow_std_db",
    )})
    if radio.frequency_mhz <= 0 or radio.shadow_std_db <= 0:
        raise ValueError("radio frequency and shadow standard deviation must be positive")

    environment = EnvironmentConfig(
        enabled=bool(environment_raw.get("enabled", False)),
        base_loss_db=float(environment_raw.get("base_loss_db", 0.0)),
        clutter_loss_db=float(environment_raw.get("clutter_loss_db", 0.0)),
        height_scale_m=float(environment_raw.get("height_scale_m", 300.0)),
        reference_height_m=float(environment_raw.get("reference_height_m", 0.0)),
    )
    if environment.height_scale_m <= 0:
        raise ValueError("environment height_scale_m must be positive")
    if environment.base_loss_db < 0 or environment.clutter_loss_db < 0:
        raise ValueError("environment losses must be non-negative")

    grade_raw = _section(raw, "station_grades")
    station_grades: dict[int, StationGradeProfile] = {}
    for grade_name, values in grade_raw.items():
        grade = int(grade_name)
        if grade not in {2, 3, 4} or not isinstance(values, dict):
            raise ValueError(f"unsupported station grade profile: {grade_name}")
        profile = StationGradeProfile(
            receiver_gain_dbi=float(values["receiver_gain_dbi"]),
            antenna_height_m=float(values["antenna_height_m"]),
            sensitivity_dbm=float(values["sensitivity_dbm"]),
            required_sinr_db=float(values["required_sinr_db"]),
        )
        if profile.antenna_height_m <= 0:
            raise ValueError("station antenna height must be positive")
        station_grades[grade] = profile
    if set(station_grades) != {2, 3, 4}:
        raise ValueError("station grade profiles 2, 3, and 4 are required")

    return ScenarioConfig(
        study=study,
        coverage=coverage,
        cost=cost,
        radio=radio,
        station_grades=station_grades,
        environment=environment,
    )
