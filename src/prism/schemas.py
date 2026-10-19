"""Internal typed contracts shared by training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MachineType = Literal["L", "M", "H"]


@dataclass(frozen=True)
class SensorMetrics:
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float


@dataclass(frozen=True)
class SensorPayload:
    sensor_id: str
    timestamp: str
    metrics: SensorMetrics
    machine_type: MachineType = "M"


@dataclass(frozen=True)
class PredictionResult:
    status: Literal["HEALTHY", "CRITICAL_WARNING"]
    failure_probability: float
    threshold: float
    machine_type: MachineType
    machine_type_assumed: bool
