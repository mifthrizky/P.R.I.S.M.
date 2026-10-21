"""Strict public API contracts."""

from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from prism.llm.schemas import MitigationResponse


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SensorMetricsRequest(StrictModel):
    air_temperature_k: float = Field(ge=150, le=1000)
    process_temperature_k: float = Field(ge=150, le=1500)
    rotational_speed_rpm: float = Field(ge=0, le=100_000)
    torque_nm: float = Field(ge=0, le=100_000)
    tool_wear_min: float = Field(ge=0, le=1_000_000)


class PredictionRequest(StrictModel):
    sensor_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    timestamp: AwareDatetime
    metrics: SensorMetricsRequest
    machine_type: Literal["L", "M", "H"] | None = None


class StageTimings(StrictModel):
    ml_ms: float = Field(ge=0)
    retrieval_ms: float | None = Field(default=None, ge=0)
    generation_ms: float | None = Field(default=None, ge=0)
    total_ms: float = Field(ge=0)


class PredictionResponse(StrictModel):
    request_id: str
    sensor_id: str
    timestamp: datetime
    status: Literal["HEALTHY", "CRITICAL_WARNING"]
    failure_probability: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    primary_failure: str | None
    failure_predictions: list[dict]
    machine_type: Literal["L", "M", "H"]
    machine_type_assumed: bool
    model_version: str
    mitigation: MitigationResponse | None
    timings: StageTimings
