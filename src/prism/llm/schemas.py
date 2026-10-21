"""Strict contracts for local mitigation generation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Citation(StrictModel):
    document_id: str = Field(min_length=1)
    page: int = Field(ge=1)
    title: str = Field(min_length=1)


class MitigationPlan(StrictModel):
    immediate_action: str = Field(min_length=1)
    required_ppe: list[str] = Field(min_length=1)
    safety_steps: list[str] = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)


class MitigationResponse(StrictModel):
    status: Literal["MITIGATION_READY", "MANUAL_REVIEW_REQUIRED"]
    predicted_failure: str = Field(min_length=1)
    confidence_score: float = Field(ge=0.0, le=1.0)
    mitigation_plan: MitigationPlan | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> "MitigationResponse":
        if self.status == "MITIGATION_READY":
            if self.mitigation_plan is None:
                raise ValueError("MITIGATION_READY requires a mitigation_plan")
            if self.reason is not None:
                raise ValueError("MITIGATION_READY must not contain a fallback reason")
        else:
            if self.mitigation_plan is not None:
                raise ValueError("MANUAL_REVIEW_REQUIRED must not contain a mitigation_plan")
            if not self.reason:
                raise ValueError("MANUAL_REVIEW_REQUIRED requires a reason")
        return self
