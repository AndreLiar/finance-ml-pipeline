"""
src/guardrails/structured_outputs.py — Pydantic models for all agent outputs.

Every agent decision passes through one of these models before being returned.
This enforces schema contracts and prevents free-text garbage from propagating.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ── Shared enums ───────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"

class Recommendation(str, Enum):
    APPROVE              = "APPROVE"
    CONDITIONAL_APPROVAL = "CONDITIONAL_APPROVAL"
    DECLINE              = "DECLINE"

class StageOutcome(str, Enum):
    SUCCESS  = "SUCCESS"
    FAILED   = "FAILED"
    RETRIED  = "RETRIED"
    SKIPPED  = "SKIPPED"
    DEGRADED = "DEGRADED"   # ran with fallback (e.g. RF-only instead of ensemble)


# ── Agent 2: Financial Advisor ─────────────────────────────────────────────────

class AdvisorAnswer(BaseModel):
    """Structured output from the Financial Advisor Agent."""

    answer: str = Field(
        description="The full natural-language answer to the user's question."
    )
    tools_used: list[str] = Field(
        default_factory=list,
        description="Names of data tools called to produce this answer.",
    )
    data_grounded: bool = Field(
        description="True if the answer references at least one concrete number from the data tools.",
    )
    language: str = Field(
        description="Language of the answer: 'français' or 'English'.",
    )
    confidence: str = Field(
        description="Agent's self-assessed confidence: HIGH / MEDIUM / LOW.",
    )

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_valid(cls, v: str) -> str:
        if v.upper() not in {"HIGH", "MEDIUM", "LOW"}:
            raise ValueError(f"confidence must be HIGH, MEDIUM, or LOW — got '{v}'")
        return v.upper()

    @field_validator("language")
    @classmethod
    def language_must_be_valid(cls, v: str) -> str:
        if v not in {"français", "English"}:
            raise ValueError(f"language must be 'français' or 'English' — got '{v}'")
        return v


# ── Agent 3: Anomaly Investigator ─────────────────────────────────────────────

class AnomalyVerdict(BaseModel):
    """Structured verdict for a single flagged transaction."""

    date: str          = Field(description="Transaction date (YYYY-MM-DD).")
    description: str   = Field(description="Transaction description (truncated to 60 chars).")
    amount_eur: float  = Field(description="Transaction amount in EUR (positive).")
    category: str      = Field(description="Assigned spending category.")
    suspicion: RiskLevel = Field(description="Suspicion level: LOW, MEDIUM, or HIGH.")
    reason: str        = Field(
        description="One-sentence explanation citing at least one concrete number (e.g. '3.4x above TRANSFER average')."
    )
    action: str        = Field(
        description="Recommended action: MONITOR / REVIEW / FLAG_FOR_AUDIT."
    )

    @field_validator("action")
    @classmethod
    def action_must_be_valid(cls, v: str) -> str:
        valid = {"MONITOR", "REVIEW", "FLAG_FOR_AUDIT"}
        if v.upper() not in valid:
            raise ValueError(f"action must be one of {valid} — got '{v}'")
        return v.upper()

    @field_validator("reason")
    @classmethod
    def reason_must_contain_number(cls, v: str) -> str:
        import re
        if not re.search(r'\d', v):
            raise ValueError("reason must cite at least one number — got: " + v)
        return v


class AnomalyInvestigationReport(BaseModel):
    """Full report produced by the Anomaly Investigator Agent."""

    total_flagged: int               = Field(description="Total number of flagged transactions.")
    investigated:  int               = Field(description="Number actually investigated by the agent.")
    verdicts:      list[AnomalyVerdict] = Field(description="Per-transaction verdicts.")
    summary:       str               = Field(description="2-3 sentence plain-language summary of findings.")
    high_count:    int               = Field(description="Count of HIGH suspicion verdicts.")
    review_count:  int               = Field(description="Count of verdicts requiring REVIEW or FLAG_FOR_AUDIT.")


# ── Agent 1: Pipeline Supervisor ──────────────────────────────────────────────

class StageResult(BaseModel):
    """Result of a single pipeline stage execution."""

    stage_id:   int         = Field(description="Stage number (1-10).")
    stage_name: str         = Field(description="Stage script name.")
    outcome:    StageOutcome
    retries:    int         = Field(default=0, description="Number of retry attempts made.")
    error:      Optional[str] = Field(default=None, description="Error message if failed.")
    note:       Optional[str] = Field(default=None, description="Agent reasoning note (e.g. why it retried).")

class PipelineReport(BaseModel):
    """Full pipeline run report from the Pipeline Supervisor Agent."""

    run_at:         str              = Field(description="ISO timestamp of the run.")
    total_stages:   int
    succeeded:      int
    failed:         int
    skipped:        int
    auto_recovered: int              = Field(description="Stages that succeeded after auto-retry.")
    degraded:       int              = Field(description="Stages that ran in fallback/degraded mode.")
    stages:         list[StageResult]
    incident_summary: str            = Field(
        description="Plain-language summary of what happened, including any auto-recoveries or failures."
    )
    overall_status: str              = Field(description="HEALTHY / DEGRADED / FAILED.")

    @field_validator("overall_status")
    @classmethod
    def status_must_be_valid(cls, v: str) -> str:
        if v.upper() not in {"HEALTHY", "DEGRADED", "FAILED"}:
            raise ValueError(f"overall_status must be HEALTHY, DEGRADED, or FAILED — got '{v}'")
        return v.upper()
