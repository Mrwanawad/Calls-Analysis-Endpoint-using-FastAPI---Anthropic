"""Pydantic models for call transcript analysis input and output."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Input Models ──────────────────────────────────────────────────────────────


class CallTranscriptRequest(BaseModel):
    call_id: str
    agent_name: str
    call_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD")
    call_duration_seconds: int = Field(ge=0)
    department: str = Field(
        description='e.g. "Scheduling", "Onboarding", "Helpdesk", "Follow-Ups", "Records"'
    )
    transcript: str = Field(min_length=1)


class BatchAnalyzeRequest(BaseModel):
    transcripts: list[CallTranscriptRequest] = Field(min_length=1)


# ── Output Models ─────────────────────────────────────────────────────────────


class OverallAssessment(str, Enum):
    PASS = "pass"
    NEEDS_REVIEW = "needs_review"
    ESCALATE = "escalate"


class ComplianceFlagType(str, Enum):
    HIPAA_CONCERN = "hipaa_concern"
    MISINFORMATION = "misinformation"
    RUDENESS = "rudeness"
    PROTOCOL_VIOLATION = "protocol_violation"
    POSITIVE_INTERACTION = "positive_interaction"


class ComplianceFlagSeverity(str, Enum):
    CRITICAL = "critical"
    MODERATE = "moderate"
    MINOR = "minor"
    POSITIVE = "positive"


class ComplianceFlag(BaseModel):
    type: ComplianceFlagType
    severity: ComplianceFlagSeverity
    description: str = Field(description="1-2 sentence description of the issue or positive behavior")
    transcript_excerpt: str = Field(description="The relevant portion of the transcript")


class AgentPerformance(BaseModel):
    professionalism_score: float = Field(ge=0.0, le=1.0)
    accuracy_score: float = Field(ge=0.0, le=1.0)
    resolution_score: float = Field(ge=0.0, le=1.0)
    strengths: list[str] = Field(min_length=0, max_length=3)
    improvements: list[str] = Field(min_length=1, max_length=3)


class CallAnalysisResponse(BaseModel):
    call_id: str
    overall_assessment: OverallAssessment
    assessment_reasoning: str = Field(
        description="2-4 sentences explaining the overall assessment"
    )
    compliance_flags: list[ComplianceFlag]
    agent_performance: AgentPerformance
    escalation_required: bool
    escalation_reason: Optional[str] = None


class BatchAnalyzeResponse(BaseModel):
    results: list[CallAnalysisResponse]
