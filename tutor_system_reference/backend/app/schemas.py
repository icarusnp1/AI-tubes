# backend/app/schemas.py
from __future__ import annotations

from typing import Dict, Optional, Literal, Any, List
from pydantic import BaseModel, Field


# ----------------------------
# Step event (Student -> Backend)
# ----------------------------
class StepEvent(BaseModel):
    student_id: str = Field(..., description="Opaque student identifier (no PII).")
    session_id: Optional[str] = Field(
        None,
        description="Optional session identifier (recommended for per-session GRU state).",
    )
    kc_id: str = Field(..., description="Knowledge component ID.")
    timestep: int = Field(..., ge=0, description="Step index within a student sequence.")

    # Engineered numeric features. Must align with seq_dataset_meta.json feature_names.
    features: Dict[str, float] = Field(
        default_factory=dict,
        description="Step-level numeric features (already engineered).",
    )

    raw_answer: Optional[str] = Field(None, description="Optional raw answer text (kept minimal).")

    # Optional correctness label if available:
    # - If provided, backend can update BKT online post-step.
    # - If omitted, BKT update can be skipped or handled elsewhere.
    is_correct: Optional[bool] = Field(
        None,
        description="Optional correctness label if available (for logging/BKT update).",
    )


# ----------------------------
# /infer_step response (Backend -> Student UI)
# MUST always include minimum contract fields.
# ----------------------------
class InferenceResponse(BaseModel):
    p_correct: float = Field(..., ge=0.0, le=1.0)

    # Optional latent (e.g., GRU hidden or other debugging vector).
    latent: Optional[List[float]] = None

    # Model metadata for UI/dashboard observability
    model_version: str = Field(..., description="Model version string.")
    model_status: Literal["heuristic", "gru_wired"] = Field(
        ...,
        description="Whether inference uses heuristic fallback or GRU wired model.",
    )

    # These are part of your feature_names and are required for dashboard/policy visibility.
    mastery_mean: float = Field(..., ge=0.0, le=1.0)
    mastery_min: float = Field(..., ge=0.0, le=1.0)

    warnings: List[str] = Field(default_factory=list)


# ----------------------------
# Policy engine schemas
# Don't break your current policy endpoint.
# ----------------------------
Action = Literal[
    "CONFIRM_AND_PROGRESS",
    "HINT_LIGHT",
    "HINT_STRONG",
    "REMEDIAL_MICRO_LESSON",
    "TARGETED_PRACTICE",
    "CHALLENGE",
    "SUGGEST_AR_VISUAL",
    "ESCALATE_TO_TEACHER",
]


class PolicyDecisionRequest(BaseModel):
    student_id: str
    session_id: Optional[str] = None
    kc_id: str
    timestep: int

    p_correct: float = Field(..., ge=0.0, le=1.0)
    bkt_mastery: float = Field(..., ge=0.0, le=1.0)
    time_sec: float = Field(..., ge=0.0)
    error_streak: int = Field(..., ge=0)
    hint_count: int = Field(..., ge=0)

    trend: float = Field(0.0, description="Change in p_correct vs short rolling baseline.")
    extra: Dict[str, Any] = Field(default_factory=dict)


class PolicyDecisionResponse(BaseModel):
    action: Action
    rationale: str
    ui_suggestions: Dict[str, Any] = Field(default_factory=dict)


# ----------------------------
# Content generation (Gemini)
# ----------------------------
class GenerateItemRequest(BaseModel):
    kc_id: str
    kc_description: str
    difficulty_target: int = Field(3, ge=1, le=5)
    misconception_tag: Optional[str] = None
    desired_format: Literal["short_answer", "mcq", "step_by_step"] = "step_by_step"
    constraints: Dict[str, Any] = Field(default_factory=dict)


class GeneratedItem(BaseModel):
    problem_statement: str
    correct_answer: str
    solution_steps: List[str]
    common_wrong_answers: List[dict] = Field(default_factory=list)
    kc_alignment_explanation: str


# ----------------------------
# Teacher dashboard
# ----------------------------
class TeacherDashboardResponse(BaseModel):
    generated_at_iso: str
    students: List[dict]
    kc_heatmap: List[dict]
    alerts: List[dict]


# ----------------------------
# Optional BKT update endpoint (debug/testing)
# ----------------------------
class BKTUpdateRequest(BaseModel):
    student_id: str
    kc_id: str
    y_correct: int = Field(..., ge=0, le=1)


class BKTUpdateResponse(BaseModel):
    kc_id: str
    mastery: float = Field(..., ge=0.0, le=1.0)
