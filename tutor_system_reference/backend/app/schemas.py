from __future__ import annotations
from typing import Dict, Optional, Literal, Any
from pydantic import BaseModel, Field

class StepEvent(BaseModel):
    student_id: str = Field(..., description="Opaque student identifier (no PII).")
    kc_id: str = Field(..., description="Knowledge component ID.")
    timestep: int = Field(..., ge=0, description="Step index within a student sequence.")
    features: Dict[str, float] = Field(..., description="Step-level numeric features (already engineered).")
    raw_answer: Optional[str] = Field(None, description="Optional raw answer text (kept minimal).")
    is_correct: Optional[bool] = Field(None, description="Optional correctness label if available (for logging).")

class InferenceResponse(BaseModel):
    p_correct: float
    latent: Optional[list[float]] = None
    model_id: str
    warnings: list[str] = Field(default_factory=list)

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

class GenerateItemRequest(BaseModel):
    kc_id: str
    kc_description: str
    difficulty_target: int = Field(3, ge=1, le=5)
    misconception_tag: Optional[str] = None
    desired_format: Literal["short_answer","mcq","step_by_step"] = "step_by_step"
    constraints: Dict[str, Any] = Field(default_factory=dict)

class GeneratedItem(BaseModel):
    problem_statement: str
    correct_answer: str
    solution_steps: list[str]
    common_wrong_answers: list[dict] = Field(default_factory=list)
    kc_alignment_explanation: str

class TeacherDashboardResponse(BaseModel):
    generated_at_iso: str
    students: list[dict]
    kc_heatmap: list[dict]
    alerts: list[dict]
    
class BKTUpdateRequest(BaseModel):
    student_id: str
    kc_id: str
    y_correct: int = Field(..., ge=0, le=1)

class BKTUpdateResponse(BaseModel):
    kc_id: str
    mastery: float = Field(..., ge=0.0, le=1.0)

