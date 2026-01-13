from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import yaml
import os

@dataclass
class PolicyConfig:
    remedial_threshold: float
    hint_threshold: float
    challenge_threshold: float
    ar_suggest_threshold: float
    stuck_time_sec: float
    max_hints_per_kc: int
    escalate_error_streak: int

def load_policy_config(path: str) -> PolicyConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = cfg["policy"]
    return PolicyConfig(
        remedial_threshold=float(p["remedial_threshold"]),
        hint_threshold=float(p["hint_threshold"]),
        challenge_threshold=float(p["challenge_threshold"]),
        ar_suggest_threshold=float(p["ar_suggest_threshold"]),
        stuck_time_sec=float(p["stuck_time_sec"]),
        max_hints_per_kc=int(p["max_hints_per_kc"]),
        escalate_error_streak=int(p["escalate_error_streak"]),
    )

def decide_action(
    *,
    cfg: PolicyConfig,
    p_correct: float,
    bkt_mastery: float,
    time_sec: float,
    error_streak: int,
    hint_count: int,
    trend: float,
    misconception_tag: str | None = None,
) -> Tuple[str, str, Dict[str, Any]]:
    # Governance first: escalate if repeatedly stuck
    if error_streak >= cfg.escalate_error_streak and time_sec >= cfg.stuck_time_sec:
        return (
            "ESCALATE_TO_TEACHER",
            "Student is stuck with repeated errors and high time-on-step; flag teacher assistance.",
            {"alert": "stuck_escalation"},
        )

    # AR suggestion is opt-in; we only suggest when a likely visual misconception exists
    # and the model confidence is low enough to justify a different modality.
    if (p_correct <= cfg.ar_suggest_threshold) and (time_sec >= cfg.stuck_time_sec or error_streak >= 2):
        return (
            "SUGGEST_AR_VISUAL",
            "Low predicted correctness and signs of being stuck; suggest Visual (AR) mode as an optional support.",
            {"show_ar_button": True, "fallback_mode": "VISUAL_2D"},
        )

    # Remedial vs hint decisions
    if p_correct < cfg.remedial_threshold:
        return (
            "REMEDIAL_MICRO_LESSON",
            "Predicted correctness is low; provide a short remedial micro-lesson focused on the current KC.",
            {"micro_lesson": True},
        )

    if p_correct < cfg.hint_threshold:
        # Hint budget control
        if hint_count >= cfg.max_hints_per_kc:
            return (
                "TARGETED_PRACTICE",
                "Hint budget reached; switch to a guided practice item to re-anchor the concept.",
                {"generate_item": True, "format": "step_by_step"},
            )
        strength = "HINT_STRONG" if error_streak >= 1 else "HINT_LIGHT"
        return (
            strength,
            "Moderate predicted correctness; provide scaffolded hint to unlock the next step.",
            {"hint_strength": strength},
        )

    # Challenge / accelerate
    if p_correct >= cfg.challenge_threshold and trend >= 0.0:
        return (
            "CHALLENGE",
            "High predicted correctness; offer a challenge item to promote near-transfer and engagement.",
            {"generate_item": True, "difficulty_bump": 1},
        )

    return (
        "CONFIRM_AND_PROGRESS",
        "Predicted correctness is sufficient; confirm and move to the next step.",
        {"progress": True},
    )
