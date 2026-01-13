# backend/app/main.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import os
import datetime as dt

from . import inference  # IMPORTANT: module import (live ADAPTER_READY)
from .schemas import (
    StepEvent, InferenceResponse,
    PolicyDecisionRequest, PolicyDecisionResponse,
    GenerateItemRequest, GeneratedItem,
    TeacherDashboardResponse,
    BKTUpdateRequest, BKTUpdateResponse,
)
from .policy import load_policy_config, decide_action
from .gemini import generate_with_gemini
from .validators import basic_math_sanity, enforce_constraints
from .bkt_online import get_mastery, update_mastery
from .logger import log_event


APP_ROOT = os.path.dirname(os.path.dirname(__file__))
CFG_PATH = os.getenv("POLICY_CONFIG_PATH", os.path.join(APP_ROOT, "config", "policy_config.yaml"))

app = FastAPI(title="TOPCER Adaptive Tutor API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = inference.load_model_bundle()
POLICY = load_policy_config(CFG_PATH)

# In-memory demo store (replace with persistent DB later)
STATE: Dict[str, Dict[str, Any]] = {}


@app.get("/health")
def health():
    model_status = "gru_wired" if (inference.ADAPTER_READY and MODEL.model is not None) else "heuristic"
    return {
        "status": "ok",
        "model_version": MODEL.version,
        "model_status": model_status,
        "torch_available": True,
        "model_loaded": (MODEL.model is not None),
        "adapter_ready": inference.ADAPTER_READY,
    }


@app.post("/infer_step", response_model=InferenceResponse)
def infer_step_endpoint(evt: StepEvent):
    sid = evt.student_id
    kc_raw = evt.kc_id
    session_id = evt.session_id

    # Resolve KC index safely (handles "KC_12", etc.)
    kc_index = inference.resolve_kc_index(MODEL.meta, kc_raw)

    # Prepare features with BKT PRIOR (no leakage)
    features = dict(evt.features or {})
    if "mastery_mean" not in features or "mastery_min" not in features:
        prior = get_mastery(sid, str(kc_raw))
        features.setdefault("mastery_mean", float(prior))
        features.setdefault("mastery_min", float(prior))

    # Run inference (GRU or heuristic)
    p, latent, warnings = inference.infer_step(
        MODEL,
        features,
        student_id=sid,
        session_id=session_id,
        kc_id=kc_index,
    )

    model_status = "gru_wired" if (inference.ADAPTER_READY and MODEL.model is not None) else "heuristic"

    # Update BKT AFTER observing y_correct if present
    mastery_post = float(features.get("mastery_mean", 0.0))
    if "y_correct" in features:
        try:
            mastery_post = update_mastery(sid, str(kc_raw), int(features["y_correct"]))
        except Exception:
            warnings.append("BKT_UPDATE_FAILED")

    # Keep minimal state for dashboard demo
    STATE.setdefault(sid, {})
    STATE[sid].update({
        "student_id": sid,
        "session_id": session_id,
        "kc_id": kc_raw,
        "kc_index": kc_index,
        "timestep": evt.timestep,
        "p_correct": p,
        "duration": float(features.get("duration", 0.0)),
        "incorrects": int(features.get("incorrects", 0)),
        "hints": int(features.get("hints", 0)),
        "mastery_mean": float(mastery_post),
        "mastery_min": float(mastery_post),
        "model_status": model_status,
        "model_version": MODEL.version,
        "updated_at": dt.datetime.utcnow().isoformat() + "Z",
    })

    log_event(
        "infer_step",
        {
            "evt": evt.model_dump(),
            "kc_index": kc_index,
            "p_correct": p,
            "model_version": MODEL.version,
            "model_status": model_status,
            "mastery_mean": mastery_post,
            "mastery_min": mastery_post,
            "warnings": warnings,
        },
        student_id=sid,
        session_id=session_id,
        kc_id=str(kc_raw),
    )

    return InferenceResponse(
        p_correct=p,
        latent=latent,
        model_version=MODEL.version,
        model_status=model_status,
        mastery_mean=float(mastery_post),
        mastery_min=float(mastery_post),
        warnings=warnings,
    )


@app.post("/policy_decide", response_model=PolicyDecisionResponse)
def policy_decide_endpoint(req: PolicyDecisionRequest):
    action, rationale, ui = decide_action(
        cfg=POLICY,
        p_correct=req.p_correct,
        bkt_mastery=req.bkt_mastery,
        time_sec=req.time_sec,
        error_streak=req.error_streak,
        hint_count=req.hint_count,
        trend=req.trend,
        misconception_tag=req.extra.get("misconception_tag"),
    )

    log_event(
        "policy_decide",
        {"req": req.model_dump(), "action": action, "rationale": rationale, "ui": ui},
        student_id=req.student_id,
        session_id=req.session_id,
        kc_id=str(req.kc_id),
    )

    return PolicyDecisionResponse(action=action, rationale=rationale, ui_suggestions=ui)


@app.post("/generate_item", response_model=GeneratedItem)
def generate_item_endpoint(req: GenerateItemRequest):
    payload = req.model_dump()
    item = generate_with_gemini(payload)

    ok, _ = basic_math_sanity(item)
    if not ok:
        item = generate_with_gemini(payload)

    ok2, _ = enforce_constraints(item, req.constraints or {})
    if not ok2:
        item["correct_answer"] = item.get("correct_answer") or "x = 5"

    return GeneratedItem(**item)


@app.get("/teacher/dashboard", response_model=TeacherDashboardResponse)
def teacher_dashboard():
    now = dt.datetime.utcnow().isoformat() + "Z"
    students = list(STATE.values())

    kc_heatmap = []
    for s in students:
        kc_heatmap.append({
            "kc_id": s.get("kc_id"),
            "avg_p_correct": s.get("p_correct"),
            "avg_mastery": s.get("mastery_mean"),
        })

    alerts = []
    stuck_time_sec = getattr(POLICY, "stuck_time_sec", 40)
    for s in students:
        if s.get("p_correct", 1.0) < 0.35 and s.get("duration", 0.0) >= stuck_time_sec:
            alerts.append({
                "type": "stuck_risk",
                "student_id": s["student_id"],
                "kc_id": s.get("kc_id"),
                "message": "Low predicted correctness with high time-on-step; consider intervention.",
            })

    return TeacherDashboardResponse(
        generated_at_iso=now,
        students=students,
        kc_heatmap=kc_heatmap,
        alerts=alerts,
    )


@app.post("/bkt_update", response_model=BKTUpdateResponse)
def bkt_update_endpoint(req: BKTUpdateRequest):
    p_next = update_mastery(req.student_id, str(req.kc_id), int(req.y_correct))
    return BKTUpdateResponse(kc_id=req.kc_id, mastery=p_next)
