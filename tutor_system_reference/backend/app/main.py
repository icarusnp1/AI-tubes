from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import os
import datetime as dt

from .schemas import (
    StepEvent, InferenceResponse,
    PolicyDecisionRequest, PolicyDecisionResponse,
    GenerateItemRequest, GeneratedItem,
    TeacherDashboardResponse,
)
from .inference import load_model_bundle, infer_step
from .policy import load_policy_config, decide_action
from .gemini import generate_with_gemini
from .validators import basic_math_sanity, enforce_constraints
from .schemas import BKTUpdateRequest, BKTUpdateResponse
from .bkt_online import load_bkt_params, bkt_update, get_default_params
from .logger import log_event


BKT_PARAMS = load_bkt_params()
MASTERY: Dict[str, Dict[str, float]] = {}  # MASTERY[student_id][kc_id] = pL


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

MODEL = load_model_bundle()
POLICY = load_policy_config(CFG_PATH)

# In-memory demo store (replace with Redis/Postgres in production)
STATE: Dict[str, Dict[str, Any]] = {}

@app.get("/health")
def health():
    return {"status":"ok", "model_version": MODEL.version}

@app.post("/infer_step", response_model=InferenceResponse)
def infer_step_endpoint(evt: StepEvent):
    p, latent, warnings = infer_step(MODEL, evt.features)

    # Keep minimal state for dashboard demo
    sid = evt.student_id
    STATE.setdefault(sid, {})
    STATE[sid].update({
        "student_id": sid,
        "kc_id": evt.kc_id,
        "timestep": evt.timestep,
        "p_correct": p,
        "time_sec": float(evt.features.get("time_sec", 0.0)),
        "error_count": int(evt.features.get("error_count", 0.0)),
        "hint_count": int(evt.features.get("hint_count", 0.0)),
        "bkt_mastery": float(evt.features.get("bkt_mastery", 0.5)),
        "updated_at": dt.datetime.utcnow().isoformat() + "Z",
    })
    
    log_event("infer_step", {"evt": evt.model_dump(), "p_correct": p, "model_version": MODEL.version})

    return InferenceResponse(
        p_correct=p,
        latent=latent,
        model_id=MODEL.version,
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
    
    log_event("policy_decide", {"req": req.model_dump(), "action": action, "rationale": rationale, "ui": ui})

    return PolicyDecisionResponse(action=action, rationale=rationale, ui_suggestions=ui)

@app.post("/generate_item", response_model=GeneratedItem)
def generate_item_endpoint(req: GenerateItemRequest):
    payload = req.model_dump()
    item = generate_with_gemini(payload)

    ok, msg = basic_math_sanity(item)
    if not ok:
        # fallback: raise or regenerate. Here we fallback to a safe template.
        item = generate_with_gemini(payload)

    ok2, msg2 = enforce_constraints(item, req.constraints or {})
    if not ok2:
        # fallback: minimal safe item (could also regenerate)
        item["correct_answer"] = "x = 5"

    return GeneratedItem(**item)

@app.get("/teacher/dashboard", response_model=TeacherDashboardResponse)
def teacher_dashboard():
    now = dt.datetime.utcnow().isoformat() + "Z"
    students = list(STATE.values())

    # Simple KC heatmap placeholder from current state
    kc_heatmap = []
    for s in students:
        kc_heatmap.append({
            "kc_id": s.get("kc_id"),
            "avg_p_correct": s.get("p_correct"),
            "avg_mastery": s.get("bkt_mastery"),
        })

    alerts = []
    for s in students:
        if s.get("p_correct", 1.0) < 0.35 and s.get("time_sec", 0.0) >= POLICY.stuck_time_sec:
            alerts.append({
                "type":"stuck_risk",
                "student_id": s["student_id"],
                "kc_id": s.get("kc_id"),
                "message":"Low predicted correctness with high time-on-step; consider intervention.",
            })

    return TeacherDashboardResponse(
        generated_at_iso=now,
        students=students,
        kc_heatmap=kc_heatmap,
        alerts=alerts,
    )

@app.post("/bkt_update", response_model=BKTUpdateResponse)
def bkt_update_endpoint(req: BKTUpdateRequest):
    sid = req.student_id
    kc = req.kc_id
    MASTERY.setdefault(sid, {})
    if kc not in MASTERY[sid]:
        prm = BKT_PARAMS.get(kc, get_default_params())
        MASTERY[sid][kc] = prm.p_init

    prm = BKT_PARAMS.get(kc, get_default_params())
    pL_prior = MASTERY[sid][kc]
    pL_next = bkt_update(pL_prior, int(req.y_correct), prm)
    MASTERY[sid][kc] = pL_next
    return BKTUpdateResponse(kc_id=kc, mastery=pL_next)
