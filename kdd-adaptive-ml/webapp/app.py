import os
import time
import json
from datetime import datetime

import pandas as pd
import streamlit as st
from joblib import load

from emotion_engine import estimate_emotion
from feature_extractor import compute_features, top_wrong_concepts
from mastery_tracker import init_mastery, update_mastery
from policy import decide
from question_sampler import generate_set

APP_DIR = os.path.dirname(__file__)
CONTENT_DIR = os.path.join(APP_DIR, "content")
MODEL_PATH = os.path.join(APP_DIR, "models", "struggle_tree.joblib")
DATA_DIR = os.path.join(APP_DIR, "data")
EVENTS_CSV = os.path.join(DATA_DIR, "events.csv")

MODEL_FEATURES = [
    "num_steps",
    "avg_step_time",
    "max_step_time",
    "total_time",
    "error_streak_max",
    "repeat_error_rate",
]

def now_iso():
    return datetime.utcnow().isoformat()

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(APP_DIR, "models"), exist_ok=True)

def log_event(row: dict):
    ensure_dirs()
    df = pd.DataFrame([row])
    if not os.path.exists(EVENTS_CSV):
        df.to_csv(EVENTS_CSV, index=False)
    else:
        df.to_csv(EVENTS_CSV, mode="a", header=False, index=False)

@st.cache_resource
def load_resources():
    with open(os.path.join(CONTENT_DIR, "materials.json"), "r", encoding="utf-8") as f:
        materials = json.load(f)
    with open(os.path.join(CONTENT_DIR, "remedials.json"), "r", encoding="utf-8") as f:
        remedials = json.load(f)
    model = load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    return materials, remedials, model

def predict_struggle(model, feats: dict) -> str:
    if model is None:
        # fallback heuristic (aman untuk demo)
        if feats.get("error_streak_max", 0) >= 3 or feats.get("repeat_error_rate", 0) >= 0.3:
            return "STRUGGLE"
        return "OK"
    X = [[float(feats.get(f, 0.0)) for f in MODEL_FEATURES]]
    return str(model.predict(X)[0])

def init_state(concept_list):
    ss = st.session_state
    ss.setdefault("page", "start")
    ss.setdefault("user_id", "")
    ss.setdefault("level", 1)
    ss.setdefault("mastery_streak", 0)
    ss.setdefault("focus_concepts", [])
    ss.setdefault("hint_mode", "normal")
    ss.setdefault("seen_fingerprints", set())
    ss.setdefault("mastery", init_mastery(concept_list))
    ss.setdefault("quiz", {
        "q_index": 0,
        "set_size": 5,
        "question_set": [],
        "answer_log": [],
        "q_start_ts": None,
        "hint_step_used": 0,
        "used_hint": False,
        "used_explanation": False
    })

def reset_quiz(question_set, set_size):
    st.session_state.quiz = {
        "q_index": 0,
        "set_size": set_size,
        "question_set": question_set,
        "answer_log": [],
        "q_start_ts": None,
        "hint_step_used": 0,
        "used_hint": False,
        "used_explanation": False
    }

def seed_key(user_id, level, mastery_streak, focus):
    return f"{user_id}|lvl={level}|ms={mastery_streak}|focus={','.join(focus)}"

def show_remedial(remedials, focus):
    if not focus:
        return
    st.markdown("### Remedial Fokus (Top-2 Konsep)")
    for tag in focus[:2]:
        card = remedials.get(tag)
        if not card:
            continue
        with st.expander(f"{card.get('title', tag)}  —  [{tag}]"):
            for b in card.get("bullets", []):
                st.write("• " + b)
            ex = card.get("example")
            if ex:
                st.info("Contoh: " + ex)

def main():
    st.set_page_config(page_title="Adaptive Algebra Tutor (Prototype)", layout="centered")
    materials, remedials, model = load_resources()

    concept_list = list(remedials.keys())
    init_state(concept_list)

    st.title("Adaptive Algebra Tutor (Prototype)")
    st.caption("Template soal per level + angka random + mastery per konsep + top-2 remedial + emotion + struggle")

    with st.sidebar:
        st.header("Status (Debug)")
        st.write("User:", st.session_state.user_id or "-")
        st.write("Level:", st.session_state.level)
        st.write("Mastery streak:", st.session_state.mastery_streak)
        st.write("Focus:", st.session_state.focus_concepts)
        st.write("Hint mode:", st.session_state.hint_mode)
        st.markdown("**Mastery (terendah):**")
        low = sorted(st.session_state.mastery.items(), key=lambda kv: kv[1])[:5]
        for k, v in low:
            st.write(f"- {k}: {v:.2f}")
        if st.button("Reset Seen Questions"):
            st.session_state.seen_fingerprints = set()
            st.success("seen_fingerprints cleared.")

    if st.session_state.page == "start":
        st.subheader("Mulai Sesi")
        user_id = st.text_input("Nama/ID (untuk logging)", value=st.session_state.user_id)
        level = st.selectbox("Mulai dari level", options=[1, 2, 3], index=st.session_state.level - 1)
        if st.button("Mulai"):
            st.session_state.user_id = (user_id.strip() or "guest")
            st.session_state.level = int(level)
            st.session_state.mastery_streak = 0
            st.session_state.focus_concepts = []
            st.session_state.hint_mode = "normal"
            st.session_state.seen_fingerprints = set()
            st.session_state.mastery = init_mastery(concept_list)
            st.session_state.page = "material"
            log_event({"ts": now_iso(), "user_id": st.session_state.user_id, "event": "SESSION_START", "level": st.session_state.level})
            st.rerun()

    elif st.session_state.page == "material":
        lvl = str(st.session_state.level)
        m = materials["levels"][lvl]
        st.subheader(m["title"])
        st.write(m["summary"])
        st.markdown("---")
        for p in m["content"]:
            st.write("• " + p)

        show_remedial(remedials, st.session_state.focus_concepts)

        cols = st.columns(3)
        with cols[0]:
            if st.button("Mulai Latihan"):
                seed = seed_key(st.session_state.user_id, st.session_state.level, st.session_state.mastery_streak, st.session_state.focus_concepts)
                qset, sf = generate_set(st.session_state.level, 5, st.session_state.focus_concepts, st.session_state.seen_fingerprints, seed=seed)
                st.session_state.seen_fingerprints = sf
                reset_quiz(qset, 5)
                st.session_state.page = "quiz"
                log_event({"ts": now_iso(), "user_id": st.session_state.user_id, "event": "START_SET", "level": st.session_state.level, "focus": ",".join(st.session_state.focus_concepts)})
                st.rerun()
        with cols[1]:
            if st.button("Tambah Contoh"):
                st.info("\n".join(["Contoh tambahan:"] + [f"- {x}" for x in m["extra_examples"]]))
                log_event({"ts": now_iso(), "user_id": st.session_state.user_id, "event": "REQUEST_EXTRA_EXAMPLE", "level": st.session_state.level})
        with cols[2]:
            if st.button("Latihan Fokus Konsep"):
                seed = seed_key(st.session_state.user_id, st.session_state.level, st.session_state.mastery_streak, st.session_state.focus_concepts) + "|focusbtn"
                qset, sf = generate_set(st.session_state.level, 5, st.session_state.focus_concepts, st.session_state.seen_fingerprints, seed=seed)
                st.session_state.seen_fingerprints = sf
                reset_quiz(qset, 5)
                st.session_state.page = "quiz"
                log_event({"ts": now_iso(), "user_id": st.session_state.user_id, "event": "START_FOCUS_SET", "level": st.session_state.level, "focus": ",".join(st.session_state.focus_concepts)})
                st.rerun()

        st.markdown("---")
        st.write(f"Level: **{st.session_state.level}/3** | Mastery streak: **{st.session_state.mastery_streak}**")

    elif st.session_state.page == "quiz":
        quiz = st.session_state.quiz
        qi = quiz["q_index"]
        set_size = quiz["set_size"]
        qset = quiz["question_set"]

        if qi >= set_size:
            st.session_state.page = "result"
            st.rerun()

        q = qset[qi]
        st.subheader(f"Kuis Level {st.session_state.level} — Soal {qi + 1}/{set_size}")
        st.write(q["prompt"])

        if quiz["q_start_ts"] is None:
            quiz["q_start_ts"] = time.time()
            quiz["hint_step_used"] = 0
            quiz["used_hint"] = False
            quiz["used_explanation"] = False

        choice = st.radio("Pilih jawaban:", q["choices"], index=None)

        c1, c2, c3 = st.columns([1, 1, 2])

        with c1:
            if st.button("Hint"):
                steps = q.get("hint_steps", [])
                if quiz["hint_step_used"] < len(steps):
                    st.info(steps[quiz["hint_step_used"]])
                    quiz["hint_step_used"] += 1
                    quiz["used_hint"] = True
                    log_event({"ts": now_iso(), "user_id": st.session_state.user_id, "event": "HINT_USED", "level": st.session_state.level, "concept_tag": q.get("concept_tag", ""), "hint_step": quiz["hint_step_used"]})
                else:
                    st.info("Hint sudah maksimal.")

        with c2:
            if st.button("Submit"):
                if choice is None:
                    st.warning("Pilih jawaban dulu.")
                else:
                    rt = time.time() - quiz["q_start_ts"]
                    is_correct = (choice == q["answer"])

                    if is_correct:
                        st.success("Benar.")
                    else:
                        st.error("Salah.")
                        quiz["used_explanation"] = True
                        st.write("Pembahasan:")
                        st.write(q.get("explanation", "-"))

                    # Update mastery (langsung setelah setiap soal)
                    st.session_state.mastery = update_mastery(
                        st.session_state.mastery,
                        q.get("concept_tag", ""),
                        is_correct=is_correct,
                        used_hint=quiz["used_hint"]
                    )

                    quiz["answer_log"].append({
                        "concept_tag": q.get("concept_tag", ""),
                        "template_id": q.get("template_id", ""),
                        "is_correct": bool(is_correct),
                        "response_time": float(rt),
                        "used_hint": bool(quiz["used_hint"]),
                        "used_explanation": bool(quiz["used_explanation"]),
                    })

                    log_event({
                        "ts": now_iso(),
                        "user_id": st.session_state.user_id,
                        "event": "SUBMIT_ANSWER",
                        "level": st.session_state.level,
                        "concept_tag": q.get("concept_tag", ""),
                        "template_id": q.get("template_id", ""),
                        "is_correct": int(is_correct),
                        "response_time": float(rt),
                        "used_hint": int(quiz["used_hint"]),
                        "hint_steps_used": int(quiz["hint_step_used"]),
                        "used_explanation": int(quiz["used_explanation"])
                    })

                    # Early warning setelah 3 soal
                    if len(quiz["answer_log"]) >= 3 and st.session_state.hint_mode != "aggressive":
                        feats_mid = compute_features(quiz["answer_log"])
                        struggle_mid = predict_struggle(model, feats_mid)
                        if struggle_mid == "STRUGGLE":
                            st.session_state.hint_mode = "aggressive"
                            wrong_mid = top_wrong_concepts(quiz["answer_log"], k=2)
                            if wrong_mid:
                                st.session_state.focus_concepts = wrong_mid
                            st.warning("Early warning: terdeteksi STRUGGLE. Mode bantuan ditingkatkan.")
                            log_event({"ts": now_iso(), "user_id": st.session_state.user_id, "event": "EARLY_WARNING_STRUGGLE", "level": st.session_state.level})

                    quiz["q_index"] += 1
                    quiz["q_start_ts"] = None
                    st.rerun()

        with c3:
            st.caption("Soal parametrik: angka random, hint berbasis langkah (template) sehingga konsisten.")

    elif st.session_state.page == "result":
        answer_log = st.session_state.quiz["answer_log"]
        feats = compute_features(answer_log)
        struggle_pred = predict_struggle(model, feats)
        emo = estimate_emotion(feats)
        wrong_top2 = top_wrong_concepts(answer_log, k=2)

        # streak naik jika OK dan akurasi tinggi
        if struggle_pred == "OK" and feats.get("accuracy", 0.0) >= 0.8:
            st.session_state.mastery_streak += 1
        else:
            st.session_state.mastery_streak = 0

        decision = decide(
            level=st.session_state.level,
            struggle_pred=struggle_pred,
            emotion_state=emo.state,
            mastery_streak=st.session_state.mastery_streak,
            mastery=st.session_state.mastery,
            wrong_top2=wrong_top2
        )

        st.session_state.focus_concepts = decision.focus_concepts
        st.session_state.hint_mode = decision.hint_mode

        st.subheader("Hasil & Adaptasi")
        st.write(f"Akurasi set: **{feats.get('accuracy', 0.0):.2f}**")
        st.write(f"Avg time/soal: **{feats.get('avg_step_time', 0.0):.2f}s** | Error streak max: **{feats.get('error_streak_max', 0.0):.0f}**")
        st.write(f"Repeat error rate: **{feats.get('repeat_error_rate', 0.0):.2f}** | Help rate: **{feats.get('help_rate', 0.0):.2f}**")
        st.markdown("---")
        st.write(f"Struggle: **{struggle_pred}** | Emotion: **{emo.state}**")

        if decision.feedback_style == "supportive":
            st.info(decision.message)
        else:
            st.success(decision.message)

        st.markdown("**Fokus konsep berikutnya (Top-2):** " + (", ".join([f"`{c}`" for c in decision.focus_concepts]) if decision.focus_concepts else "Tidak ada."))
        show_remedial(remedials, decision.focus_concepts)

        log_event({
            "ts": now_iso(),
            "user_id": st.session_state.user_id,
            "event": "SET_RESULT",
            "level": st.session_state.level,
            "accuracy": feats.get("accuracy"),
            "avg_step_time": feats.get("avg_step_time"),
            "max_step_time": feats.get("max_step_time"),
            "total_time": feats.get("total_time"),
            "error_streak_max": feats.get("error_streak_max"),
            "repeat_error_rate": feats.get("repeat_error_rate"),
            "help_rate": feats.get("help_rate"),
            "rapid_guess_rate": feats.get("rapid_guess_rate"),
            "struggle_pred": struggle_pred,
            "emotion_state": emo.state,
            "wrong_top2": ",".join(wrong_top2),
            "policy_next_level": decision.next_level,
            "policy_next_set_size": decision.next_set_size,
            "policy_focus": ",".join(decision.focus_concepts),
            "policy_hint_mode": decision.hint_mode
        })

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Lanjut"):
                st.session_state.level = decision.next_level
                seed = seed_key(st.session_state.user_id, st.session_state.level, st.session_state.mastery_streak, st.session_state.focus_concepts) + "|next"
                qset, sf = generate_set(
                    st.session_state.level,
                    decision.next_set_size,
                    st.session_state.focus_concepts,
                    st.session_state.seen_fingerprints,
                    seed=seed
                )
                st.session_state.seen_fingerprints = sf
                reset_quiz(qset, decision.next_set_size)
                st.session_state.page = "material"
                st.rerun()
        with c2:
            if st.button("Restart"):
                st.session_state.page = "start"
                st.rerun()

if __name__ == "__main__":
    main()
