import json
import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from joblib import load

from emotion_engine import estimate_emotion
from policy import decide

APP_DIR = os.path.dirname(__file__)
CONTENT_DIR = os.path.join(APP_DIR, "content")
DATA_DIR = os.path.join(APP_DIR, "data")
MODEL_PATH = os.path.join(APP_DIR, "models", "struggle_tree.joblib")
EVENTS_CSV = os.path.join(DATA_DIR, "events.csv")

FEATURES_FOR_MODEL = [
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
def load_content():
    with open(os.path.join(CONTENT_DIR, "materials.json"), "r", encoding="utf-8") as f:
        materials = json.load(f)
    with open(os.path.join(CONTENT_DIR, "questions.json"), "r", encoding="utf-8") as f:
        questions = json.load(f)
    return materials, questions

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return load(MODEL_PATH)
    return None

def compute_features(answer_log: list):
    """
    answer_log item:
      {
        "is_correct": bool,
        "response_time": float,
        "used_hint": bool,
        "used_explanation": bool,
        "concept_tag": str
      }
    """
    num_steps = len(answer_log)
    if num_steps == 0:
        return {}

    times = [x["response_time"] for x in answer_log]
    corrects = [1 if x["is_correct"] else 0 for x in answer_log]
    accuracy = sum(corrects) / num_steps

    # error streak max
    streak = 0
    streak_max = 0
    for c in corrects:
        if c == 0:
            streak += 1
            streak_max = max(streak_max, streak)
        else:
            streak = 0

    # repeat error rate by concept_tag
    errors = [x for x in answer_log if not x["is_correct"]]
    if len(errors) > 0:
        counts = {}
        for e in errors:
            tag = e.get("concept_tag", "") or ""
            counts[tag] = counts.get(tag, 0) + 1
        repeated = sum(v for v in counts.values() if v >= 2)
        repeat_error_rate = repeated / len(errors)
    else:
        repeat_error_rate = 0.0

    # help rate
    help_used = sum(1 for x in answer_log if x["used_hint"] or x["used_explanation"])
    help_rate = help_used / num_steps

    # rapid guessing rate (wrong and too fast)
    rapid_wrong = sum(1 for x in answer_log if (not x["is_correct"] and x["response_time"] < 2.5))
    rapid_guess_rate = rapid_wrong / num_steps

    return {
        "num_steps": float(num_steps),
        "accuracy": float(accuracy),
        "avg_step_time": float(sum(times) / num_steps),
        "max_step_time": float(max(times)),
        "total_time": float(sum(times)),
        "error_streak_max": float(streak_max),
        "repeat_error_rate": float(repeat_error_rate),
        "help_rate": float(help_rate),
        "rapid_guess_rate": float(rapid_guess_rate),
    }

def predict_struggle(model, feats: dict):
    # Fallback jika model belum ada
    if model is None:
        # simple heuristic fallback
        if feats.get("error_streak_max", 0) >= 3 or feats.get("repeat_error_rate", 0) >= 0.3:
            return "STRUGGLE"
        return "OK"

    X = [[float(feats.get(f, 0.0)) for f in FEATURES_FOR_MODEL]]
    return str(model.predict(X)[0])

def init_state():
    if "page" not in st.session_state:
        st.session_state.page = "start"
    if "user_id" not in st.session_state:
        st.session_state.user_id = ""
    if "level" not in st.session_state:
        st.session_state.level = 1
    if "mastery_streak" not in st.session_state:
        st.session_state.mastery_streak = 0
    if "quiz" not in st.session_state:
        st.session_state.quiz = {
            "q_index": 0,
            "set_size": 5,
            "answer_log": [],
            "q_start_ts": None,
            "used_hint": False,
            "used_explanation": False,
        }
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

def reset_quiz(set_size: int):
    st.session_state.quiz = {
        "q_index": 0,
        "set_size": set_size,
        "answer_log": [],
        "q_start_ts": None,
        "used_hint": False,
        "used_explanation": False,
    }

def main():
    st.set_page_config(page_title="Adaptive Algebra Tutor (Prototype)", layout="centered")
    init_state()
    materials, questions = load_content()
    model = load_model()

    st.title("Adaptive Algebra Tutor (Prototype)")
    st.caption("Streamlit prototype — Emotion-aware (rules) + Struggle detector (Decision Tree) + Adaptation policy")

    if st.session_state.page == "start":
        st.subheader("Mulai Sesi")
        user_id = st.text_input("Nama/ID (untuk logging)", value=st.session_state.user_id)
        level = st.selectbox("Mulai dari level", options=[1, 2, 3], index=st.session_state.level - 1)
        if st.button("Mulai Belajar"):
            st.session_state.user_id = user_id.strip() or "guest"
            st.session_state.level = int(level)
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

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Mulai Latihan"):
                reset_quiz(set_size=5)
                st.session_state.page = "quiz"
                log_event({"ts": now_iso(), "user_id": st.session_state.user_id, "event": "VIEW_MATERIAL_DONE", "level": st.session_state.level})
                st.rerun()
        with col2:
            if st.button("Tambah Contoh"):
                st.info("\n".join(["Contoh tambahan:"] + [f"- {x}" for x in m["extra_examples"]]))
                log_event({"ts": now_iso(), "user_id": st.session_state.user_id, "event": "REQUEST_EXTRA_EXAMPLE", "level": st.session_state.level})

        st.markdown("---")
        st.write(f"Level: **{st.session_state.level}/3** | Mastery streak: **{st.session_state.mastery_streak}**")

    elif st.session_state.page == "quiz":
        lvl = str(st.session_state.level)
        qset = questions["levels"][lvl]

        q_index = st.session_state.quiz["q_index"]
        set_size = st.session_state.quiz["set_size"]
        if q_index >= set_size:
            # finish -> go to results
            st.session_state.page = "result"
            st.rerun()

        q = qset[q_index % len(qset)]
        st.subheader(f"Kuis Level {st.session_state.level} — Soal {q_index+1}/{set_size}")
        st.write(q["prompt"])

        if st.session_state.quiz["q_start_ts"] is None:
            st.session_state.quiz["q_start_ts"] = time.time()
            st.session_state.quiz["used_hint"] = False
            st.session_state.quiz["used_explanation"] = False
            log_event({"ts": now_iso(), "user_id": st.session_state.user_id, "event": "QUESTION_SHOWN", "level": st.session_state.level, "question_id": q["id"]})

        choice = st.radio("Pilih jawaban:", q["choices"], index=None)
        c1, c2, c3 = st.columns([1,1,2])

        with c1:
            if st.button("Hint"):
                st.session_state.quiz["used_hint"] = True
                st.info(q["hint"])
                log_event({"ts": now_iso(), "user_id": st.session_state.user_id, "event": "HINT_USED", "level": st.session_state.level, "question_id": q["id"]})

        with c2:
            if st.button("Submit"):
                if choice is None:
                    st.warning("Pilih salah satu jawaban dulu.")
                else:
                    rt = time.time() - st.session_state.quiz["q_start_ts"]
                    is_correct = (q["choices"].index(choice) == q["answer_index"])

                    if is_correct:
                        st.success("Benar.")
                    else:
                        st.error("Salah.")
                        st.session_state.quiz["used_explanation"] = True
                        st.write("Pembahasan:")
                        st.write(q["explanation"])
                        log_event({"ts": now_iso(), "user_id": st.session_state.user_id, "event": "VIEW_EXPLANATION", "level": st.session_state.level, "question_id": q["id"]})

                    st.session_state.quiz["answer_log"].append({
                        "question_id": q["id"],
                        "concept_tag": q.get("concept_tag", ""),
                        "is_correct": bool(is_correct),
                        "response_time": float(rt),
                        "used_hint": bool(st.session_state.quiz["used_hint"]),
                        "used_explanation": bool(st.session_state.quiz["used_explanation"]),
                    })

                    log_event({
                        "ts": now_iso(),
                        "user_id": st.session_state.user_id,
                        "event": "SUBMIT_ANSWER",
                        "level": st.session_state.level,
                        "question_id": q["id"],
                        "is_correct": int(is_correct),
                        "response_time": float(rt),
                        "used_hint": int(st.session_state.quiz["used_hint"]),
                        "used_explanation": int(st.session_state.quiz["used_explanation"]),
                        "concept_tag": q.get("concept_tag", ""),
                    })

                    # next question
                    st.session_state.quiz["q_index"] += 1
                    st.session_state.quiz["q_start_ts"] = None
                    st.rerun()

        with c3:
            st.caption("Catatan: Setelah submit, sistem otomatis lanjut ke soal berikutnya.")

    elif st.session_state.page == "result":
        answer_log = st.session_state.quiz["answer_log"]
        feats = compute_features(answer_log)

        struggle_pred = predict_struggle(model, feats)
        emo = estimate_emotion(feats)

        # mastery streak update (contoh: OK + accuracy>=0.8 dianggap konsisten)
        if struggle_pred == "OK" and feats.get("accuracy", 0) >= 0.8:
            st.session_state.mastery_streak += 1
        else:
            st.session_state.mastery_streak = 0

        decision = decide(
            level=st.session_state.level,
            struggle_pred=struggle_pred,
            emotion_state=emo.state,
            mastery_streak=st.session_state.mastery_streak,
        )

        st.subheader("Hasil & Adaptasi")
        st.write(f"Accuracy: **{feats['accuracy']:.2f}**")
        st.write(f"Avg time/soal: **{feats['avg_step_time']:.2f}s** | Error streak max: **{feats['error_streak_max']:.0f}**")
        st.write(f"Repeat error rate: **{feats['repeat_error_rate']:.2f}** | Help rate: **{feats['help_rate']:.2f}**")
        st.markdown("---")
        st.write(f"Struggle detector (ML): **{struggle_pred}**")
        st.write(f"Emotion engine (rules): **{emo.state}**")
        st.markdown("---")
        if decision.feedback_style == "supportive":
            st.info(decision.message)
        else:
            st.success(decision.message)

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
            "next_level": decision.next_level,
            "next_set_size": decision.next_set_size,
        })

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Lanjut (Next Step)"):
                st.session_state.level = decision.next_level
                reset_quiz(set_size=decision.next_set_size)
                st.session_state.page = "material"
                st.rerun()
        with col2:
            if st.button("Restart Sesi"):
                st.session_state.page = "start"
                st.session_state.mastery_streak = 0
                reset_quiz(set_size=5)
                st.rerun()

if __name__ == "__main__":
    main()
