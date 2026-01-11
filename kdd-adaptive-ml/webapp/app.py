import os
import time
import json
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px  # Pastikan install: pip install plotly
from joblib import load

# Import modul lokal Anda
from emotion_engine import estimate_emotion
from feature_extractor import compute_features, top_wrong_concepts
from mastery_tracker import init_mastery, update_mastery
from policy import decide
from question_sampler import generate_set   

# --- KONFIGURASI PATH ---
APP_DIR = os.path.dirname(__file__)
CONTENT_DIR = os.path.join(APP_DIR, "content")
MODEL_PATH = os.path.join(APP_DIR, "models", "struggle_tree.joblib")
DATA_DIR = os.path.join(APP_DIR, "data")
EVENTS_CSV = os.path.join(DATA_DIR, "events.csv")

MODEL_FEATURES = [
    "num_steps", "avg_step_time", "max_step_time", "total_time",
    "error_streak_max", "repeat_error_rate",
]

# --- HELPER FUNCTIONS ---

# 1. Tambahkan daftar lengkap semua kolom yang mungkin ada
LOG_COLUMNS = [
    "ts", "user_id", "event", "level", 
    "concept_tag", "template_id", "is_correct", "response_time", 
    "used_hint", "hint_steps_used", "used_explanation",
    "accuracy", "avg_step_time", "max_step_time", "total_time",
    "error_streak_max", "repeat_error_rate", "help_rate", "rapid_guess_rate",
    "struggle_pred", "emotion_state", "wrong_top2", 
    "policy_next_level", "policy_next_set_size", "policy_focus", "policy_hint_mode"
]

MODEL_FEATURES = [
    "num_steps", "avg_step_time", "max_step_time", "total_time",
    "error_streak_max", "repeat_error_rate",
]

def now_iso():
    return datetime.utcnow().isoformat()

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(APP_DIR, "models"), exist_ok=True)

# 2. Update fungsi log_event agar selalu menggunakan kolom yang konsisten
def log_event(row: dict):
    ensure_dirs()
    
    # Memaksa baris data memiliki semua kolom yang ada di LOG_COLUMNS
    # Jika data tidak ada di row, diisi None (kosong)
    standardized_row = {col: row.get(col, None) for col in LOG_COLUMNS}
    
    df = pd.DataFrame([standardized_row])
    
    # Tulis header hanya jika file belum ada
    if not os.path.exists(EVENTS_CSV):
        df.to_csv(EVENTS_CSV, index=False, columns=LOG_COLUMNS)
    else:
        # Append tanpa header
        df.to_csv(EVENTS_CSV, mode="a", header=False, index=False, columns=LOG_COLUMNS)

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
        "q_index": 0, "set_size": 5, "question_set": [], "answer_log": [],
        "q_start_ts": None, "hint_step_used": 0, "used_hint": False, "used_explanation": False
    })

def reset_quiz(question_set, set_size):
    st.session_state.quiz = {
        "q_index": 0, "set_size": set_size, "question_set": question_set, "answer_log": [],
        "q_start_ts": None, "hint_step_used": 0, "used_hint": False, "used_explanation": False
    }

def seed_key(user_id, level, mastery_streak, focus):
    return f"{user_id}|lvl={level}|ms={mastery_streak}|focus={','.join(focus)}"

def show_remedial(remedials, focus):
    if not focus: return
    st.markdown("### Remedial Fokus (Top-2 Konsep)")
    for tag in focus[:2]:
        card = remedials.get(tag)
        if not card: continue
        with st.expander(f"{card.get('title', tag)}  —  [{tag}]"):
            for b in card.get("bullets", []): st.write("• " + b)
            ex = card.get("example")
            if ex: st.info("Contoh: " + ex)

# --- TEACHER DASHBOARD LOGIC ---

def teacher_dashboard():
    st.header("🎓 Dashboard Monitoring Guru")
    
    # Simple Authentication
    pwd = st.sidebar.text_input("Password Guru", type="password")
    if pwd != "admin123":  # Ganti password sesuai keinginan
        st.warning("Masukkan password guru di sidebar untuk mengakses data.")
        return

    if not os.path.exists(EVENTS_CSV):
        st.info("Belum ada data siswa yang terekam.")
        return

    # Load Data
    try:
        df = pd.read_csv(EVENTS_CSV)
    except Exception as e:
        st.error(f"Gagal membaca database: {e}")
        return

    if df.empty:
        st.warning("Data kosong.")
        return

    # --- TAB 1: OVERVIEW KELAS ---
    tab1, tab2 = st.tabs(["Overview Kelas", "Detail Siswa"])
    
    with tab1:
        # Filter hanya event SET_RESULT untuk statistik umum
        df_results = df[df['event'] == 'SET_RESULT'].copy()
        
        if df_results.empty:
            st.warning("Belum ada siswa yang menyelesaikan satu set soal.")
        else:
            # Metrics Utama
            col1, col2, col3, col4 = st.columns(4)
            n_students = df['user_id'].nunique()
            avg_acc = df_results['accuracy'].mean() * 100
            total_struggle = df_results[df_results['struggle_pred'] == 'STRUGGLE'].shape[0]
            
            # Hitung emosi dominan kelas
            emo_counts = df_results['emotion_state'].value_counts()
            dom_emo = emo_counts.idxmax() if not emo_counts.empty else "-"

            col1.metric("Total Siswa", n_students)
            col2.metric("Rata-rata Akurasi", f"{avg_acc:.1f}%")
            col3.metric("Kejadian Struggle", total_struggle)
            col4.metric("Emosi Dominan", dom_emo)

            st.markdown("---")
            
            # Grafik Distribusi Emosi Kelas
            st.subheader("Distribusi Emosi Kelas")
            fig_emo = px.pie(df_results, names='emotion_state', title='Proporsi Kondisi Emosional Siswa')
            st.plotly_chart(fig_emo, use_container_width=True)

            # Tabel Ringkasan Siswa
            st.subheader("Peringkat & Status Terkini")
            # Ambil data terakhir untuk setiap user
            df_latest = df_results.sort_values('ts').groupby('user_id').tail(1)
            df_display = df_latest[['user_id', 'level', 'accuracy', 'struggle_pred', 'emotion_state', 'ts']]
            df_display.columns = ['Nama Siswa', 'Level Terakhir', 'Akurasi Terakhir', 'Prediksi Kesulitan', 'Emosi Terakhir', 'Waktu']
            st.dataframe(df_display, use_container_width=True)

    # --- TAB 2: DETAIL SISWA ---
    with tab2:
        students = df['user_id'].unique()
        selected_student = st.selectbox("Pilih Siswa:", students)
        
        if selected_student:
            # Filter data siswa
            student_df = df[df['user_id'] == selected_student]
            res_df = student_df[student_df['event'] == 'SET_RESULT']
            ans_df = student_df[student_df['event'] == 'SUBMIT_ANSWER']

            st.markdown(f"### Analisis: {selected_student}")

            # Grafik Perkembangan Akurasi
            if not res_df.empty:
                st.write("**Perkembangan Akurasi per Set Soal**")
                # Membuat index urutan set (1, 2, 3...)
                res_df = res_df.reset_index(drop=True)
                res_df['Set Ke'] = res_df.index + 1
                
                fig_line = px.line(res_df, x='Set Ke', y='accuracy', markers=True, 
                                   title="Tren Akurasi", range_y=[0, 1.1])
                st.plotly_chart(fig_line, use_container_width=True)

            # Analisis Kelemahan Konsep
            if not ans_df.empty:
                st.write("**Analisis Konsep (Berdasarkan Jawaban Salah)**")
                # Filter jawaban salah
                wrong_df = ans_df[ans_df['is_correct'] == 0]
                if not wrong_df.empty:
                    concept_counts = wrong_df['concept_tag'].value_counts().reset_index()
                    concept_counts.columns = ['Konsep', 'Jumlah Salah']
                    
                    fig_bar = px.bar(concept_counts, x='Jumlah Salah', y='Konsep', orientation='h', 
                                     title="Frekuensi Kesalahan per Konsep", color='Jumlah Salah')
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.success("Siswa ini belum pernah menjawab salah!")

            # Riwayat Aktivitas Log
            with st.expander("Lihat Log Aktivitas Mentah"):
                st.dataframe(student_df[['ts', 'event', 'level', 'concept_tag', 'is_correct']].sort_values('ts', ascending=False))

# --- STUDENT INTERFACE (ORIGINAL MAIN) ---

def restore_student_state(user_id, materials, remedials):
    """
    Membangun kembali status siswa (Level & Mastery) berdasarkan riwayat log di CSV.
    """
    # Default state untuk siswa baru
    state = {
        "level": 1,
        "mastery": init_mastery(list(remedials.keys())),
        "mastery_streak": 0,
        "history_found": False
    }

    if not os.path.exists(EVENTS_CSV):
        return state

    try:
        df = pd.read_csv(EVENTS_CSV)
    except:
        return state

    # Filter data hanya untuk user ini
    df_user = df[df['user_id'] == user_id].copy()
    
    if df_user.empty:
        return state

    # 1. Pulihkan Level Terakhir
    # Ambil level dari event terakhir yang tercatat
    last_level = df_user['level'].iloc[-1]
    state['level'] = int(last_level)
    state['history_found'] = True

    # 2. Pulihkan Mastery (Replay History)
    # Kita ulangi proses update_mastery dari jawaban-jawaban sebelumnya
    # Filter hanya event jawaban
    df_answers = df_user[df_user['event'] == 'SUBMIT_ANSWER']
    
    for _, row in df_answers.iterrows():
        # Pastikan data boolean/integer terbaca benar
        is_correct = bool(row['is_correct']) if pd.notna(row['is_correct']) else False
        used_hint = bool(row['used_hint']) if pd.notna(row['used_hint']) else False
        concept = row['concept_tag'] if pd.notna(row['concept_tag']) else ""

        # Update mastery seolah-olah kejadiannya baru terjadi
        state['mastery'] = update_mastery(state['mastery'], concept, is_correct, used_hint)

    # 3. Pulihkan Streak (Opsional, logika sederhana)
    # Cek event terakhir, jika SET_RESULT dan hasilnya OK, mungkin streak nambah
    # Untuk simplifikasi, kita reset streak ke 0 saat login ulang agar aman.
    state['mastery_streak'] = 0 

    return state

def student_interface():
    materials, remedials, model = load_resources()
    concept_list = list(remedials.keys())
    
    # Inisialisasi state dasar Streamlit jika belum ada
    if "page" not in st.session_state:
        init_state(concept_list)

    if st.session_state.page == "start":
        st.subheader("🎓 Masuk Kelas")
        
        # Input Nama
        user_input = st.text_input("Masukkan Nama/ID Anda", value=st.session_state.user_id)
        clean_user_id = user_input.strip()

        # Cek apakah siswa lama atau baru secara real-time
        is_returning = False
        detected_level = 1
        
        if clean_user_id:
            # Cek cepat ke CSV tanpa load full logic dulu
            if os.path.exists(EVENTS_CSV):
                try:
                    df_check = pd.read_csv(EVENTS_CSV, usecols=['user_id', 'level'])
                    user_rows = df_check[df_check['user_id'] == clean_user_id]
                    if not user_rows.empty:
                        is_returning = True
                        detected_level = int(user_rows.iloc[-1]['level'])
                except:
                    pass

        # Tampilan dinamis berdasarkan status siswa
        if is_returning:
            st.info(f"👋 Selamat datang kembali, **{clean_user_id}**! Sistem mendeteksi progress terakhir Anda di **Level {detected_level}**.")
            st.write("Klik tombol di bawah untuk melanjutkan pembelajaran.")
            # Level dikunci (disabled) ke level terakhir, atau user boleh nurunin (tapi tidak boleh loncat naik)
            level_selection = st.selectbox("Lanjut di Level:", options=[1, 2, 3], index=detected_level-1)
        else:
            st.write("Halo siswa baru! Silakan pilih level awal.")
            level_selection = st.selectbox("Pilih Level Awal", options=[1, 2, 3], index=0)
        
        if st.button(f"{'Lanjutkan' if is_returning else 'Mulai'} Belajar"):
            if not clean_user_id:
                st.warning("Nama harus diisi.")
                return

            st.session_state.user_id = clean_user_id
            
            # --- LOGIKA RESTORE ---
            if is_returning:
                # Panggil fungsi helper untuk restore mastery
                restored_data = restore_student_state(clean_user_id, materials, remedials)
                st.session_state.mastery = restored_data['mastery']
                # Gunakan level dari pilihan (user mungkin mau mengulang level sebelumnya)
                st.session_state.level = int(level_selection)
                log_event({"ts": now_iso(), "user_id": clean_user_id, "event": "SESSION_RESUME", "level": st.session_state.level})
                st.toast("Progress berhasil dipulihkan!", icon="✅")
            else:
                # Siswa Baru: Reset total
                st.session_state.level = int(level_selection)
                st.session_state.mastery = init_mastery(concept_list)
                log_event({"ts": now_iso(), "user_id": clean_user_id, "event": "SESSION_START", "level": st.session_state.level})
            
            # Reset state sesi (streak & pertanyaan dilihat di-reset tiap sesi baru)
            st.session_state.mastery_streak = 0
            st.session_state.focus_concepts = []
            st.session_state.hint_mode = "normal"
            st.session_state.seen_fingerprints = set()
            
            st.session_state.page = "material"
            st.rerun()

    # ... (SISA KODE 'elif st.session_state.page == "material":' KE BAWAH TETAP SAMA) ...
    elif st.session_state.page == "material":
        # ... copy paste kode lama bagian material ...
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
        # Tampilkan mastery rata-rata agar siswa tahu progress-nya
        avg_mastery = sum(st.session_state.mastery.values()) / len(st.session_state.mastery)
        st.write(f"Level: **{st.session_state.level}/3** | Rata-rata Penguasaan: **{avg_mastery*100:.0f}%**")

    # ... (Bagian QUIZ dan RESULT tetap sama persis seperti kode sebelumnya) ...
    elif st.session_state.page == "quiz":
        # Paste kode quiz yang lama di sini
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
            st.caption("Soal parametrik: angka random.")

    elif st.session_state.page == "result":
        # Paste kode result yang lama di sini
        answer_log = st.session_state.quiz["answer_log"]
        feats = compute_features(answer_log)
        struggle_pred = predict_struggle(model, feats)
        emo = estimate_emotion(feats)
        wrong_top2 = top_wrong_concepts(answer_log, k=2)

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
        st.write(f"Akurasi: **{feats.get('accuracy', 0.0):.2f}** | Struggle: **{struggle_pred}** | Emosi: **{emo.state}**")
        
        if decision.feedback_style == "supportive":
            st.info(decision.message)
        else:
            st.success(decision.message)

        st.markdown("**Fokus konsep berikutnya:** " + (", ".join([f"`{c}`" for c in decision.focus_concepts]) if decision.focus_concepts else "Tidak ada."))
        show_remedial(remedials, decision.focus_concepts)

        log_event({
            "ts": now_iso(),
            "user_id": st.session_state.user_id,
            "event": "SET_RESULT",
            "level": st.session_state.level,
            "accuracy": feats.get("accuracy"),
            "struggle_pred": struggle_pred,
            "emotion_state": emo.state,
            "wrong_top2": ",".join(wrong_top2),
            "policy_next_level": decision.next_level
        })

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Lanjut"):
                st.session_state.level = decision.next_level
                seed = seed_key(st.session_state.user_id, st.session_state.level, st.session_state.mastery_streak, st.session_state.focus_concepts) + "|next"
                qset, sf = generate_set(st.session_state.level, decision.next_set_size, st.session_state.focus_concepts, st.session_state.seen_fingerprints, seed=seed)
                st.session_state.seen_fingerprints = sf
                reset_quiz(qset, decision.next_set_size)
                st.session_state.page = "material"
                st.rerun()
        with c2:
            if st.button("Restart"):
                st.session_state.page = "start"
                st.rerun()

# --- MAIN APP ROUTER ---

def main():
    st.set_page_config(page_title="AI Adaptive Tutor", layout="wide")
    
    # Sidebar Navigation
    st.sidebar.title("Navigasi")
    app_mode = st.sidebar.selectbox("Pilih Mode", ["👨‍🎓 Area Siswa", "👩‍🏫 Dashboard Guru"])
    
    if app_mode == "👨‍🎓 Area Siswa":
        student_interface()
    elif app_mode == "👩‍🏫 Dashboard Guru":
        teacher_dashboard()

if __name__ == "__main__":
    main()