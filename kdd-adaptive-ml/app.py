import time
import numpy as np
import pandas as pd
import random  # <--- TAMBAHAN: Untuk mengacak soal
from joblib import load

# ===============================
# MODEL & FEATURES
# ===============================
FEATURES = [
    "accuracy",
    "num_steps",
    "avg_step_time",
    "max_step_time",
    "total_time",
    "error_streak_max",
    "repeat_error_rate"
]

model = load("models/struggle_tree.joblib")

# ===============================
# GLOBAL STATE
# ===============================
attempts = 0
corrects = 0
error_streak = 0
step_times = []
current_level = "MED"
start_time = time.time()

high_streak = 0
MAX_HIGH_STREAK = 3

# ===============================
# CONTENT ENGINE (UPDATED)
# ===============================
materials = {
    "LOW": "Materi Dasar: Persamaan linear sederhana (x +/- a = b).",
    "MED": "Materi Menengah: Persamaan dengan koefisien (ax + b = c).",
    "HIGH": "Materi Lanjutan: Persamaan kompleks (ax - b = c)."
}

# PERUBAHAN DI SINI: Sekarang setiap level punya BANYAK soal
questions = {
    "LOW": [
        ("x + 2 = 5", 3),
        ("x + 5 = 10", 5),
        ("x - 3 = 4", 7),
        ("x + 1 = 9", 8),
        ("x - 2 = 1", 3)
    ],
    "MED": [
        ("2x + 3 = 11", 4),
        ("3x + 1 = 10", 3),
        ("2x - 4 = 6", 5),
        ("4x + 2 = 14", 3),
        ("2x + 5 = 15", 5)
    ],
    "HIGH": [
        ("3x - 5 = 10", 5),
        ("5x + 2 = 27", 5),
        ("4x - 7 = 9", 4),
        ("6x - 10 = 20", 5),
        ("3x + 12 = 27", 5)
    ]
}

# ===============================
# FEATURE EXTRACTOR
# ===============================
def extract_features():
    accuracy = corrects / attempts if attempts > 0 else 0
    avg_step_time = np.mean(step_times) if step_times else 0
    max_step_time = np.max(step_times) if step_times else 0
    total_time = np.sum(step_times)
    num_steps = attempts
    error_streak_max = error_streak
    repeat_error_rate = error_streak / attempts if attempts > 0 else 0

    return {
        "accuracy": accuracy,
        "avg_step_time": avg_step_time,
        "max_step_time": max_step_time,
        "total_time": total_time,
        "num_steps": num_steps,
        "error_streak_max": error_streak_max,
        "repeat_error_rate": repeat_error_rate
    }

# ===============================
# EMOTION ENGINE
# ===============================
def infer_emotion(f):
    if f["accuracy"] >= 0.8 and f["avg_step_time"] < 15: 
        return "CONFIDENT"
    elif f["error_streak_max"] >= 2:
        return "FRUSTRATED"
    else:
        return "CONFUSED"

# ===============================
# ML RISK PREDICTOR
# ===============================
def predict_risk(f):
    X = pd.DataFrame([f], columns=FEATURES)
    return model.predict(X)[0]

# ===============================
# ADAPTATION POLICY
# ===============================
def decide_next_level(emotion, risk):
    # Debugging agar terlihat di terminal
    print(f"   -> Logic Check: Emotion='{emotion}', ML_Prediction='{risk}'")
    
    if emotion == "CONFIDENT" or risk == "HIGH":
        return "HIGH"
        
    if emotion == "FRUSTRATED" or risk == "LOW":
        return "LOW"

    return "MED"

# ===============================
# MAIN LEARNING LOOP
# ===============================
print("\n=== ADAPTIVE LEARNING SYSTEM (RANDOMIZED) ===\n")

while True:
    print("\n--------------------------------")
    print(materials[current_level])
    
    # PERUBAHAN DI SINI: Pilih soal secara ACAK dari list
    question, correct_answer = random.choice(questions[current_level])
    
    print("Soal:", question)

    start_step = time.time()
    user_input = input("Jawaban x = ").strip()
    step_time = time.time() - start_step
    step_times.append(step_time)

    attempts += 1

    if user_input.isdigit() and int(user_input) == correct_answer:
        corrects += 1
        error_streak = 0
        print("✅ Jawaban benar")
    else:
        error_streak += 1
        print("❌ Jawaban salah")

    # ===== AI PIPELINE =====
    features = extract_features()
    emotion = infer_emotion(features)
    risk = predict_risk(features)
    
    print(f"   -> Debug Stats: Acc={features['accuracy']:.2f}, AvgTime={features['avg_step_time']:.2f}s")
    
    next_level = decide_next_level(emotion, risk)

    print("\n[AI FEEDBACK]")
    print("Emotion:", emotion)
    print("Risk (ML):", risk)
    print("Next Level:", next_level)

    # ===== RESET STATISTIK JIKA CONFIDENT =====
    if emotion == "CONFIDENT":
        print(">> Mereset statistik agar evaluasi berikutnya fresh...")
        attempts = 0
        corrects = 0
        error_streak = 0
        step_times = [] 

    # ===== STOP CONDITION =====
    if next_level == "HIGH":
        high_streak += 1
    else:
        high_streak = 0

    if high_streak >= MAX_HIGH_STREAK:
        print("\n🎉 PEMBELAJARAN SELESAI")
        print("Siswa telah mencapai pemahaman yang stabil.")
        break

    current_level = next_level