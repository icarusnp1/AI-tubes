import pandas as pd
import numpy as np
from collections import defaultdict
import os

# ===== Konfigurasi =====
# Jika file kddb-raw tidak berada di folder kerja (kdd-adaptive-ml),
# ubah INPUT_FILE ke path lengkap.
INPUT_FILE = "data/raw/kddb-raw"
OUTPUT_FILE = "data/processed/sessions.csv"

CHUNK_SIZE = 150_000  # kalau RAM kecil, turunkan ke 50_000 - 100_000

# Kolom yang benar sesuai hasil inspect_header.py kamu
USECOLS = [
    "Anon Student Id",
    "Problem Name",
    "Correct First Attempt",
    "Step Duration (sec)",
    "KC(SubSkills)",          # <-- ini pengganti KC (Skills)
]

DTYPE = {
    "Anon Student Id": "string",
    "Problem Name": "string",
    "Correct First Attempt": "int8",
    "Step Duration (sec)": "float32",
    "KC(SubSkills)": "string",
}

# ===== Aggregator per (student, problem) =====
stats = defaultdict(lambda: {
    "n": 0,
    "correct_sum": 0,
    "time_sum": 0.0,
    "time_max": 0.0,
    "error_streak": 0,
    "error_streak_max": 0,
    "error_kc": defaultdict(int),   # KC -> count
    "error_total": 0,
})

def update_group(key, correct_arr, time_arr, kc_arr):
    s = stats[key]

    n = len(correct_arr)
    if n == 0:
        return

    s["n"] += n
    s["correct_sum"] += int(correct_arr.sum())

    # time aggregates
    t_sum = float(time_arr.sum())
    s["time_sum"] += t_sum
    t_max = float(time_arr.max())
    if t_max > s["time_max"]:
        s["time_max"] = t_max

    # error streak
    for c in correct_arr:
        if c == 0:
            s["error_streak"] += 1
            if s["error_streak"] > s["error_streak_max"]:
                s["error_streak_max"] = s["error_streak"]
        else:
            s["error_streak"] = 0

    # repeat error per KC
    for c, kc in zip(correct_arr, kc_arr):
        if c == 0:
            s["error_total"] += 1
            kc = str(kc).strip() if kc is not None else ""
            if not kc:
                continue

            # Di KDD, KC sering multi-value dipisah "~~"
            if "~~" in kc:
                for part in kc.split("~~"):
                    part = part.strip()
                    if part:
                        s["error_kc"][part] += 1
            else:
                s["error_kc"][kc] += 1


print("Reading in chunks...")

reader = pd.read_csv(
    INPUT_FILE,
    sep="\t",
    usecols=USECOLS,
    dtype=DTYPE,
    chunksize=CHUNK_SIZE,
    low_memory=False,
    engine="c",
)

for i, chunk in enumerate(reader, start=1):
    chunk = chunk.rename(columns={
        "Anon Student Id": "student_id",
        "Problem Name": "problem_id",
        "Correct First Attempt": "cfa",
        "Step Duration (sec)": "step_time",
        "KC(SubSkills)": "kc",
    })

    chunk["step_time"] = chunk["step_time"].fillna(0).astype("float32")
    chunk["cfa"] = chunk["cfa"].fillna(0).astype("int8")
    chunk["kc"] = chunk["kc"].fillna("").astype("string")

    for (sid, pid), g in chunk.groupby(["student_id", "problem_id"], sort=False):
        update_group(
            (str(sid), str(pid)),
            g["cfa"].to_numpy(),
            g["step_time"].to_numpy(),
            g["kc"].to_numpy(),
        )

    print(f"Processed chunk {i} (rows={len(chunk):,})")

# ===== Build session-level dataframe =====
rows = []
for (sid, pid), s in stats.items():
    n = s["n"]
    if n == 0:
        continue

    accuracy = s["correct_sum"] / n
    avg_step_time = s["time_sum"] / n
    max_step_time = s["time_max"]
    total_time = s["time_sum"]
    error_streak_max = s["error_streak_max"]

    if s["error_total"] > 0:
        repeated_errors = sum(cnt for cnt in s["error_kc"].values() if cnt >= 2)
        repeat_error_rate = repeated_errors / s["error_total"]
    else:
        repeat_error_rate = 0.0

    # label 3 kelas
    if accuracy < 0.50:
        cls = "LOW"
    elif accuracy < 0.80:
        cls = "MED"
    else:
        cls = "HIGH"

    rows.append({
        "student_id": sid,
        "problem_id": pid,
        "num_steps": n,
        "accuracy": float(accuracy),
        "avg_step_time": float(avg_step_time),
        "max_step_time": float(max_step_time),
        "total_time": float(total_time),
        "error_streak_max": int(error_streak_max),
        "repeat_error_rate": float(repeat_error_rate),
        "understanding_class": cls,
    })

sessions = pd.DataFrame(rows)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
sessions.to_csv(OUTPUT_FILE, index=False)

print("Saved:", OUTPUT_FILE)
print("Class distribution:")
print(sessions["understanding_class"].value_counts())
print("Total sessions:", len(sessions))
