from dataclasses import dataclass
from mastery_tracker import weakest_concepts

@dataclass
class PolicyDecision:
    next_level: int
    next_set_size: int
    focus_concepts: list[str]
    hint_mode: str
    feedback_style: str
    message: str

def decide(level: int, struggle_pred: str, emotion_state: str, mastery_streak: int, mastery: dict, wrong_top2: list[str]) -> PolicyDecision:
    next_level = level
    next_set_size = 5
    hint_mode = "normal"
    feedback_style = "neutral"

    # focus priority: wrong_top2, else weakest mastery
    focus = wrong_top2[:] if wrong_top2 else weakest_concepts(mastery, k=2)

    if struggle_pred == "STRUGGLE":
        hint_mode = "aggressive"
        feedback_style = "supportive"
        next_set_size = 3
        if emotion_state in {"FRUSTRATED", "ANXIOUS"}:
            next_level = max(1, level - 1)
            msg = "Terdeteksi STRUGGLE. Kita turunkan beban dan remedial pada 2 konsep terlemah."
        else:
            msg = "Terdeteksi STRUGGLE. Kita tetap di level ini dan remedial pada 2 konsep terlemah."
        return PolicyDecision(next_level, next_set_size, focus, hint_mode, feedback_style, msg)

    # OK case
    if emotion_state == "CONFIDENT":
        if level < 3:
            next_level = level + 1
            msg = "Kamu terlihat stabil dan percaya diri. Kita naik level."
        else:
            msg = "Kamu stabil di level tertinggi. Lanjut pemantapan."
    elif emotion_state in {"FRUSTRATED", "ANXIOUS"}:
        next_set_size = 3
        feedback_style = "supportive"
        msg = "Hasil OK, tapi kamu tampak tegang. Kita lanjut set singkat dan fokus konsep terlemah."
    else:
        if mastery_streak >= 1 and level < 3:
            next_level = level + 1
            msg = "Kamu konsisten. Kita naik level."
        else:
            msg = "Bagus. Ulang 1 set untuk memastikan konsistensi."

    return PolicyDecision(next_level, next_set_size, focus, hint_mode, feedback_style, msg)
