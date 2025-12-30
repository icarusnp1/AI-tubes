from dataclasses import dataclass

@dataclass
class PolicyDecision:
    next_level: int
    next_set_size: int
    hint_mode: str         # "normal" / "aggressive"
    feedback_style: str    # "neutral" / "supportive"
    message: str

def decide(level: int, struggle_pred: str, emotion_state: str, mastery_streak: int) -> PolicyDecision:
    """
    level: 1..3
    struggle_pred: "STRUGGLE" or "OK"
    emotion_state: from emotion engine
    mastery_streak: how many consecutive OK with good performance (tracked in app)
    """

    # Defaults
    next_level = level
    next_set_size = 5
    hint_mode = "normal"
    feedback_style = "neutral"
    message = ""

    if struggle_pred == "STRUGGLE":
        hint_mode = "aggressive"
        feedback_style = "supportive"
        next_set_size = 3  # reduce load

        # If frustrated/anxious, avoid escalating difficulty
        if emotion_state in {"FRUSTRATED", "ANXIOUS"}:
            next_level = max(1, level - 1)
            message = "Kamu terlihat sedang kesulitan. Kita turunkan beban dan ulang konsep dengan contoh tambahan."
        else:
            next_level = level  # stay
            message = "Kita tetap di level ini dulu. Aku tambahkan contoh dan latihan yang lebih terstruktur."

    else:  # OK
        if emotion_state == "CONFIDENT":
            # advance if possible
            if level < 3:
                next_level = level + 1
                message = "Bagus. Kamu stabil dan percaya diri. Kita naik level."
            else:
                message = "Bagus. Kamu sudah di level tertinggi. Lanjut pemantapan."
        elif emotion_state == "FRUSTRATED":
            # keep level but reduce stress
            next_level = level
            next_set_size = 3
            feedback_style = "supportive"
            message = "Hasilmu OK, tapi kamu tampak tegang. Kita lanjut pelan-pelan dengan set lebih pendek."
        else:
            # neutral progression: require streak to level up
            if mastery_streak >= 1 and level < 3:
                next_level = level + 1
                message = "Kamu konsisten. Kita naik level."
            else:
                next_level = level
                message = "Bagus. Kita lanjut 1 set lagi untuk memastikan konsisten."

    return PolicyDecision(
        next_level=next_level,
        next_set_size=next_set_size,
        hint_mode=hint_mode,
        feedback_style=feedback_style,
        message=message,
    )
