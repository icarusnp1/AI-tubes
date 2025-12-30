from dataclasses import dataclass

@dataclass
class EmotionOutput:
    state: str  # CONFIDENT / NEUTRAL / CONFUSED / FRUSTRATED / ANXIOUS
    confusion: float
    frustration: float
    confidence: float

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def estimate_emotion(features: dict) -> EmotionOutput:
    """
    features expected (some may be missing; defaults applied):
      - accuracy
      - avg_step_time
      - error_streak_max
      - help_rate
      - rapid_guess_rate
    """
    accuracy = float(features.get("accuracy", 0.0))
    avg_t = float(features.get("avg_step_time", 0.0))
    streak = float(features.get("error_streak_max", 0.0))
    help_rate = float(features.get("help_rate", 0.0))
    rapid = float(features.get("rapid_guess_rate", 0.0))

    # Normalize (prototype thresholds; nanti bisa dituning)
    # avg_t: >20s dianggap tinggi
    time_norm = clamp01(avg_t / 20.0)
    streak_norm = clamp01(streak / 3.0)
    help_norm = clamp01(help_rate / 0.6)
    rapid_norm = clamp01(rapid / 0.4)

    confusion = 100.0 * (0.45 * time_norm + 0.35 * streak_norm + 0.20 * help_norm)
    frustration = 100.0 * (0.50 * streak_norm + 0.30 * help_norm + 0.20 * rapid_norm)
    confidence = 100.0 * (0.65 * accuracy + 0.25 * (1.0 - time_norm) + 0.10 * (1.0 - help_norm))

    # State decision (priority)
    if frustration >= 70.0:
        state = "FRUSTRATED"
    elif confusion >= 60.0:
        state = "CONFUSED"
    elif rapid >= 0.3 and accuracy < 0.6:
        state = "ANXIOUS"
    elif confidence >= 70.0 and confusion < 40.0:
        state = "CONFIDENT"
    else:
        state = "NEUTRAL"

    return EmotionOutput(state=state, confusion=confusion, frustration=frustration, confidence=confidence)
