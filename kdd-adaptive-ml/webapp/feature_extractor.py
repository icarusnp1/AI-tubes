from collections import Counter

def compute_features(answer_log: list) -> dict:
    n = len(answer_log)
    if n == 0:
        return {}

    times = [float(x.get("response_time", 0.0)) for x in answer_log]
    corrects = [1 if x.get("is_correct") else 0 for x in answer_log]
    accuracy = sum(corrects) / n

    # max consecutive errors
    streak = 0
    streak_max = 0
    for c in corrects:
        if c == 0:
            streak += 1
            streak_max = max(streak_max, streak)
        else:
            streak = 0

    # repeated wrong concept rate
    errors = [x for x in answer_log if not x.get("is_correct")]
    if errors:
        counts = Counter([(e.get("concept_tag") or "") for e in errors])
        if "" in counts:
            del counts[""]
        repeated = sum(v for v in counts.values() if v >= 2)
        repeat_error_rate = repeated / len(errors)
    else:
        repeat_error_rate = 0.0

    help_used = sum(1 for x in answer_log if x.get("used_hint") or x.get("used_explanation"))
    help_rate = help_used / n

    rapid_wrong = sum(1 for x in answer_log if (not x.get("is_correct") and float(x.get("response_time", 0.0)) < 2.5))
    rapid_guess_rate = rapid_wrong / n

    return {
        "num_steps": float(n),
        "accuracy": float(accuracy),
        "avg_step_time": float(sum(times) / n),
        "max_step_time": float(max(times)),
        "total_time": float(sum(times)),
        "error_streak_max": float(streak_max),
        "repeat_error_rate": float(repeat_error_rate),
        "help_rate": float(help_rate),
        "rapid_guess_rate": float(rapid_guess_rate),
    }

def top_wrong_concepts(answer_log: list, k: int = 2) -> list[str]:
    errors = [x for x in answer_log if not x.get("is_correct")]
    if not errors:
        return []
    c = Counter([(e.get("concept_tag") or "") for e in errors])
    if "" in c:
        del c[""]
    return [t for t, _ in c.most_common(k)]
