"""Step simulator for TOPCER tutor backend.

Usage:
  python simulate_student.py --api http://localhost:8000 --student S1 --kc KC_12 --steps 20
"""
from __future__ import annotations
import argparse, time, random
import requests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--student", default="S1")
    ap.add_argument("--kc", default="KC_12")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args()

    hint_count = 0
    error_streak = 0
    rolling_p = []

    def trend(p: float) -> float:
        rolling_p.append(p)
        if len(rolling_p) > 5:
            rolling_p.pop(0)
        avg = sum(rolling_p) / len(rolling_p)
        return p - avg

    for t in range(args.steps):
        is_correct = random.random() < (0.55 + 0.02*t - 0.05*error_streak)
        if not is_correct:
            error_streak += 1
        else:
            error_streak = 0

        if not is_correct and random.random() < 0.30:
            hint_count += 1

        features = {
            "time_sec": max(5.0, random.gauss(25.0 + 10.0*error_streak, 6.0)),
            "error_count": 0 if is_correct else 1,
            "hint_count": hint_count,
            "streak": (1 if is_correct else -error_streak),
            "bkt_mastery": min(0.95, max(0.05, 0.5 + 0.02*t - 0.04*error_streak)),
        }

        infer = requests.post(
            f"{args.api}/infer_step",
            json={
                "student_id": args.student,
                "kc_id": args.kc,
                "timestep": t,
                "features": features,
                "raw_answer": "x=?" if not is_correct else "x=5",
                "is_correct": is_correct,
            },
            timeout=10,
        ).json()

        p = float(infer.get("p_correct", 0.5))

        pol = requests.post(
            f"{args.api}/policy_decide",
            json={
                "student_id": args.student,
                "kc_id": args.kc,
                "timestep": t,
                "p_correct": p,
                "bkt_mastery": float(features["mastery_mean"]),
                "time_sec": float(features["duration"]),
                "error_streak": int(error_streak),
                "hint_count": int(hint_count),
                "trend": float(trend(p)),
                "extra": {"misconception_tag": "move_terms_wrong"},
            },
            timeout=10,
        ).json()

        print(f"t={t:02d} correct={is_correct} p={p:.3f} action={pol.get('action')}")
        time.sleep(args.sleep)

if __name__ == "__main__":
    main()
