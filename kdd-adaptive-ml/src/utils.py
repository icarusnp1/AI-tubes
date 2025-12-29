from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


def guess_delimiter(path: str) -> str:
    # Simple heuristic: if many tabs in first line -> \t else comma
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        line = f.readline()
    return "\t" if line.count("\t") > line.count(",") else ","


def normalize_colname(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]
    return df


def find_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


@dataclass
class KDDColumnMap:
    student_id: str
    problem_id: str
    correct: str
    # optional
    step_id: Optional[str] = None
    skill: Optional[str] = None
    attempt: Optional[str] = None
    timestamp: Optional[str] = None
    step_time: Optional[str] = None


def infer_kdd_columns(df: pd.DataFrame) -> KDDColumnMap:
    """
    Tries to infer key columns from KDD-style logs.
    You may need to edit candidate lists depending on your exact file.
    """
    student = find_first_existing(df, [
        "student_id", "anon_student_id", "user_id", "subject_id", "student"
    ])
    problem = find_first_existing(df, [
        "problem_id", "problem_name", "problem", "problem_view", "item_id"
    ])
    correct = find_first_existing(df, [
        "correct", "is_correct", "correct_first_attempt", "cfa", "outcome"
    ])

    step_id = find_first_existing(df, [
        "step_id", "step_name", "step", "kc_step", "transaction_id"
    ])
    skill = find_first_existing(df, [
        "skill", "kc", "knowledge_component", "kc_default", "kcs", "kc_name"
    ])
    attempt = find_first_existing(df, [
        "attempt", "attempt_count", "opportunity", "try_count"
    ])
    timestamp = find_first_existing(df, [
        "timestamp", "time", "start_time", "end_time", "action_time"
    ])
    step_time = find_first_existing(df, [
        "step_duration", "step_time", "duration", "response_time", "first_response_time"
    ])

    missing = [("student_id", student), ("problem_id", problem), ("correct", correct)]
    missing = [name for name, val in missing if val is None]
    if missing:
        raise ValueError(
            f"Could not infer required columns: {missing}. "
            f"Available columns: {list(df.columns)[:50]}..."
        )

    return KDDColumnMap(
        student_id=student,
        problem_id=problem,
        correct=correct,
        step_id=step_id,
        skill=skill,
        attempt=attempt,
        timestamp=timestamp,
        step_time=step_time,
    )


def coerce_correctness(series: pd.Series) -> pd.Series:
    """
    Converts correctness-like values into 0/1.
    Handles booleans, 0/1, "correct"/"incorrect", etc.
    """
    s = series.copy()

    if s.dtype == bool:
        return s.astype(int)

    # numeric strings or floats
    try:
        sn = pd.to_numeric(s, errors="coerce")
        # If values look like 0/1 or 0..1 probability -> threshold at 0.5
        if sn.notna().mean() > 0.7:
            # If values are in {0,1} mostly
            uniq = set(sn.dropna().unique().tolist())
            if uniq.issubset({0, 1}):
                return sn.fillna(0).astype(int)
            # If it's a probability-ish
            if sn.min() >= 0 and sn.max() <= 1:
                return (sn >= 0.5).astype(int)
    except Exception:
        pass

    # categorical
    s_str = s.astype(str).str.lower().str.strip()
    true_set = {"1", "true", "t", "yes", "y", "correct", "right", "success"}
    false_set = {"0", "false", "f", "no", "n", "incorrect", "wrong", "failure"}

    out = pd.Series([None] * len(s_str), index=s_str.index, dtype="float")
    out[s_str.isin(true_set)] = 1
    out[s_str.isin(false_set)] = 0
    out = out.fillna(0).astype(int)
    return out


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)
