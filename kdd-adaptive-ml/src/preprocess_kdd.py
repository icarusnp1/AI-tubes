from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    guess_delimiter,
    normalize_columns,
    infer_kdd_columns,
    coerce_correctness,
    safe_to_datetime,
)

def build_sessions(df: pd.DataFrame, colmap, max_gap_minutes: int = 30) -> pd.DataFrame:
    """
    Build session_id per student+problem by splitting when a time gap is large,
    or (if no timestamp) treat each student+problem group as one session.
    """
    df = df.copy()

    # correctness to 0/1
    df["_correct01"] = coerce_correctness(df[colmap.correct])

    # timestamp handling
    has_ts = colmap.timestamp is not None
    if has_ts:
        df["_ts"] = safe_to_datetime(df[colmap.timestamp])
    else:
        df["_ts"] = pd.NaT

    # step time handling (seconds)
    has_step_time = colmap.step_time is not None
    if has_step_time:
        st = pd.to_numeric(df[colmap.step_time], errors="coerce")
        # Some datasets store ms; if huge, convert to seconds
        st_sec = st.copy()
        if st_sec.dropna().median() and st_sec.dropna().median() > 1000:
            st_sec = st_sec / 1000.0
        df["_step_time_sec"] = st_sec.fillna(0.0)
    else:
        df["_step_time_sec"] = np.nan

    # skill tagging optional
    if colmap.skill is not None:
        df["_skill"] = df[colmap.skill].astype(str)
    else:
        df["_skill"] = None

    # attempt optional
    if colmap.attempt is not None:
        df["_attempt"] = pd.to_numeric(df[colmap.attempt], errors="coerce")
    else:
        df["_attempt"] = np.nan

    # sorting key
    sort_cols = [colmap.student_id, colmap.problem_id]
    if has_ts:
        sort_cols.append("_ts")
    elif colmap.step_id is not None:
        sort_cols.append(colmap.step_id)

    df = df.sort_values(sort_cols).reset_index(drop=True)

    # create session_id
    df["_session_index"] = 0

    if has_ts:
        # split session when gap between consecutive events > max_gap_minutes
        df["_prev_ts"] = df.groupby([colmap.student_id, colmap.problem_id])["_ts"].shift(1)
        gap = (df["_ts"] - df["_prev_ts"]).dt.total_seconds() / 60.0
        new_session = (gap.isna()) | (gap > max_gap_minutes)
        # cumulative sum within each student+problem
        df["_session_index"] = new_session.groupby([df[colmap.student_id], df[colmap.problem_id]]).cumsum().astype(int) - 1
    else:
        df["_session_index"] = 0

    df["session_id"] = (
        df[colmap.student_id].astype(str)
        + "__"
        + df[colmap.problem_id].astype(str)
        + "__"
        + df["_session_index"].astype(str)
    )

    return df


def aggregate_session_features(df: pd.DataFrame, colmap) -> pd.DataFrame:
    grp_cols = ["session_id", colmap.student_id, colmap.problem_id]

    def error_streak_max(arr: np.ndarray) -> int:
        # arr is correctness 0/1 in time order
        max_streak = 0
        current = 0
        for v in arr:
            if v == 0:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return int(max_streak)

    agg = df.groupby(grp_cols).agg(
        num_steps=("_correct01", "size"),
        num_correct_steps=("_correct01", "sum"),
        avg_step_time=("_step_time_sec", "mean"),
        max_step_time=("_step_time_sec", "max"),
        total_step_time=("_step_time_sec", "sum"),
    ).reset_index()

    agg["num_incorrect_steps"] = agg["num_steps"] - agg["num_correct_steps"]
    agg["accuracy"] = np.where(agg["num_steps"] > 0, agg["num_correct_steps"] / agg["num_steps"], 0.0)

    # error streak max
    streak = (
        df.groupby(grp_cols)["_correct01"]
        .apply(lambda s: error_streak_max(s.to_numpy()))
        .reset_index(name="error_streak_max")
    )
    agg = agg.merge(streak, on=grp_cols, how="left")

    # repeat_error_rate / unique_skills (optional)
    if df["_skill"].notna().any():
        # compute error rate per skill inside session then measure repeats
        def repeat_error_rate(session_df: pd.DataFrame) -> float:
            err = session_df[session_df["_correct01"] == 0]
            if len(err) == 0:
                return 0.0
            # if skill missing, treat as no info
            skills = err["_skill"].dropna()
            if len(skills) == 0:
                return 0.0
            counts = skills.value_counts()
            # errors that are in skills with count>=2 are repeats
            repeated = counts[counts >= 2].sum()
            return float(repeated) / float(len(err))

        rer = (
            df.groupby(grp_cols)
            .apply(repeat_error_rate)
            .reset_index(name="repeat_error_rate")
        )

        uniq = (
            df.groupby(grp_cols)["_skill"]
            .nunique(dropna=True)
            .reset_index(name="unique_skills")
        )
        agg = agg.merge(rer, on=grp_cols, how="left")
        agg = agg.merge(uniq, on=grp_cols, how="left")
    else:
        agg["repeat_error_rate"] = 0.0
        agg["unique_skills"] = 0

    # timestamps (optional)
    if "_ts" in df.columns and df["_ts"].notna().any():
        ts = df.groupby(grp_cols).agg(
            timestamp_start=("_ts", "min"),
            timestamp_end=("_ts", "max"),
        ).reset_index()
        agg = agg.merge(ts, on=grp_cols, how="left")
        agg["total_time"] = (agg["timestamp_end"] - agg["timestamp_start"]).dt.total_seconds().fillna(agg["total_step_time"])
    else:
        agg["timestamp_start"] = pd.NaT
        agg["timestamp_end"] = pd.NaT
        agg["total_time"] = agg["total_step_time"].fillna(0.0)

    # final feature cleanup
    agg["avg_step_time"] = agg["avg_step_time"].fillna(0.0)
    agg["max_step_time"] = agg["max_step_time"].fillna(0.0)
    agg["total_time"] = agg["total_time"].fillna(0.0)

    # label 3-class from accuracy
    conditions = [
        agg["accuracy"] < 0.50,
        (agg["accuracy"] >= 0.50) & (agg["accuracy"] < 0.80),
        agg["accuracy"] >= 0.80,
    ]
    labels = ["LOW", "MED", "HIGH"]
    agg["understanding_class"] = np.select(conditions, labels, default="MED")

    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to raw KDD log file (CSV/TSV).")
    ap.add_argument("--out", default="data/processed/sessions.csv", help="Output CSV path.")
    ap.add_argument("--max_gap_minutes", type=int, default=30, help="Session split gap in minutes (if timestamp exists).")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    delim = guess_delimiter(str(in_path))
    df = pd.read_csv(in_path, sep=delim, low_memory=False)
    df = normalize_columns(df)

    colmap = infer_kdd_columns(df)

    df_sessions = build_sessions(df, colmap, max_gap_minutes=args.max_gap_minutes)
    sessions = aggregate_session_features(df_sessions, colmap)

    sessions.to_csv(out_path, index=False)

    # Save column map for reproducibility
    meta_path = out_path.with_suffix(".column_map.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(colmap.__dict__, f, indent=2)

    print(f"Saved: {out_path}")
    print(f"Saved: {meta_path}")
    print("Class distribution:")
    print(sessions["understanding_class"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
