#!/usr/bin/env python
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from topcer.kdd_schema import KDD_COLS, MULTI_DELIM
from topcer.utils import parse_timestamp, split_multi_field, safe_float, safe_int


def build_primary_kc(row: pd.Series) -> str:
    # Prefer SubSkills; fallback to KTraced; else NO_KC
    kc = split_multi_field(row.get('kc_subskills'), MULTI_DELIM)
    if not kc:
        kc = split_multi_field(row.get('kc_ktraced'), MULTI_DELIM)
    if not kc:
        return 'NO_KC'
    # Remove sentinel
    kc = [k for k in kc if "DON'T TRACK ME" not in k]
    return kc[0] if kc else 'NO_KC'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_csv', required=True, help='Path to KDD CSV (Bridge to Algebra)')
    ap.add_argument('--out_dir', required=True, help='Output directory')
    ap.add_argument('--chunksize', type=int, default=300000, help='CSV chunksize for large files')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load in chunks to support huge files
    chunks = []
    usecols = list(KDD_COLS.keys())
    for chunk in pd.read_csv(args.in_csv, usecols=lambda c: c in usecols, chunksize=args.chunksize):
        chunk = chunk.rename(columns=KDD_COLS)
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    # Parse timestamps
    for c in ['first_tx_time', 'step_start_time', 'step_end_time', 'correct_tx_time']:
        if c in df.columns:
            df[c] = df[c].apply(parse_timestamp)

    # Canonical timestamp
    df['t'] = df['first_tx_time']
    missing = df['t'].isna()
    df.loc[missing, 't'] = df.loc[missing, 'step_start_time']

    # Numeric fields
    df['y_correct'] = df['correct_first_attempt'].apply(lambda x: 1 if safe_int(x, 0) == 1 else 0).astype('int8')
    df['incorrects'] = df['incorrects'].apply(lambda x: safe_int(x, 0)).astype('int16')
    df['hints'] = df['hints'].apply(lambda x: safe_int(x, 0)).astype('int16')
    df['duration'] = df['step_duration'].apply(lambda x: safe_float(x, np.nan)).astype('float32')

    # If duration missing, try to compute from timestamps
    dur_missing = df['duration'].isna() & df['step_start_time'].notna() & df['step_end_time'].notna()
    df.loc[dur_missing, 'duration'] = (df.loc[dur_missing, 'step_end_time'] - df.loc[dur_missing, 'step_start_time']).dt.total_seconds().astype('float32')

    # Primary KC id for sequence embedding
    df['primary_kc'] = df.apply(build_primary_kc, axis=1)

    # Keep raw multi-fields for later
    df['kc_list'] = df['kc_subskills'].apply(lambda v: split_multi_field(v, MULTI_DELIM))
    df['opp_list'] = df['opp_subskills'].apply(lambda v: split_multi_field(v, MULTI_DELIM))

    # Sort per student
    df = df.sort_values(['student_id', 't'], kind='mergesort')

    # Temporal features per student
    df['dt_prev'] = df.groupby('student_id')['t'].diff().dt.total_seconds().astype('float32')
    df['dt_prev'] = df['dt_prev'].fillna(0.0).clip(lower=0.0)

    # Running error streak
    def _error_streak(y: pd.Series) -> pd.Series:
        out = np.zeros(len(y), dtype=np.int16)
        streak = 0
        for i, v in enumerate(y.to_numpy()):
            if int(v) == 0:
                streak += 1
            else:
                streak = 0
            out[i] = streak
        return pd.Series(out, index=y.index)

    df['error_streak_run'] = (
        df.groupby('student_id')['y_correct']
            .transform(_error_streak)
            .astype('int16')
    )   

    # Rolling metrics (window=10)
    w = 10
    df['acc_run'] = df.groupby('student_id')['y_correct'].transform(lambda s: s.rolling(w, min_periods=1).mean()).astype('float32')
    df['hint_rate_run'] = df.groupby('student_id')['hints'].transform(lambda s: (s > 0).rolling(w, min_periods=1).mean()).astype('float32')

    # Rapid wrong flag (for later weak labels)
    df['rapid_wrong'] = ((df['y_correct'] == 0) & (df['duration'].fillna(9999.0) < 2.5)).astype('int8')

    # Save steps.parquet
    steps_cols = [
        'student_id', 'problem_hierarchy', 'problem_name', 'problem_view', 'step_name', 't',
        'y_correct', 'incorrects', 'hints', 'duration',
        'dt_prev', 'error_streak_run', 'acc_run', 'hint_rate_run', 'rapid_wrong',
        'primary_kc', 'kc_list', 'opp_list',
        'kc_subskills', 'opp_subskills', 'kc_ktraced', 'opp_ktraced'
    ]
    steps_cols = [c for c in steps_cols if c in df.columns]
    steps_path = out_dir / 'steps.parquet'
    df[steps_cols].to_parquet(steps_path, index=False)

    # Explode KC for KT (use subskills by default; fallback to ktraced if empty)
    exp_rows = []
    for r in df[['student_id', 't', 'y_correct', 'kc_subskills', 'opp_subskills', 'kc_ktraced', 'opp_ktraced']].itertuples(index=False):
        kc = split_multi_field(getattr(r, 'kc_subskills'), MULTI_DELIM)
        opp = split_multi_field(getattr(r, 'opp_subskills'), MULTI_DELIM)
        if not kc:
            kc = split_multi_field(getattr(r, 'kc_ktraced'), MULTI_DELIM)
            opp = split_multi_field(getattr(r, 'opp_ktraced'), MULTI_DELIM)
        if not kc:
            continue
        # align opp length
        if len(opp) != len(kc):
            opp = (opp + [''] * len(kc))[:len(kc)]
        for k, o in zip(kc, opp):
            if "DON'T TRACK ME" in str(k):
                continue
            try:
                o_int = int(float(o)) if str(o).strip() != '' else 0
            except Exception:
                o_int = 0
            exp_rows.append((r.student_id, r.t, int(r.y_correct), str(k), int(o_int)))

    kc_steps = pd.DataFrame(exp_rows, columns=['student_id', 't', 'y_correct', 'kc', 'opportunity'])
    kc_steps = kc_steps.sort_values(['student_id', 't'], kind='mergesort')
    kc_steps_path = out_dir / 'kc_steps.parquet'
    kc_steps.to_parquet(kc_steps_path, index=False)

    print('Wrote:', steps_path)
    print('Wrote:', kc_steps_path)


if __name__ == '__main__':
    main()