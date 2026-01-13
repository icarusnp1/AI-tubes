#!/usr/bin/env python
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from topcer.bkt import BKTParams, bkt_update, fit_global_bkt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps_parquet', required=True, help='Path to steps.parquet')
    ap.add_argument('--kc_steps_parquet', default=None, help='Path to kc_steps.parquet (optional)')
    ap.add_argument('--out_dir', required=True, help='Output directory')
    ap.add_argument('--trials', type=int, default=200, help='Random search trials for global BKT')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = pd.read_parquet(args.steps_parquet)
    steps = steps.sort_values(['student_id', 't'], kind='mergesort')

    # Build skill sequences (kc -> list[y])
    skill_sequences = {}
    if args.kc_steps_parquet:
        kc_steps = pd.read_parquet(args.kc_steps_parquet)
        kc_steps = kc_steps.sort_values(['kc', 'student_id', 't'], kind='mergesort')
        for kc, g in kc_steps.groupby('kc'):
            skill_sequences[str(kc)] = g['y_correct'].astype(int).tolist()
    else:
        # Fallback: use primary_kc only
        for kc, g in steps.groupby('primary_kc'):
            if str(kc) == 'NO_KC':
                continue
            skill_sequences[str(kc)] = g['y_correct'].astype(int).tolist()

    params = fit_global_bkt(skill_sequences, n_trials=args.trials, seed=args.seed)

    # Save params
    params_path = out_dir / 'bkt_global_params.json'
    params_path.write_text(json.dumps(params.__dict__, indent=2))
    print('Saved:', params_path)

    # Add per-step mastery features (online per student)
    mastery_mean = np.zeros(len(steps), dtype=np.float32)
    mastery_min = np.zeros(len(steps), dtype=np.float32)

    # steps['kc_list'] is stored as list in parquet (pyarrow) – should load as python list
    # Ensure it's list-like
    def _as_list(v):
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v if str(x).strip()]
        # some parquet readers may load as string repr
        try:
            if isinstance(v, str) and v.startswith('[') and v.endswith(']'):
                import ast
                return [str(x) for x in ast.literal_eval(v)]
        except Exception:
            pass
        return []

    idx = 0
    for sid, g in steps.groupby('student_id', sort=False):
        m = {}  # kc -> pL
        for row_i, row in g.iterrows():
            kcs = _as_list(row.get('kc_list'))
            kcs = [k for k in kcs if "DON'T TRACK ME" not in k]
            if not kcs:
                mastery_mean[row_i] = float(params.pL0)
                mastery_min[row_i] = float(params.pL0)
            else:
                vals = [m.get(k, params.pL0) for k in kcs]
                mastery_mean[row_i] = float(np.mean(vals))
                mastery_min[row_i] = float(np.min(vals))

            # Update mastery after observing correctness for each KC on this step
            y = int(row.get('y_correct', 0))
            for k in kcs:
                cur = m.get(k, params.pL0)
                m[k] = bkt_update(cur, y, params)

    steps_out = steps.copy()
    steps_out['mastery_mean'] = mastery_mean
    steps_out['mastery_min'] = mastery_min

    out_path = out_dir / 'steps_with_mastery.parquet'
    steps_out.to_parquet(out_path, index=False)
    print('Wrote:', out_path)


if __name__ == '__main__':
    main()
