#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from topcer.seq_dataset import SeqConfig, build_vocab_kc, build_windows
from topcer.train_utils import save_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps_parquet', required=True, help='Path to steps_with_mastery.parquet')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--window', type=int, default=100)
    ap.add_argument('--stride', type=int, default=50)
    ap.add_argument('--test_size', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.steps_parquet)

    # Basic cleaning
    df = df[df['t'].notna()].copy()
    df['duration'] = df['duration'].fillna(df['duration'].median()).astype('float32')
    df['dt_prev'] = df['dt_prev'].fillna(0.0).astype('float32')

    # Student split
    students = df['student_id'].astype(str).unique()
    tr_stu, te_stu = train_test_split(students, test_size=args.test_size, random_state=args.seed)

    train_df = df[df['student_id'].isin(tr_stu)].copy()
    test_df = df[df['student_id'].isin(te_stu)].copy()

    # Build KC vocab from train only
    kc_vocab = build_vocab_kc(train_df)

    cfg = SeqConfig(window=args.window, stride=args.stride)

    X_tr, tgt_tr = build_windows(train_df, cfg, kc_vocab)
    X_te, tgt_te = build_windows(test_df, cfg, kc_vocab)

    feature_names = [
        'y_correct','duration','incorrects','hints','dt_prev',
        'error_streak_run','hint_rate_run','acc_run','mastery_mean','mastery_min','rapid_wrong'
    ]

    meta = {
        'window': args.window,
        'stride': args.stride,
        'feature_dim': int(X_tr.shape[-1]) if len(X_tr) else int(X_te.shape[-1]),
        'kc_vocab': kc_vocab,
        'feature_names': feature_names,
        'train_students': len(tr_stu),
        'test_students': len(te_stu),
    }

    save_npz(out_dir / 'seq_train.npz', X_tr, tgt_tr, meta)
    save_npz(out_dir / 'seq_test.npz', X_te, tgt_te, meta)

    # Convenience combined path reference
    (out_dir / 'seq_dataset_meta.json').write_text(json.dumps(meta, indent=2))

    print('Wrote:', out_dir / 'seq_train.npz')
    print('Wrote:', out_dir / 'seq_test.npz')
    print('Wrote:', out_dir / 'seq_dataset_meta.json')


if __name__ == '__main__':
    main()
