from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class SeqConfig:
    window: int = 100
    stride: int = 50
    rapid_thr: float = 2.5


def make_timebin(durations: np.ndarray, q=(0.33, 0.66)) -> np.ndarray:
    """Discretize duration into 3 bins using global quantiles."""
    d = np.asarray(durations, dtype=np.float32)
    q1, q2 = np.quantile(d[~np.isnan(d)], q)
    out = np.zeros(len(d), dtype=np.int64)
    out[d > q1] = 1
    out[d > q2] = 2
    return out


def build_vocab_kc(df: pd.DataFrame) -> Dict[str, int]:
    """Build KC vocab from a 'primary_kc' column."""
    vals = df['primary_kc'].fillna('NO_KC').astype(str)
    uniq = sorted(vals.unique().tolist())
    return {kc: i for i, kc in enumerate(uniq)}


def build_windows(df: pd.DataFrame, cfg: SeqConfig, kc_vocab: Dict[str, int]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return X and targets as numpy arrays.

    Expected columns:
    - student_id, t
    - y_correct (0/1)
    - hints, incorrects
    - duration, dt_prev
    - error_streak_run, hint_rate_run, acc_run
    - mastery_mean, mastery_min
    - primary_kc
    """
    feature_cols = [
        'y_correct', 'duration', 'incorrects', 'hints', 'dt_prev',
        'error_streak_run', 'hint_rate_run', 'acc_run',
        'mastery_mean', 'mastery_min',
        'rapid_wrong'
    ]

    # Ensure sorted
    df = df.sort_values(['student_id', 't'])

    # Prepare per-row arrays
    feats = df[feature_cols].astype('float32').to_numpy()
    y_correct = df['y_correct'].astype('int64').to_numpy()
    y_hint = (df['hints'].fillna(0).astype('float32').to_numpy() > 0).astype('int64')
    kc_ids = (
        df['primary_kc']
            .fillna('NO_KC')
            .astype(str)
            .map(kc_vocab)
            .fillna(kc_vocab.get('NO_KC', 0))
            .astype('int64')
            .to_numpy()
    )


    # time-bin will be computed per full df (global quantiles)
    y_timebin = make_timebin(df['duration'].astype('float32').to_numpy())

    # Build windows
    X_list = []
    yC_list, yH_list, yT_list, kc_list, mask_list = [], [], [], [], []

    for sid, g in df.groupby('student_id', sort=False):
        idx = g.index.to_numpy()
        n = len(idx)
        start = 0
        while start < n:
            end = start + cfg.window
            w_idx = idx[start:end]
            if len(w_idx) < 5:
                break

            # slice
            f = feats[df.index.get_indexer(w_idx)]
            yc = y_correct[df.index.get_indexer(w_idx)]
            yh = y_hint[df.index.get_indexer(w_idx)]
            yt = y_timebin[df.index.get_indexer(w_idx)]
            k = kc_ids[df.index.get_indexer(w_idx)]

            # pad
            L = cfg.window
            pad_len = L - len(w_idx)
            if pad_len > 0:
                f = np.pad(f, ((0, pad_len), (0, 0)), mode='constant')
                yc = np.pad(yc, (0, pad_len), mode='constant')
                yh = np.pad(yh, (0, pad_len), mode='constant')
                yt = np.pad(yt, (0, pad_len), mode='constant')
                k = np.pad(k, (0, pad_len), mode='constant')

            mask = np.zeros(L, dtype=np.float32)
            mask[:len(w_idx)] = 1.0

            X_list.append(f)
            yC_list.append(yc)
            yH_list.append(yh)
            yT_list.append(yt)
            kc_list.append(k)
            mask_list.append(mask)

            start += cfg.stride

    X = np.stack(X_list).astype('float32')
    targets = {
        'y_correct': np.stack(yC_list).astype('int64'),
        'y_hint': np.stack(yH_list).astype('int64'),
        'y_timebin': np.stack(yT_list).astype('int64'),
        'kc_id': np.stack(kc_list).astype('int64'),
        'mask': np.stack(mask_list).astype('float32'),
    }
    return X, targets
