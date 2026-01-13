#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from topcer.models import GRUMultiTask, ModelConfig
from topcer.train_utils import load_npz, masked_bce_with_logits, masked_ce


def pseudo_state_from_features(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Weak-supervised affective state labels.

    States (ids):
      0 FLOW
      1 CONFUSED
      2 FRUSTRATED
      3 GUESSING
      4 DISENGAGED

    This is intentionally simple; refine thresholds based on distribution.
    """
    fi = {n: i for i, n in enumerate(feature_names)}
    y = X[:, :, fi['y_correct']]
    dur = X[:, :, fi['duration']]
    hints = X[:, :, fi['hints']]
    dt = X[:, :, fi['dt_prev']]
    rapid_wrong = X[:, :, fi.get('rapid_wrong', fi['y_correct'])]  # fallback

    # rolling proxies per window (simple, per timestep)
    # Note: thresholds are heuristic.
    slow = dur > np.nanpercentile(dur, 70)
    many_hints = hints > 0
    long_gap = dt > np.nanpercentile(dt, 85)

    wrong = y < 0.5
    guessing = (rapid_wrong > 0.5) | (wrong & (dur < np.nanpercentile(dur, 25)))
    frustrated = wrong & many_hints
    confused = wrong & slow & (~many_hints)
    disengaged = long_gap

    state = np.zeros_like(y, dtype=np.int64)  # FLOW
    state[confused] = 1
    state[frustrated] = 2
    state[guessing] = 3
    state[disengaged] = 4
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_npz', required=True, help='Path to seq_train.npz')
    ap.add_argument('--test_npz', required=True, help='Path to seq_test.npz')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--w_hint', type=float, default=0.5)
    ap.add_argument('--w_time', type=float, default=0.5)
    ap.add_argument('--w_state', type=float, default=0.2)
    ap.add_argument('--no_state_head', action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_tr, tgt_tr, meta = load_npz(args.train_npz)
    X_te, tgt_te, _ = load_npz(args.test_npz)

    feature_names = meta.get('feature_names')
    if not feature_names:
        # fallback assumption
        feature_names = ['y_correct','duration','incorrects','hints','dt_prev','error_streak_run','hint_rate_run','acc_run','mastery_mean','mastery_min','rapid_wrong']

    # Add pseudo state labels
    if not args.no_state_head:
        tgt_tr['y_state'] = pseudo_state_from_features(X_tr, feature_names)
        tgt_te['y_state'] = pseudo_state_from_features(X_te, feature_names)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Torch datasets
    def to_ds(X, tgt):
        tensors = {
            'X': torch.from_numpy(X).float(),
            'kc_id': torch.from_numpy(tgt['kc_id']).long(),
            'mask': torch.from_numpy(tgt['mask']).float(),
            'y_correct': torch.from_numpy(tgt['y_correct']).long(),
            'y_hint': torch.from_numpy(tgt['y_hint']).long(),
            'y_timebin': torch.from_numpy(tgt['y_timebin']).long(),
        }
        if 'y_state' in tgt:
            tensors['y_state'] = torch.from_numpy(tgt['y_state']).long()
        return tensors

    tr = to_ds(X_tr, tgt_tr)
    te = to_ds(X_te, tgt_te)

    train_ds = TensorDataset(tr['X'], tr['kc_id'], tr['mask'], tr['y_correct'], tr['y_hint'], tr['y_timebin'], tr.get('y_state', torch.zeros_like(tr['y_correct'])))
    test_ds = TensorDataset(te['X'], te['kc_id'], te['mask'], te['y_correct'], te['y_hint'], te['y_timebin'], te.get('y_state', torch.zeros_like(te['y_correct'])))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    cfg = ModelConfig(
        input_dim=int(meta['feature_dim']),
        kc_vocab_size=len(meta['kc_vocab']),
        kc_emb_dim=16,
        rnn_hidden=64,
        rnn_layers=1,
        dropout=0.1,
        num_states=5,
    )

    model = GRUMultiTask(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def run_epoch(loader, train: bool):
        model.train(train)
        tot_loss = 0.0
        n = 0
        for X, kc_id, mask, yC, yH, yT, yS in loader:
            X = X.to(device)
            kc_id = kc_id.to(device)
            mask = mask.to(device)
            yC = yC.to(device)
            yH = yH.to(device)
            yT = yT.to(device)
            yS = yS.to(device)

            out = model(X, kc_id)

            loss_c = masked_bce_with_logits(out['logit_correct'], yC, mask)
            loss_h = masked_bce_with_logits(out['logit_hint'], yH, mask)
            loss_t = masked_ce(out['logit_time'], yT, mask)

            loss = loss_c + args.w_hint * loss_h + args.w_time * loss_t

            if not args.no_state_head:
                loss_s = masked_ce(out['logit_state'], yS, mask)
                loss = loss + args.w_state * loss_s

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            tot_loss += float(loss.detach().cpu().item())
            n += 1
        return tot_loss / max(n, 1)

    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        tr_loss = run_epoch(train_loader, train=True)
        te_loss = run_epoch(test_loader, train=False)
        print(f'Epoch {epoch:02d} | train_loss={tr_loss:.4f} | test_loss={te_loss:.4f}')
        if te_loss < best:
            best = te_loss
            ckpt = {
                'model_state': model.state_dict(),
                'config': cfg.__dict__,
                'meta': meta,
            }
            torch.save(ckpt, out_dir / 'state_seq_model.pt')

    print('Saved best model to:', out_dir / 'state_seq_model.pt')


if __name__ == '__main__':
    main()
