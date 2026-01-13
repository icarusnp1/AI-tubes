#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from topcer.models import GRUMultiTask, ModelConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_pt', required=True)
    ap.add_argument('--npz', required=True, help='A seq_train.npz or seq_test.npz to demo inference')
    ap.add_argument('--n', type=int, default=2, help='Number of windows to run')
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    X = data['X'][:args.n]
    kc_id = data['kc_id'][:args.n]
    mask = data['mask'][:args.n]
    meta = __import__('json').loads(str(data['meta']))

    ckpt = torch.load(args.model_pt, map_location='cpu')
    cfg = ModelConfig(**ckpt['config'])
    model = GRUMultiTask(cfg)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    with torch.no_grad():
        out = model(torch.from_numpy(X).float(), torch.from_numpy(kc_id).long())
        p_correct = torch.sigmoid(out['logit_correct']).numpy()
        p_hint = torch.sigmoid(out['logit_hint']).numpy()
        state = out['logit_state'].argmax(-1).numpy()

    print('p_correct_next (first window, first 10 steps):', np.round(p_correct[0,:10], 3).tolist())
    print('p_hint_next (first window, first 10 steps):', np.round(p_hint[0,:10], 3).tolist())
    print('state_id (first window, first 10 steps):', state[0,:10].tolist())


if __name__ == '__main__':
    main()
