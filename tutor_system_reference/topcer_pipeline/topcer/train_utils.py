from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def masked_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # logits/target/mask: (B, T)
    loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction='none')
    loss = loss * mask
    return loss.sum() / mask.sum().clamp(min=1.0)


def masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # logits: (B, T, C), target: (B, T), mask: (B, T)
    B, T, C = logits.shape
    loss = F.cross_entropy(logits.reshape(B*T, C), target.reshape(B*T), reduction='none')
    loss = loss.reshape(B, T) * mask
    return loss.sum() / mask.sum().clamp(min=1.0)


def save_npz(path: str | Path, X: np.ndarray, targets: Dict[str, np.ndarray], meta: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, **targets, meta=json.dumps(meta))


def load_npz(path: str | Path) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict]:
    data = np.load(path, allow_pickle=True)
    X = data['X']
    targets = {k: data[k] for k in data.files if k not in {'X', 'meta'}}
    meta = json.loads(str(data['meta']))
    return X, targets, meta
