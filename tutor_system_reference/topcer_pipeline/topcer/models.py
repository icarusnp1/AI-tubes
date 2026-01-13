from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    input_dim: int
    kc_vocab_size: int
    kc_emb_dim: int = 16
    rnn_hidden: int = 64
    rnn_layers: int = 1
    dropout: float = 0.1
    num_states: int = 5  # optional state head


class GRUMultiTask(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.kc_emb = nn.Embedding(cfg.kc_vocab_size, cfg.kc_emb_dim)
        self.gru = nn.GRU(
            input_size=cfg.input_dim + cfg.kc_emb_dim,
            hidden_size=cfg.rnn_hidden,
            num_layers=cfg.rnn_layers,
            batch_first=True,
            dropout=(cfg.dropout if cfg.rnn_layers > 1 else 0.0),
        )
        self.dropout = nn.Dropout(cfg.dropout)

        # Heads
        self.head_correct = nn.Linear(cfg.rnn_hidden, 1)
        self.head_hint = nn.Linear(cfg.rnn_hidden, 1)
        self.head_time = nn.Linear(cfg.rnn_hidden, 3)
        self.head_state = nn.Linear(cfg.rnn_hidden, cfg.num_states)

    def forward(self, x: torch.Tensor, kc_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: (B, T, input_dim), kc_id: (B, T)"""
        kc = self.kc_emb(kc_id)
        z = torch.cat([x, kc], dim=-1)
        h, _ = self.gru(z)
        h = self.dropout(h)

        out = {
            'logit_correct': self.head_correct(h).squeeze(-1),
            'logit_hint': self.head_hint(h).squeeze(-1),
            'logit_time': self.head_time(h),
            'logit_state': self.head_state(h),
            'embedding': h,
        }
        return out
