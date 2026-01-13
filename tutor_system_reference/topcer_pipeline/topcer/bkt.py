from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class BKTParams:
    pL0: float = 0.2  # initial mastery
    pT: float = 0.15  # learn/transition
    pS: float = 0.1  # slip
    pG: float = 0.2  # guess

    def clamp(self) -> "BKTParams":
        self.pL0 = float(np.clip(self.pL0, 1e-4, 1 - 1e-4))
        self.pT = float(np.clip(self.pT, 1e-4, 1 - 1e-4))
        self.pS = float(np.clip(self.pS, 1e-4, 1 - 1e-4))
        self.pG = float(np.clip(self.pG, 1e-4, 1 - 1e-4))
        return self


def bkt_update(pL: float, correct: int, params: BKTParams) -> float:
    """One BKT step update for a single skill observation."""
    pL = float(np.clip(pL, 1e-6, 1 - 1e-6))
    if correct not in (0, 1):
        correct = 0

    # P(Correct | Learned) = 1 - slip; P(Correct | Not Learned) = guess
    if correct == 1:
        num = pL * (1.0 - params.pS)
        den = num + (1.0 - pL) * params.pG
    else:
        num = pL * params.pS
        den = num + (1.0 - pL) * (1.0 - params.pG)

    pL_given_obs = num / max(den, 1e-12)

    # Learning transition
    pL_next = pL_given_obs + (1.0 - pL_given_obs) * params.pT
    return float(np.clip(pL_next, 0.0, 1.0))


def neg_loglik(sequence: List[int], params: BKTParams) -> float:
    """Negative log-likelihood for a single correctness sequence for one skill."""
    pL = params.pL0
    nll = 0.0
    for y in sequence:
        y = 1 if y == 1 else 0
        # emission prob
        p_correct = pL * (1.0 - params.pS) + (1.0 - pL) * params.pG
        p_y = p_correct if y == 1 else (1.0 - p_correct)
        nll -= float(np.log(max(p_y, 1e-12)))
        pL = bkt_update(pL, y, params)
    return nll


def fit_global_bkt(skill_sequences: Dict[str, List[int]],
                   n_trials: int = 200,
                   seed: int = 42) -> BKTParams:
    """Fit a single global BKT parameter set via random search (fast, robust baseline).

    skill_sequences: dict kc -> list of 0/1 correctness observations (in order).

    This is intentionally lightweight for a scaffold; you can replace with EM later.
    """
    rng = np.random.default_rng(seed)

    def sample_params() -> BKTParams:
        return BKTParams(
            pL0=float(rng.uniform(0.05, 0.6)),
            pT=float(rng.uniform(0.01, 0.35)),
            pS=float(rng.uniform(0.01, 0.25)),
            pG=float(rng.uniform(0.01, 0.35)),
        ).clamp()

    # Precompute sequences list to speed scoring
    seqs = [seq for seq in skill_sequences.values() if len(seq) >= 5]
    if not seqs:
        return BKTParams().clamp()

    best = BKTParams().clamp()
    best_score = float("inf")

    # Include a few sensible baselines
    candidates = [
        BKTParams(0.2, 0.15, 0.1, 0.2).clamp(),
        BKTParams(0.3, 0.10, 0.1, 0.15).clamp(),
        BKTParams(0.1, 0.20, 0.05, 0.2).clamp(),
    ]

    for _ in range(max(0, n_trials - len(candidates))):
        candidates.append(sample_params())

    for p in candidates:
        score = 0.0
        for seq in seqs:
            score += neg_loglik(seq, p)
        if score < best_score:
            best_score = score
            best = p

    return best.clamp()


def apply_bkt_to_student_steps(
    kc_steps: List[Tuple[str, int]],
    params: BKTParams,
    init: float | None = None,
) -> Dict[str, float]:
    """Given a list of (kc, correct) in chronological order, return final mastery per kc."""
    mastery: Dict[str, float] = {}
    for kc, y in kc_steps:
        cur = mastery.get(kc, params.pL0 if init is None else float(init))
        mastery[kc] = bkt_update(cur, 1 if y == 1 else 0, params)
    return mastery
