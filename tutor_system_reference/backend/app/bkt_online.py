# backend/app/bkt_online.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class BKTParams:
    # Map from your JSON keys:
    # pL0 -> p_init, pT -> p_transit, pS -> p_slip, pG -> p_guess
    p_init: float = 0.35919390355566466
    p_transit: float = 0.04464401348694526
    p_slip: float = 0.16670131037535962
    p_guess: float = 0.33481880558385574


def _resolve_default_path() -> str:
    # backend/app/bkt_online.py -> backend/app -> backend -> tutor_system_reference -> project_root
    import pathlib
    here = pathlib.Path(__file__).resolve()
    tutor_root = here.parents[2]        # tutor_system_reference
    project_root = tutor_root.parent    # project_root
    return str(project_root / "topcer_pipeline" / "data" / "processed" / "bkt_global_params.json")


def load_global_bkt_params() -> BKTParams:
    """
    Expects a GLOBAL JSON:
      { "pL0": float, "pT": float, "pS": float, "pG": float }
    Env override:
      TOPCER_BKT_PARAMS_PATH
    """
    path = os.getenv("TOPCER_BKT_PARAMS_PATH", "") or _resolve_default_path()
    if not os.path.exists(path):
        return BKTParams()

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # robust read with defaults
    return BKTParams(
        p_init=float(raw.get("pL0", BKTParams.p_init)),
        p_transit=float(raw.get("pT", BKTParams.p_transit)),
        p_slip=float(raw.get("pS", BKTParams.p_slip)),
        p_guess=float(raw.get("pG", BKTParams.p_guess)),
    )


def bkt_update(pL: float, y_correct: int, prm: BKTParams) -> float:
    """
    Standard BKT update:
      posterior given observation
      then learning transition
    """
    y = 1 if int(y_correct) == 1 else 0
    guess = prm.p_guess
    slip = prm.p_slip
    transit = prm.p_transit

    if y == 1:
        num = pL * (1.0 - slip)
        den = num + (1.0 - pL) * guess
    else:
        num = pL * slip
        den = num + (1.0 - pL) * (1.0 - guess)

    post = num / den if den > 1e-12 else pL
    p_next = post + (1.0 - post) * transit

    # clamp
    if p_next < 0.0:
        p_next = 0.0
    elif p_next > 1.0:
        p_next = 1.0
    return float(p_next)


# -----------------------------
# Online state (in-memory MVP)
# Global params; store mastery per (student,kc)
# -----------------------------
GLOBAL = load_global_bkt_params()
_MASTERY: Dict[Tuple[str, str], float] = {}


def get_mastery(student_id: str, kc_id: str) -> float:
    # still store per-KC mastery, but with global transition/guess/slip
    return float(_MASTERY.get((student_id, str(kc_id)), GLOBAL.p_init))


def update_mastery(student_id: str, kc_id: str, y_correct: int) -> float:
    prior = float(_MASTERY.get((student_id, str(kc_id)), GLOBAL.p_init))
    nxt = bkt_update(prior, int(y_correct), GLOBAL)
    _MASTERY[(student_id, str(kc_id))] = nxt
    return nxt
