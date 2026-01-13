# backend/app/bkt_online.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class BKTParams:
    p_init: float = 0.2
    p_transit: float = 0.1
    p_guess: float = 0.2
    p_slip: float = 0.1


def get_default_params() -> BKTParams:
    return BKTParams()


def load_bkt_params() -> Dict[str, BKTParams]:
    """
    Env override:
      TOPCER_BKT_PARAMS_PATH
    Default path tries:
      project_root/topcer_pipeline/data/processed/bkt_global_params.json
    """
    # Try env
    path = os.getenv("TOPCER_BKT_PARAMS_PATH", "")

    # Default relative guess
    if not path:
        # backend/app/bkt_online.py -> backend/app -> backend -> tutor_system_reference -> project_root
        import pathlib
        here = pathlib.Path(__file__).resolve()
        tutor_root = here.parents[2]        # tutor_system_reference
        project_root = tutor_root.parent    # project_root
        path = str(project_root / "topcer_pipeline" / "data" / "processed" / "bkt_global_params.json")

    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    params: Dict[str, BKTParams] = {}
    for kc_id, obj in raw.items():
        params[str(kc_id)] = BKTParams(
            p_init=float(obj.get("p_init", 0.2)),
            p_transit=float(obj.get("p_transit", 0.1)),
            p_guess=float(obj.get("p_guess", 0.2)),
            p_slip=float(obj.get("p_slip", 0.1)),
        )
    return params


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
# -----------------------------
BKT_PARAMS = load_bkt_params()
_MASTERY: Dict[Tuple[str, str], float] = {}


def get_mastery(student_id: str, kc_id: str) -> float:
    prm = BKT_PARAMS.get(str(kc_id), get_default_params())
    return float(_MASTERY.get((student_id, str(kc_id)), prm.p_init))


def update_mastery(student_id: str, kc_id: str, y_correct: int) -> float:
    prm = BKT_PARAMS.get(str(kc_id), get_default_params())
    prior = float(_MASTERY.get((student_id, str(kc_id)), prm.p_init))
    nxt = bkt_update(prior, int(y_correct), prm)
    _MASTERY[(student_id, str(kc_id))] = nxt
    return nxt
