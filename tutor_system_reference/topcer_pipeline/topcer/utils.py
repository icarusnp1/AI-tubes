from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def parse_timestamp(s: object) -> Optional[pd.Timestamp]:
    """Parse timestamps like '2009-02-03 14:11:58.0'."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    if isinstance(s, pd.Timestamp):
        return s
    txt = str(s).strip()
    if txt == "" or txt.lower() == "nan":
        return None
    try:
        return pd.to_datetime(txt, errors="coerce")
    except Exception:
        return None


def split_multi_field(val: object, delim: str) -> List[str]:
    if val is None:
        return []
    if isinstance(val, float) and np.isnan(val):
        return []
    txt = str(val)
    if txt.strip() == "" or txt.lower() == "nan":
        return []
    return [x.strip() for x in txt.split(delim) if x.strip()]


def safe_int(x: object, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return int(float(x))
    except Exception:
        return default


def safe_float(x: object, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default
