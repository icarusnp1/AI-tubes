# backend/app/logger.py
from __future__ import annotations

import json
import os
import datetime as dt
from typing import Dict, Any, Optional

APP_ROOT = os.path.dirname(os.path.dirname(__file__))
LOG_DIR = os.path.join(APP_ROOT, "logs")
LOG_PATH = os.getenv("EVENT_LOG_PATH", os.path.join(LOG_DIR, "events.jsonl"))

os.makedirs(LOG_DIR, exist_ok=True)


def log_event(event_type: str, payload: Dict[str, Any], *, student_id: Optional[str] = None,
              session_id: Optional[str] = None, kc_id: Optional[str] = None) -> None:
    """
    Append-only JSONL event logger.
    Always includes timestamp + event_type.
    """
    rec = {
        "ts": dt.datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "student_id": student_id,
        "session_id": session_id,
        "kc_id": kc_id,
        "payload": payload,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
