from __future__ import annotations
from typing import Any, Dict
import os
import json
import re

from google import genai

import json, re

def _extract_json(text: str) -> dict:
    text = text.strip()

    # Remove ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()

    # If still contains leading/trailing junk, try to locate first { ... last }
    if "{" in text and "}" in text:
        text = text[text.find("{"):text.rfind("}")+1]

    return json.loads(text)

# === PASTIKAN API KEY TERBACA ===
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY tidak ditemukan. "
        "Set environment variable GEMINI_API_KEY sebelum menjalankan uvicorn."
    )

client = genai.Client(api_key=API_KEY)

DEFAULT_MODEL = "gemini-2.5-flash-lite"

def generate_with_gemini(payload: Dict[str, Any]) -> Dict[str, Any]:
    kc_id = payload.get("kc_id", "KC_UNKNOWN")
    kc_desc = payload.get("kc_description", "Algebra skill")
    diff = payload.get("difficulty_target", 3)
    fmt = payload.get("desired_format", "step_by_step")
    misc = payload.get("misconception_tag", None)
    constraints = payload.get("constraints", {}) or {}

    prompt = f"""
Kamu adalah AI Tutor pembuat soal aljabar.

Buat 1 soal latihan yang:
- Selaras dengan KC berikut
- Mudah dipahami siswa
- Solusinya benar secara matematis

KC_ID: {kc_id}
KC_Description: {kc_desc}
Difficulty (1-5): {diff}
Format: {fmt}
Misconception tag: {misc}

BALAS DALAM JSON SAJA dengan format:
{{
  "problem_statement": "...",
  "correct_answer": "...",
  "solution_steps": ["...", "..."],
  "common_wrong_answers": [{{"answer":"...","feedback":"..."}}],
  "kc_alignment_explanation": "..."
}}
"""

    resp = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
    )

    text = (resp.text or "").strip()

    try:
        return _extract_json(text)
    except Exception:
        raise ValueError(f"Gemini output bukan JSON yang valid:\\n{text}")
