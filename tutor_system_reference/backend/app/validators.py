from __future__ import annotations
from typing import Any, Dict
import re

def basic_math_sanity(item: Dict[str, Any]) -> tuple[bool, str]:
    # Minimal sanity check: must have non-empty fields and at least 1 solution step.
    required = ["problem_statement","correct_answer","solution_steps","kc_alignment_explanation"]
    for k in required:
        if k not in item or not item[k]:
            return False, f"Missing or empty field: {k}"
    if not isinstance(item["solution_steps"], list) or len(item["solution_steps"]) < 1:
        return False, "solution_steps must be a non-empty list"
    return True, "ok"

def enforce_constraints(item: Dict[str, Any], constraints: Dict[str, Any]) -> tuple[bool, str]:
    # Example constraints: integer_only answers
    if constraints.get("integer_only"):
        # crude integer pattern
        if not re.fullmatch(r"-?\d+", str(item.get("correct_answer","")).strip()):
            return False, "Answer is not an integer as required."
    return True, "ok"
