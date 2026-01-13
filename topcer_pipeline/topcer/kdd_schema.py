from __future__ import annotations

# Canonical mapping from common KDD 2010 Bridge to Algebra column names.
# If your CSV uses slightly different headers, adjust this dict.

KDD_COLS = {
    "Anon Student Id": "student_id",
    "Problem Hierarchy": "problem_hierarchy",
    "Problem Name": "problem_name",
    "Problem View": "problem_view",
    "Step Name": "step_name",
    "Step Start Time": "step_start_time",
    "First Transaction Time": "first_tx_time",
    "Correct Transaction Time": "correct_tx_time",
    "Step End Time": "step_end_time",
    "Step Duration (sec)": "step_duration",
    "Correct Step Duration (sec)": "correct_step_duration",
    "Error Step Duration (sec)": "error_step_duration",
    "Correct First Attempt": "correct_first_attempt",
    "Incorrects": "incorrects",
    "Hints": "hints",
    "Corrects": "corrects",
    "KC(SubSkills)": "kc_subskills",
    "Opportunity(SubSkills)": "opp_subskills",
    "KC(KTracedSkills)": "kc_ktraced",
    "Opportunity(KTracedSkills)": "opp_ktraced",
}

# Delimiter used in KDD logs for multi-skill fields
MULTI_DELIM = "~~"
