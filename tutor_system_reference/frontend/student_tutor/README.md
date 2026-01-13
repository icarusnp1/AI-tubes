# Student Tutor (Reference UI)

Minimal student-facing loop:
step event → /infer_step → /policy_decide → action → (optional) /generate_item → UI.

## Run
1) Ensure backend is running:
   - http://localhost:8000/health is OK
2) Open:
   frontend/student_tutor/index.html

## Notes
- Features are placeholders. Replace with your engineered features from TOPCER pipeline.
- bkt_mastery is a placeholder (0.5). Next step is to compute mastery online or query backend.
- AR mode is a placeholder that checks camera permission and falls back to Visual 2D mode.
