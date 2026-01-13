# Adaptive AI Tutor – Backend (FastAPI)

This is a **reference implementation** for:
- Online step inference (`/infer_step`)
- Policy decision (`/policy_decide`)
- Gemini content generation (stub) (`/generate_item`)
- Teacher dashboard data (`/teacher/dashboard`)

## 1) Install
```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Configure paths (connect to your existing pipeline)
Edit `backend/config/policy_config.yaml` and the env vars below.

Recommended environment variables:
- `TOPCER_MODEL_PATH` : path to your `state_seq_model.pt`
- `TOPCER_META_PATH`  : path to your `seq_dataset_meta.json`
- `GEMINI_API_KEY`     : (optional) your Gemini key if you implement the real call

Example:
```bash
export TOPCER_MODEL_PATH="../topcer_pipeline/models/state_seq_model.pt"
export TOPCER_META_PATH="../topcer_pipeline/data/sequences/seq_dataset_meta.json"
```

## 3) Run
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 4) Quick test
```bash
curl -X POST http://localhost:8000/infer_step \
  -H "Content-Type: application/json" \
  -d '{
    "student_id":"S1",
    "kc_id":"KC_12",
    "timestep":10,
    "features":{
      "time_sec":22.5,
      "error_count":1,
      "hint_count":0,
      "streak":-1,
      "bkt_mastery":0.42
    }
  }'
```

## 5) Frontend dashboard
Open `frontend/teacher_dashboard/index.html` in a browser (or serve it).
It calls the backend at `http://localhost:8000` by default; change the `API_BASE` constant in the HTML if needed.
