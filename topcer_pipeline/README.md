# Topcer Pipeline (KDD 2010 → Step-level → BKT → GRU/LSTM)

A production-oriented scaffold for your **Topcer** roadmap:

## What this includes
- `scripts/preprocess_kdd.py` → KDD CSV → `steps.parquet` + `kc_steps.parquet`
- `scripts/train_bkt.py` → Global BKT fit + mastery feature export
- `scripts/build_sequences.py` → Student-split + windowing → `.npz` batches
- `scripts/train_seq_model.py` → GRU multi-task (correctness + hint + time-bin) + optional weak-supervised state
- `scripts/infer_online.py` → Example online inference wrapper

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage (end-to-end)
1) Preprocess KDD CSV
```bash
python scripts/preprocess_kdd.py --in_csv path/to/kdd.csv --out_dir data/processed
```

2) Train (or set) Global BKT + add mastery features
```bash
python scripts/train_bkt.py --steps_parquet data/processed/steps.parquet --out_dir data/processed
```

3) Build sequence windows (split by student)
```bash
python scripts/build_sequences.py --steps_parquet data/processed/steps_with_mastery.parquet --out_dir data/processed --window 100 --stride 50
```

4) Train GRU multi-task
```bash
python scripts/train_seq_model.py --data_npz data/processed/seq_dataset.npz --out_dir models
```

## Notes
- Split is **by student** (no leakage).
- KDD `KC(SubSkills)` and `Opportunity(SubSkills)` may contain multi-values separated by `~~`. We keep:
  - `kc_list` in `steps.parquet`
  - exploded `kc_steps.parquet` for KT.
- If KC is missing for a step, we keep the row for sequence modeling but skip KT updates.
