import pandas as pd

# GANTI path sesuai lokasi file KDD kamu
KDD_FILE = "kddb_raw"
OUT_SAMPLE = "kdd_sample_10k.csv"

# Baca sebagian saja (hemat RAM)
chunks = pd.read_csv(
    KDD_FILE,
    sep="\t",          # KDD biasanya tab-separated
    chunksize=20000,   # baca bertahap
    low_memory=False
)

sample_rows = []

for chunk in chunks:
    sample_rows.append(chunk)
    if sum(len(c) for c in sample_rows) >= 10000:
        break

df_sample = pd.concat(sample_rows).head(10000)

df_sample.to_csv(OUT_SAMPLE, index=False)
print("✅ Sample saved:", OUT_SAMPLE)
