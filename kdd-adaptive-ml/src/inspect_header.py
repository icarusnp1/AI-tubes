import pandas as pd

INPUT_FILE = "data/raw/kddb-raw"   # kalau filenya tidak di folder kerja, ganti jadi path lengkap
df0 = pd.read_csv(INPUT_FILE, sep="\t", nrows=0)
print("Columns count:", len(df0.columns))
for c in df0.columns:
    print(repr(c))
