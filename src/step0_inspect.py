"""
Step 0 - Look at the raw files before building anything.
Prints the columns, the first rows, row counts and the date range.
Run from the PlayerPulse folder:  python src/step0_inspect.py
"""
from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

for name in ["wowah_data.csv", "zones.csv"]:
    f = RAW / name
    print(f"\n=== {name}  ({f.stat().st_size / 1e6:.1f} MB) ===")
    head = pd.read_csv(f, nrows=5, encoding="latin-1")
    print("columns:", list(head.columns))
    print(head.to_string())

# Full pass over the big file, in chunks, to get counts and the date range
f = RAW / "wowah_data.csv"
rows, chars, tmin, tmax = 0, set(), None, None
first = pd.read_csv(f, nrows=1, encoding="latin-1")
cols = [c.strip() for c in first.columns]
ts_col = [c for c in cols if "time" in c.lower()][0]
char_col = [c for c in cols if "char" in c.lower() and "class" not in c.lower()][0]
for chunk in pd.read_csv(f, chunksize=1_000_000, encoding="latin-1"):
    chunk.columns = [c.strip() for c in chunk.columns]
    rows += len(chunk)
    chars.update(chunk[char_col].unique())
    t = pd.to_datetime(chunk[ts_col], errors="coerce")
    tmin = t.min() if tmin is None else min(tmin, t.min())
    tmax = t.max() if tmax is None else max(tmax, t.max())
print(f"\nwowah_data.csv: {rows:,} rows, {len(chars):,} characters, {tmin} to {tmax}")
