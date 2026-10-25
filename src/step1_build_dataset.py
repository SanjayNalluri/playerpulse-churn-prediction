"""
Step 1 - Turn the raw WoW activity log into one row per player per month.

Input : data/raw/wowah_data.csv   (one row = a character seen online in a 10-minute snapshot, 2008)
        data/raw/zones.csv        (zone type: City, Dungeon, Battleground, Arena, Zone ...)
Output: data/processed/player_months.csv

Label : churned = 1 if the character is not seen at all in the NEXT calendar month.
        December has no next month in the data, so December rows are used only as the
        "next month" for November and get no label.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

print("Reading wowah_data.csv (about 1-2 minutes) ...")
df = pd.read_csv(RAW / "wowah_data.csv", encoding="latin-1", skipinitialspace=True,
                 dtype={"char": "int32", "level": "int16", "race": "category",
                        "charclass": "category", "zone": "category", "guild": "int32"})
df.columns = [c.strip() for c in df.columns]
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%y %H:%M:%S")
df["month"] = df["timestamp"].dt.month.astype("int8")
df["date"] = df["timestamp"].dt.normalize()
print(f"  {len(df):,} snapshots, {df['char'].nunique():,} characters, "
      f"{df['timestamp'].min()} to {df['timestamp'].max()}")

# Zone type for each snapshot (City / Dungeon / Battleground / Arena / Zone ...)
zones = pd.read_csv(RAW / "zones.csv", encoding="latin-1")[["Zone_Name", "Type"]]
ztype = dict(zip(zones["Zone_Name"].str.strip(), zones["Type"].str.strip()))
df["ztype"] = df["zone"].astype(str).str.strip().map(ztype).fillna("Unknown")
for t in ["City", "Dungeon", "Battleground", "Arena"]:
    df[f"is_{t.lower()}"] = (df["ztype"] == t).astype("int8")
df["in_guild"] = (df["guild"] != -1).astype("int8")
unmatched = (df["ztype"] == "Unknown").mean()
print(f"  zone type matched for {100 * (1 - unmatched):.1f}% of snapshots")

print("Building player-month features ...")
g = df.groupby(["char", "month"], observed=True)
pm = g.agg(snapshots=("timestamp", "size"),
           active_days=("date", "nunique"),
           first_seen=("timestamp", "min"),
           last_seen=("timestamp", "max"),
           level_start=("level", "min"),
           level_end=("level", "max"),
           n_zones=("zone", "nunique"),
           n_guilds=("guild", "nunique"),
           share_city=("is_city", "mean"),
           share_dungeon=("is_dungeon", "mean"),
           share_battleground=("is_battleground", "mean"),
           share_arena=("is_arena", "mean"),
           share_in_guild=("in_guild", "mean")).reset_index()
pm["hours_played"] = pm["snapshots"] * 10 / 60
pm["levels_gained"] = pm["level_end"] - pm["level_start"]
month_end = pd.to_datetime("2008-" + pm["month"].astype(str) + "-01") + pd.offsets.MonthEnd(0)
pm["days_since_last_seen"] = (month_end - pm["last_seen"].dt.normalize()).dt.days

# Race and class do not change; take each character's value
rc = df.groupby("char", observed=True).agg(race=("race", "first"), charclass=("charclass", "first"))
pm = pm.merge(rc, on="char", how="left")

# Tenure and previous-month activity
first_ever = df.groupby("char", observed=True)["timestamp"].min().rename("first_ever")
pm = pm.merge(first_ever, on="char", how="left")
pm["tenure_days"] = (month_end - pm["first_ever"].dt.normalize()).dt.days
pm = pm.sort_values(["char", "month"])
prev = pm.groupby("char")[["month", "hours_played"]].shift(1)
pm["prev_hours"] = np.where(prev["month"] == pm["month"] - 1, prev["hours_played"], 0.0)
pm["hours_change"] = pm["hours_played"] - pm["prev_hours"]

# Label: not seen next month
seen = set(zip(pm["char"], pm["month"]))
pm["churned"] = [0 if (c, m + 1) in seen else 1 for c, m in zip(pm["char"], pm["month"])]
pm = pm[pm["month"] <= 11].copy()      # December has no next month to check

keep = ["char", "month", "race", "charclass", "level_end", "levels_gained", "hours_played",
        "active_days", "snapshots", "n_zones", "n_guilds", "share_city", "share_dungeon",
        "share_battleground", "share_arena", "share_in_guild", "days_since_last_seen",
        "tenure_days", "prev_hours", "hours_change", "churned"]
pm[keep].to_csv(OUT / "player_months.csv", index=False)

print(f"\nSaved {len(pm):,} player-months for {pm['char'].nunique():,} characters")
print("\nChurn rate by month (share not seen next month):")
print(pm.groupby("month")["churned"].agg(rows="size", churn_rate="mean").round(3).to_string())
print(f"\nOverall churn rate: {pm['churned'].mean():.3f}")
print("\nChurn rate by hours played this month:")
bins = pd.cut(pm["hours_played"], [0, 1, 5, 20, 50, 1e9], labels=["<1h", "1-5h", "5-20h", "20-50h", "50h+"])
print(pm.groupby(bins, observed=True)["churned"].agg(rows="size", churn_rate="mean").round(3).to_string())
