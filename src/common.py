"""Shared settings for PlayerPulse."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "player_months.csv"
MODELS, OUTPUTS = ROOT / "models", ROOT / "outputs"
MODELS.mkdir(exist_ok=True); OUTPUTS.mkdir(exist_ok=True)

# Who we score: engaged players (5+ hours in the month). Casual players who log in for
# a few minutes churn ~60% of the time and are not the players a live-ops team acts on.
MIN_HOURS = 5
TRAIN_MONTHS, VALID_MONTHS, TEST_MONTHS = [1, 2, 3, 4, 5, 6, 7], [8], [9, 10, 11]

NUMERIC = ["level_end", "levels_gained", "hours_played", "active_days", "n_zones", "n_guilds",
           "share_city", "share_dungeon", "share_battleground", "share_arena", "share_in_guild",
           "days_since_last_seen", "tenure_days", "prev_hours", "hours_change"]
CATEGORICAL = ["charclass"]          # race is NOT a model input; it is only used to check fairness
TARGET = "churned"


def level_band(level):
    return pd.cut(level, [0, 59, 69, 70, 80], labels=["1-59", "60-69", "70 (max pre-WotLK)", "71-80"])


def load():
    df = pd.read_csv(DATA)
    df = df[df["hours_played"] >= MIN_HOURS].copy()
    df["level_band"] = level_band(df["level_end"]).astype(str)
    return df
