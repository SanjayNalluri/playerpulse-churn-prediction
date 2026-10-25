"""
PlayerPulse API - churn risk for engaged World of Warcraft players.

POST /predict         one player-month  -> churn probability, risk level, warnings
POST /predict/batch   up to 500 player-months
GET  /model           model type, threshold and test metrics
GET  /fairness        subgroup audit: recall / false-positive rate by race, class and level band
POST /drift           a batch of recent player-months -> feature drift (PSI) against the training data
GET  /app             dashboard: try a player, see model metrics and the subgroup audit
GET  /health
"""
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Literal

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "models" / "serving.json").read_text())
BOOSTER = None
if S["model"] == "xgboost":
    BOOSTER = xgb.Booster()
    BOOSTER.load_model(str(ROOT / "models" / "booster.json"))
MEAN, SCALE = np.array(S["mean"]), np.array(S["scale"])
CLASSES = S["categories"][0]
DRIFT_FILE = ROOT / "models" / "drift_reference.json"
DRIFT = json.loads(DRIFT_FILE.read_text()) if DRIFT_FILE.exists() else None
# dashboard page lives in api/dashboard.py; loaded by path so it works the same locally and on Vercel
_spec = importlib.util.spec_from_file_location("pp_dashboard", Path(__file__).with_name("dashboard.py"))
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
DASHBOARD = _mod.PAGE

log = logging.getLogger("playerpulse")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)

app = FastAPI(title="PlayerPulse", version="1.0",
              description="Predicts which engaged World of Warcraft players will not return next month.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


class PlayerMonth(BaseModel):
    charclass: str = Field(..., examples=["Mage"])
    level_end: int = Field(..., ge=1, le=80, examples=[70])
    levels_gained: int = Field(0, ge=0, le=80)
    hours_played: float = Field(..., ge=0, le=744, examples=[22.5])
    active_days: int = Field(..., ge=0, le=31, examples=[9])
    n_zones: int = Field(..., ge=0, le=500, examples=[14])
    n_guilds: int = Field(1, ge=0, le=50)
    share_city: float = Field(0, ge=0, le=1)
    share_dungeon: float = Field(0, ge=0, le=1)
    share_battleground: float = Field(0, ge=0, le=1)
    share_arena: float = Field(0, ge=0, le=1)
    share_in_guild: float = Field(0, ge=0, le=1)
    days_since_last_seen: int = Field(..., ge=0, le=31, examples=[6])
    tenure_days: int = Field(..., ge=0, le=400, examples=[180])
    prev_hours: float = Field(0, ge=0, le=744)
    hours_change: float = Field(0, ge=-744, le=744)


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    risk: Literal["low", "medium", "high"]
    warnings: List[str]


def features(p: PlayerMonth) -> np.ndarray:
    num = (np.array([getattr(p, c) for c in S["numeric"]], dtype=float) - MEAN) / SCALE
    cat = np.array([1.0 if p.charclass == c else 0.0 for c in CLASSES])
    return np.concatenate([num, cat])


def warnings_for(p: PlayerMonth) -> List[str]:
    w = []
    if p.charclass not in CLASSES:
        w.append(f"class '{p.charclass}' was not in the training data; prediction is less reliable")
    for c, (lo, hi) in S["train_ranges"].items():
        v = getattr(p, c)
        if v < lo or v > hi:
            w.append(f"{c}={v} is outside the training range [{lo:g}, {hi:g}]")
    if p.hours_played < S["min_hours"]:
        w.append(f"model is trained on players with {S['min_hours']}+ hours in the month")
    return w


def score(rows: List[PlayerMonth]) -> List[Prediction]:
    X = np.vstack([features(r) for r in rows])
    if BOOSTER is not None:
        prob = BOOSTER.predict(xgb.DMatrix(X))
    else:
        prob = 1 / (1 + np.exp(-(X @ np.array(S["coef"]) + S["intercept"])))
    t = S["threshold"]
    out = []
    for r, pr in zip(rows, prob):
        pr = float(pr)
        risk = "high" if pr >= max(t, 0.5) else "medium" if pr >= t else "low"
        out.append(Prediction(churn_probability=round(pr, 4), will_churn=pr >= t, risk=risk,
                              warnings=warnings_for(r)))
    return out


@app.get("/", include_in_schema=False)
def root():
    # Visitors to the main link land on the dashboard
    return RedirectResponse(url="/app")


@app.get("/app", include_in_schema=False, response_class=HTMLResponse)
def dashboard():
    return DASHBOARD


@app.get("/health")
def health():
    return {"status": "ok", "model": S["model"], "threshold": S["threshold"]}


@app.get("/model")
def model_info():
    info = {k: S[k] for k in ["model", "imbalance", "threshold", "train_months", "min_hours",
                              "test_metrics", "train_churn_rate"]}
    info["classes"] = CLASSES
    return info


@app.get("/fairness")
def fairness():
    return S["fairness"]


@app.post("/predict", response_model=Prediction)
def predict(p: PlayerMonth):
    t0 = time.perf_counter()
    res = score([p])[0]
    log.info(json.dumps({"event": "predict", "p": res.churn_probability, "warnings": len(res.warnings),
                         "ms": round(1000 * (time.perf_counter() - t0), 2)}))
    return res


@app.post("/predict/batch", response_model=List[Prediction])
def predict_batch(rows: List[PlayerMonth]):
    if not 1 <= len(rows) <= 500:
        raise HTTPException(400, "send between 1 and 500 rows")
    t0 = time.perf_counter()
    res = score(rows)
    log.info(json.dumps({"event": "batch", "rows": len(rows), "ms": round(1000 * (time.perf_counter() - t0), 2)}))
    return res


class DriftReport(BaseModel):
    rows: int
    status: Literal["stable", "watch", "retrain"]
    psi: dict
    drifting_features: List[str]
    unseen_class_share: float
    note: str


def _shares(x, cuts):
    return np.bincount(np.searchsorted(cuts, x, side="right"), minlength=len(cuts) + 1) / len(x)


@app.post("/drift", response_model=DriftReport)
def drift(rows: List[PlayerMonth]):
    """Send a recent batch (e.g. last month's players) and get each feature's PSI against training."""
    if DRIFT is None:
        raise HTTPException(503, "drift reference not available")
    if not 100 <= len(rows) <= 5000:
        raise HTTPException(400, "send between 100 and 5000 rows (PSI needs a reasonable sample)")
    th = DRIFT["thresholds"]
    out = {}
    for c, r in DRIFT["features"].items():
        ref = np.clip(np.array(r["ref_shares"]), 1e-4, None)
        cur = np.clip(_shares(np.array([getattr(p, c) for p in rows], dtype=float), r["cuts"]), 1e-4, None)
        out[c] = round(float(np.sum((cur - ref) * np.log(cur / ref))), 3)
    unseen = sum(p.charclass not in DRIFT["classes"] for p in rows) / len(rows)
    skip = set(DRIFT.get("not_in_trigger", []))  # e.g. tenure grows every month by design
    worst = max(v for c, v in out.items() if c not in skip)
    status = ("retrain" if worst > th["psi_major"] or unseen > th.get("unseen_class_max", 0.05)
              else "watch" if worst > th["psi_watch"] else "stable")
    log.info(json.dumps({"event": "drift", "rows": len(rows), "status": status, "max_psi": worst}))
    return DriftReport(rows=len(rows), status=status, psi=out,
                       drifting_features=sorted([c for c, v in out.items() if v > th["psi_watch"] and c not in skip],
                                                key=lambda c: -out[c]),
                       unseen_class_share=round(unseen, 3),
                       note="PSI < 0.10 stable, 0.10-0.25 watch, > 0.25 or > 5% unseen classes: retrain "
                            "(tenure_days is reported but not used, it grows every month)")
