import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app, S

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)

GOOD = {"charclass": "Mage", "level_end": 70, "levels_gained": 0, "hours_played": 22.5, "active_days": 9,
        "n_zones": 14, "n_guilds": 1, "share_city": 0.3, "share_dungeon": 0.1, "share_battleground": 0.05,
        "share_arena": 0.0, "share_in_guild": 0.9, "days_since_last_seen": 6, "tenure_days": 180,
        "prev_hours": 30, "hours_change": -7.5}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_model_info_has_metrics():
    body = client.get("/model").json()
    assert 0 < body["threshold"] < 1
    assert "pr_auc" in body["test_metrics"]


def test_fairness_endpoint():
    rows = client.get("/fairness").json()
    assert {r["attribute"] for r in rows} == {"race", "charclass", "level_band"}


def test_predict_returns_probability():
    body = client.post("/predict", json=GOOD).json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk"] in {"low", "medium", "high"}
    assert body["will_churn"] == (body["churn_probability"] >= S["threshold"])


@pytest.mark.parametrize("field,value", [("hours_played", -1), ("share_city", 1.5), ("level_end", 0),
                                         ("days_since_last_seen", 40)])
def test_rejects_bad_input(field, value):
    assert client.post("/predict", json={**GOOD, field: value}).status_code == 422


def test_missing_field_rejected():
    bad = dict(GOOD); bad.pop("hours_played")
    assert client.post("/predict", json=bad).status_code == 422


def test_unknown_class_is_flagged():
    body = client.post("/predict", json={**GOOD, "charclass": "Monk"}).json()
    assert any("not in the training data" in w for w in body["warnings"])


def test_casual_player_is_flagged():
    body = client.post("/predict", json={**GOOD, "hours_played": 1}).json()
    assert any("5+ hours" in w for w in body["warnings"])


def test_batch():
    r = client.post("/predict/batch", json=[GOOD, {**GOOD, "days_since_last_seen": 25}])
    assert r.status_code == 200 and len(r.json()) == 2


def test_batch_limits():
    assert client.post("/predict/batch", json=[]).status_code == 400
    assert client.post("/predict/batch", json=[GOOD] * 501).status_code == 400


def test_api_matches_trained_pipeline():
    """The light API must give the same probabilities as the full scikit-learn pipeline."""
    with open(ROOT / "models" / "parity_sample.csv") as f:
        rows = list(csv.DictReader(f))
    payload, expected = [], []
    for r in rows:
        expected.append(float(r.pop("p_pipeline")))
        payload.append({k: (r[k] if k == "charclass" else float(r[k])) for k in r})
    for k in ["level_end", "levels_gained", "active_days", "n_zones", "n_guilds", "days_since_last_seen", "tenure_days"]:
        for p in payload:
            p[k] = int(round(p[k]))
    got = [p["churn_probability"] for p in client.post("/predict/batch", json=payload).json()]
    worst = max(abs(a - b) for a, b in zip(got, expected))
    assert worst < 1e-3, worst


def test_root_opens_dashboard():
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/app"
    page = client.get("/app")
    assert page.status_code == 200 and "PlayerPulse" in page.text


def _parity_rows():
    with open(ROOT / "models" / "parity_sample.csv") as f:
        rows = list(csv.DictReader(f))
    ints = ["level_end", "levels_gained", "active_days", "n_zones", "n_guilds", "days_since_last_seen", "tenure_days"]
    out = []
    for r in rows:
        r.pop("p_pipeline")
        d = {k: (r[k] if k == "charclass" else float(r[k])) for k in r}
        for k in ints:
            d[k] = int(round(d[k]))
        out.append(d)
    return out


def test_drift_report():
    body = client.post("/drift", json=_parity_rows()).json()
    assert body["status"] in {"stable", "watch", "retrain"}
    assert set(body["psi"]) == set(S["numeric"])


def test_drift_detects_shift():
    shifted = [{**r, "days_since_last_seen": 25, "hours_played": 6} for r in _parity_rows()]
    assert client.post("/drift", json=shifted).json()["status"] == "retrain"


def test_drift_needs_enough_rows():
    assert client.post("/drift", json=[GOOD] * 10).status_code == 400
