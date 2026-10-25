"""
Step 7 - Monitoring and the retraining loop.

A model in production goes stale. This step replays Aug-Nov 2008 month by month, as if the
model had been live, and checks what a monitoring job would have seen:

A. Input drift (available right away, no labels needed)
   PSI (population stability index) of every feature against the training months.
   PSI < 0.10 stable · 0.10-0.25 watch · > 0.25 major shift. Also the share of players
   with a class the model never saw. tenure_days grows every month by design, so it is
   tracked but left out of the retrain rule (otherwise it would fire every month).
B. Performance (labels arrive one month later: churn = not seen next month)
   PR-AUC and recall of the live model per month, against the August (validation) level.
C. Retraining rule and replay
   Retrain when a feature has PSI > 0.25, more than 5% of players have an unseen class,
   or PR-AUC falls more than 0.05 below August.
   Then the replay: each test month is also scored by a model retrained on every month
   whose labels were known by then, to see whether retraining would have helped.

Outputs: outputs/drift_psi.csv, outputs/monthly_monitoring.csv, outputs/retrain_replay.csv,
         models/drift_reference.json (used by the API's POST /drift endpoint)
"""
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import common as C

PSI_WATCH, PSI_MAJOR, PR_DROP, UNSEEN_MAX = 0.10, 0.25, 0.05, 0.05
EXPECTED_TO_GROW = ["tenure_days"]
df = C.load()
train = df[df.month.isin(C.TRAIN_MONTHS)]
meta = json.load(open(C.MODELS / "metadata.json"))
thr = meta["threshold"]


# ---------- A. drift reference from the training months ----------
def edges_for(x):
    e = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
    return e[1:-1].tolist()  # inner cut points; values below/above go to the end bins


def shares(x, cuts):
    idx = np.searchsorted(cuts, x, side="right")
    return np.bincount(idx, minlength=len(cuts) + 1) / len(x)


def psi(ref, cur, eps=1e-4):
    ref, cur = np.clip(ref, eps, None), np.clip(cur, eps, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


reference = {"features": {}, "classes": sorted(train.charclass.unique().tolist()),
             "thresholds": {"psi_watch": PSI_WATCH, "psi_major": PSI_MAJOR, "unseen_class_max": UNSEEN_MAX},
             "not_in_trigger": EXPECTED_TO_GROW}
for c in C.NUMERIC:
    cuts = edges_for(train[c].values)
    reference["features"][c] = {"cuts": cuts, "ref_shares": shares(train[c].values, cuts).tolist()}
(C.MODELS / "drift_reference.json").write_text(json.dumps(reference, indent=1))

psi_rows = []
for m in C.VALID_MONTHS + C.TEST_MONTHS:
    d = df[df.month == m]
    for c, r in reference["features"].items():
        psi_rows.append({"month": m, "feature": c,
                         "psi": round(psi(np.array(r["ref_shares"]), shares(d[c].values, r["cuts"])), 3)})
psi_df = pd.DataFrame(psi_rows)
psi_df.to_csv(C.OUTPUTS / "drift_psi.csv", index=False)

# ---------- B. performance of the live model per month ----------
preds = pd.concat([pd.read_csv(C.OUTPUTS / "valid_predictions.csv"), pd.read_csv(C.OUTPUTS / "test_predictions.csv")])
base_pr = None
mon = []
for m in C.VALID_MONTHS + C.TEST_MONTHS:
    d, p = df[df.month == m], preds[preds.month == m]
    pr_auc = average_precision_score(p.churned, p.p_churn)
    base_pr = pr_auc if base_pr is None else base_pr
    caught = ((p.p_churn >= thr) & (p.churned == 1)).sum() / max(1, p.churned.sum())
    pm = psi_df[(psi_df.month == m) & ~psi_df.feature.isin(EXPECTED_TO_GROW)]
    mp = pm.sort_values("psi", ascending=False).iloc[0]
    unseen = (~d.charclass.isin(reference["classes"])).mean()
    reasons = []
    if mp.psi > PSI_MAJOR:
        reasons.append(f"{mp.feature} PSI {mp.psi:.2f}")
    if unseen > UNSEEN_MAX:
        reasons.append(f"{unseen:.0%} unseen class")
    if base_pr - pr_auc > PR_DROP:
        reasons.append(f"PR-AUC {pr_auc:.2f} vs {base_pr:.2f} in Aug")
    mon.append({"month": m, "players": len(d), "churn_rate": round(p.churned.mean(), 3),
                "max_psi_feature": mp.feature, "max_psi": mp.psi,
                "features_psi_over_0.10": int((pm.psi > PSI_WATCH).sum()),
                "tenure_psi": float(psi_df[(psi_df.month == m) & (psi_df.feature == "tenure_days")].psi.iloc[0]),
                "unseen_class_share": round(unseen, 3),
                "pr_auc": round(pr_auc, 3), "recall": round(caught, 3),
                "retrain": "YES: " + "; ".join(reasons) if reasons else "no"})
mon_df = pd.DataFrame(mon)
mon_df.to_csv(C.OUTPUTS / "monthly_monitoring.csv", index=False)

# ---------- C. retraining replay ----------
# Month m's labels need month m+1, so a model scoring month m can be trained on months <= m-1.
X = C.NUMERIC + C.CATEGORICAL


def make():  # the chosen setup: XGBoost, no weighting (same as step 2)
    pre = ColumnTransformer([("num", StandardScaler(), C.NUMERIC),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), C.CATEGORICAL)])
    clf = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9,
                        colsample_bytree=0.9, eval_metric="aucpr", random_state=42, n_jobs=-1)
    return Pipeline([("pre", pre), ("clf", clf)])


rep = []
for m in C.TEST_MONTHS:
    tr = df[df.month <= m - 1]
    d = df[df.month == m]
    p_new = make().fit(tr[X], tr.churned).predict_proba(d[X])[:, 1]
    p_old = preds[preds.month == m].p_churn
    rep.append({"month": m, "retrained_on": f"months 1-{m-1}",
                "pr_auc_original": round(average_precision_score(preds[preds.month == m].churned, p_old), 3),
                "pr_auc_retrained": round(average_precision_score(d.churned, p_new), 3)})
rep_df = pd.DataFrame(rep)
rep_df["change"] = (rep_df.pr_auc_retrained - rep_df.pr_auc_original).round(3)
rep_df.to_csv(C.OUTPUTS / "retrain_replay.csv", index=False)

pd.set_option("display.width", 200)
print("A+B. Monthly monitoring (Aug = validation, Sep-Nov = test)")
print(mon_df.to_string(index=False))
print("\nTop drifting features per month (PSI)")
top = psi_df.sort_values("psi", ascending=False).groupby("month").head(3).sort_values(["month", "psi"], ascending=[True, False])
print(top.to_string(index=False))
print("\nC. Retraining replay")
print(rep_df.to_string(index=False))
