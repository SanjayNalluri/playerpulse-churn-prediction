"""
Step 6 - Why this model, and how many players to target.

Part A: validation (Aug) vs test (Sep-Nov) PR-AUC for all six model setups,
        so the model choice can be checked: it was made on validation only.
Part B: a retention team has a budget for N players a month. For the chosen model,
        rank each month's players by churn risk and contact the top 5% / 10% / 20%,
        compared with contacting everyone above the tuned threshold and with random picks.

Reads data/processed/player_months.csv and outputs/test_predictions.csv (from step 2).
Does not touch the saved model.
Outputs: outputs/model_selection.csv, outputs/budget_targeting.csv
"""
import json
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import common as C

# ---------- Part A: model selection ----------
df = C.load()
train = df[df.month.isin(C.TRAIN_MONTHS)]
valid = df[df.month.isin(C.VALID_MONTHS)]
test = df[df.month.isin(C.TEST_MONTHS)]
X = C.NUMERIC + C.CATEGORICAL
spw = (1 - train.churned.mean()) / train.churned.mean()


def make(model, weighting):  # same setup as step 2
    pre = ColumnTransformer([("num", StandardScaler(), C.NUMERIC),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), C.CATEGORICAL)])
    if model == "logistic":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced" if weighting == "class weights" else None)
    else:
        clf = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9,
                            colsample_bytree=0.9, eval_metric="aucpr", random_state=42, n_jobs=-1,
                            scale_pos_weight=spw if weighting == "class weights" else 1.0)
    return Pipeline([("pre", pre), ("clf", clf)])


def undersample(d, seed=42):
    pos, neg = d[d.churned == 1], d[d.churned == 0]
    return pd.concat([pos, neg.sample(len(pos), random_state=seed)])


rows = []
for model in ["logistic", "xgboost"]:
    for weighting in ["none", "class weights", "undersampling"]:
        tr = undersample(train) if weighting == "undersampling" else train
        pipe = make(model, weighting).fit(tr[X], tr.churned)
        rows.append({"model": model, "imbalance": weighting,
                     "valid_pr_auc": round(average_precision_score(valid.churned, pipe.predict_proba(valid[X])[:, 1]), 3),
                     "test_pr_auc": round(average_precision_score(test.churned, pipe.predict_proba(test[X])[:, 1]), 3)})
sel = pd.DataFrame(rows).sort_values("valid_pr_auc", ascending=False)
sel["valid_rank"] = range(1, len(sel) + 1)
sel["test_rank"] = sel.test_pr_auc.rank(ascending=False, method="min").astype(int)
sel.to_csv(C.OUTPUTS / "model_selection.csv", index=False)
print("A. Model selection (chosen on validation, never on test)")
print(sel.to_string(index=False))

# ---------- Part B: budget targeting ----------
tp = pd.read_csv(C.OUTPUTS / "test_predictions.csv")
thr = json.load(open(C.MODELS / "metadata.json"))["threshold"]
base = tp.churned.mean()
out = []


def add(name, picked):
    n, caught = int(picked.sum()), int(tp.churned[picked].sum())
    out.append({"strategy": name, "players_contacted": n,
                "share_of_players": round(n / len(tp), 3),
                "churners_caught": caught,
                "share_of_churners_caught": round(caught / tp.churned.sum(), 3),
                "precision": round(caught / max(n, 1), 3),
                "false_alarms": n - caught,
                "lift_vs_random": round(caught / max(n, 1) / base, 2)})


for k in [0.05, 0.10, 0.20]:
    # each month is its own campaign: rank that month's players, contact the top k%
    rank = tp.groupby("month").p_churn.rank(ascending=False, pct=True)
    add(f"top {int(k*100)}% each month", rank <= k)
add(f"threshold {thr:.2f} (F1-tuned)", tp.p_churn >= thr)
bt = pd.DataFrame(out)
bt.to_csv(C.OUTPUTS / "budget_targeting.csv", index=False)
print(f"\nB. Budget targeting on test months (churn rate {base:.3f}; random picks have precision {base:.3f}, lift 1.0)")
print(bt.to_string(index=False))
