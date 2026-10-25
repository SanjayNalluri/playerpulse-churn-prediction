"""
Step 2 - Train churn models and handle the class imbalance.

Population : engaged players (5+ hours in the month) - about 13% of them quit the next month.
Split by time (no leakage): train Jan-Jul, validate Aug (to pick the threshold), test Sep-Nov.

Compared:
  models    : logistic regression, XGBoost
  imbalance : none | class weights | random undersampling of the majority class
  threshold : 0.5 vs a threshold tuned on the validation month to maximise F1
Main metric: PR-AUC (average precision). Accuracy is misleading when only 13% are positive.

Outputs: outputs/model_comparison.csv, outputs/test_by_month.csv, outputs/feature_importance.csv,
         models/model.joblib, models/metadata.json
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import common as C

df = C.load()
train = df[df.month.isin(C.TRAIN_MONTHS)]
valid = df[df.month.isin(C.VALID_MONTHS)]
test = df[df.month.isin(C.TEST_MONTHS)]
X = C.NUMERIC + C.CATEGORICAL
print(f"Engaged player-months: train {len(train):,}  valid {len(valid):,}  test {len(test):,}")
print(f"Churn rate: train {train.churned.mean():.3f}  valid {valid.churned.mean():.3f}  test {test.churned.mean():.3f}")

pos_rate = train.churned.mean()
spw = (1 - pos_rate) / pos_rate


def make(model, weighting):
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


def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i])


def scores(y, p, t):
    pred = (p >= t).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    top = np.argsort(-p)[: max(1, len(p) // 10)]
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
            "precision": pr, "recall": rc, "f1": f1,
            "top10pct_capture": y.iloc[top].sum() / max(1, y.sum())}


rows, fitted = [], {}
for model in ["logistic", "xgboost"]:
    for weighting in ["none", "class weights", "undersampling"]:
        tr = undersample(train) if weighting == "undersampling" else train
        pipe = make(model, weighting).fit(tr[X], tr.churned)
        pv, pt = pipe.predict_proba(valid[X])[:, 1], pipe.predict_proba(test[X])[:, 1]
        t_best = best_threshold(valid.churned, pv)
        for tname, t in [("0.5", 0.5), ("tuned on Aug", t_best)]:
            s = scores(test.churned, pt, t)
            rows.append({"model": model, "imbalance": weighting, "threshold": tname,
                         "threshold_value": round(t, 3), **{k: round(v, 3) for k, v in s.items()}})
        fitted[(model, weighting)] = (pipe, t_best)
        print(f"  {model:8s} {weighting:14s} test PR-AUC {rows[-1]['pr_auc']:.3f}  "
              f"F1@tuned {rows[-1]['f1']:.3f}  recall@0.5 {rows[-2]['recall']:.3f}")

res = pd.DataFrame(rows)
res.to_csv(C.OUTPUTS / "model_comparison.csv", index=False)

# Pick the best by test PR-AUC is NOT allowed (that would tune on test). Pick on validation PR-AUC.
val_ap = {k: average_precision_score(valid.churned, p.predict_proba(valid[X])[:, 1]) for k, (p, _) in fitted.items()}
best_key = max(val_ap, key=val_ap.get)
pipe, thr = fitted[best_key]
print(f"\nChosen on validation PR-AUC: {best_key[0]} + {best_key[1]} (valid PR-AUC {val_ap[best_key]:.3f}), threshold {thr:.3f}")

pt = pipe.predict_proba(test[X])[:, 1]
by_month = []
for m in C.TEST_MONTHS:
    mask = (test.month == m).values
    s = scores(test.churned[mask], pt[mask], thr)
    by_month.append({"month": m, "rows": int(mask.sum()), "churn_rate": round(test.churned[mask].mean(), 3),
                     **{k: round(v, 3) for k, v in s.items()}})
bm = pd.DataFrame(by_month)
bm.to_csv(C.OUTPUTS / "test_by_month.csv", index=False)

# Feature importance (XGBoost gain or logistic coefficients)
names = pipe.named_steps["pre"].get_feature_names_out()
clf = pipe.named_steps["clf"]
imp = clf.feature_importances_ if hasattr(clf, "feature_importances_") else np.abs(clf.coef_[0])
fi = pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)
fi.to_csv(C.OUTPUTS / "feature_importance.csv", index=False)

test_out = test[["char", "month", "race", "charclass", "level_band", "churned"]].copy()
test_out["p_churn"] = pt
test_out.to_csv(C.OUTPUTS / "test_predictions.csv", index=False)
valid_out = valid[["char", "month", "race", "charclass", "level_band", "churned"]].copy()
valid_out["p_churn"] = pipe.predict_proba(valid[X])[:, 1]
valid_out.to_csv(C.OUTPUTS / "valid_predictions.csv", index=False)

joblib.dump(pipe, C.MODELS / "model.joblib")
meta = {"model": best_key[0], "imbalance": best_key[1], "threshold": round(thr, 4),
        "min_hours": C.MIN_HOURS, "features": X, "train_months": C.TRAIN_MONTHS,
        "valid_months": C.VALID_MONTHS, "test_months": C.TEST_MONTHS,
        "test": {k: round(float(v), 4) for k, v in scores(test.churned, pt, thr).items()},
        "train_churn_rate": round(float(pos_rate), 4)}
(C.MODELS / "metadata.json").write_text(json.dumps(meta, indent=2))

pd.set_option("display.width", 160)
print("\nAll models on the test months (Sep-Nov):")
print(res.to_string(index=False))
print("\nChosen model by test month:")
print(bm.to_string(index=False))
print("\nTop 10 features:")
print(fi.head(10).to_string(index=False))
