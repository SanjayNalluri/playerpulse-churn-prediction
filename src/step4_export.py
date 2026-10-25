"""
Step 4 - Export the trained model in a light form for the API.

The API should not need scikit-learn or pandas (they are large for a serverless function), so we save:
  models/booster.json        the XGBoost trees (or logistic weights inside serving.json)
  models/serving.json        feature order, scaling, class list, threshold, training ranges, metrics
  models/parity_sample.csv   200 test rows with the full pipeline's probability, so the tests can
                             prove the API gives the same answer as the trained model
"""
import json
import joblib
import numpy as np
import pandas as pd
import common as C

pipe = joblib.load(C.MODELS / "model.joblib")
meta = json.loads((C.MODELS / "metadata.json").read_text())
pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
scaler, ohe = pre.named_transformers_["num"], pre.named_transformers_["cat"]

df = C.load()
train = df[df.month.isin(C.TRAIN_MONTHS)]
ranges = {c: [float(train[c].quantile(0.001)), float(train[c].quantile(0.999))] for c in C.NUMERIC}

serving = {
    "model": meta["model"], "imbalance": meta["imbalance"], "threshold": meta["threshold"],
    "numeric": C.NUMERIC, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
    "categorical": C.CATEGORICAL, "categories": [list(map(str, c)) for c in ohe.categories_],
    "train_ranges": ranges, "train_months": C.TRAIN_MONTHS, "min_hours": C.MIN_HOURS,
    "test_metrics": meta["test"], "train_churn_rate": meta["train_churn_rate"],
}
if meta["model"] == "xgboost":
    clf.get_booster().save_model(str(C.MODELS / "booster.json"))
else:
    serving["coef"], serving["intercept"] = clf.coef_[0].tolist(), float(clf.intercept_[0])

fair = pd.read_csv(C.OUTPUTS / "fairness_by_group.csv")
serving["fairness"] = fair.to_dict(orient="records")
(C.MODELS / "serving.json").write_text(json.dumps(serving, indent=2))

test = df[df.month.isin(C.TEST_MONTHS)].sample(200, random_state=7)
sample = test[C.NUMERIC + C.CATEGORICAL].copy()
sample["p_pipeline"] = pipe.predict_proba(test[C.NUMERIC + C.CATEGORICAL])[:, 1]
sample.to_csv(C.MODELS / "parity_sample.csv", index=False)
print(f"Exported {meta['model']} model, threshold {meta['threshold']}, "
      f"{len(C.NUMERIC)} numeric + {sum(len(c) for c in ohe.categories_)} one-hot features")
print("Files: models/serving.json, models/parity_sample.csv" + (", models/booster.json" if meta["model"] == "xgboost" else ""))
