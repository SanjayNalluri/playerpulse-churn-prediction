"""
Step 3 - Check whether the model treats player groups differently, and reduce the gap.

For each group (race, class, level band) on the test months:
  recall (share of real churners we catch), false-positive rate, precision.
A large recall gap means some groups' churners are missed more often - those players
would get no retention offer.

Mitigation (post-processing, "equal opportunity"): choose a threshold per level band on the
VALIDATION month so each band reaches the same recall as the overall model, then re-check on test.

Outputs: outputs/fairness_by_group.csv, outputs/fairness_mitigation.csv, models/group_thresholds.json
"""
import json
import numpy as np
import pandas as pd
import common as C

meta = json.loads((C.MODELS / "metadata.json").read_text())
T = meta["threshold"]
test = pd.read_csv(C.OUTPUTS / "test_predictions.csv")
valid = pd.read_csv(C.OUTPUTS / "valid_predictions.csv")


def rates(d, thr):
    pred = d.p_churn >= thr
    y = d.churned == 1
    tp, fp = (pred & y).sum(), (pred & ~y).sum()
    return {"rows": len(d), "churn_rate": y.mean(),
            "recall": tp / max(1, y.sum()), "fpr": fp / max(1, (~y).sum()),
            "precision": tp / max(1, pred.sum())}


rows = []
for attr in ["race", "charclass", "level_band"]:
    for g, d in test.groupby(attr):
        if len(d) < 100:
            continue
        rows.append({"attribute": attr, "group": g, **{k: round(float(v), 3) for k, v in rates(d, T).items()}})
fair = pd.DataFrame(rows)
fair.to_csv(C.OUTPUTS / "fairness_by_group.csv", index=False)

print(f"Single threshold {T}. Results on test months by group:\n")
print(fair.to_string(index=False))
gaps = fair.groupby("attribute").agg(recall_gap=("recall", lambda s: s.max() - s.min()),
                                     fpr_gap=("fpr", lambda s: s.max() - s.min())).round(3)
print("\nGap between best and worst group:")
print(gaps.to_string())

# Mitigation: per-level-band threshold that matches the overall validation recall
target = rates(valid, T)["recall"]
group_thr = {}
for g, d in valid.groupby("level_band"):
    if len(d) < 100:          # too few players to set a reliable threshold
        continue
    cands = np.linspace(0.02, 0.98, 97)
    recs = np.array([rates(d, t)["recall"] for t in cands])
    ok = cands[recs >= target]
    group_thr[g] = float(ok.max()) if len(ok) else float(cands.min())
(C.MODELS / "group_thresholds.json").write_text(json.dumps({"attribute": "level_band", "thresholds": group_thr,
                                                           "default": T}, indent=2))

after = []
for g, d in test.groupby("level_band"):
    if len(d) < 100:
        continue
    b, a = rates(d, T), rates(d, group_thr.get(g, T))
    after.append({"level_band": g, "rows": len(d), "threshold_before": T, "threshold_after": round(group_thr.get(g, T), 3),
                  "recall_before": round(b["recall"], 3), "recall_after": round(a["recall"], 3),
                  "fpr_before": round(b["fpr"], 3), "fpr_after": round(a["fpr"], 3)})
mit = pd.DataFrame(after)
mit.to_csv(C.OUTPUTS / "fairness_mitigation.csv", index=False)

thr_row = test["level_band"].map(group_thr).fillna(T)
overall_b, overall_a = rates(test, T), rates(test.assign(p_churn=test.p_churn - thr_row + T), T)
print("\nPer-level-band thresholds (fit on August):")
print(mit.to_string(index=False))
print(f"\nRecall gap across level bands: {mit.recall_before.max() - mit.recall_before.min():.3f} -> "
      f"{mit.recall_after.max() - mit.recall_after.min():.3f}")
print(f"Overall test: recall {overall_b['recall']:.3f} -> {overall_a['recall']:.3f}, "
      f"precision {overall_b['precision']:.3f} -> {overall_a['precision']:.3f}")
