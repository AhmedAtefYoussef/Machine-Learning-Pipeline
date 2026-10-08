"""E5 (diagnostic, H13): how much headroom does the Phase 5 classifier have? A tree model on the raw columns with the
same label (thresholds from the training rows) gives a reference ROC-AUC next to our logistic model's stored one.
Output: exp/v2/e5_classifier.json
"""
import json

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from exp.v2.lib import TRAIN, VAL, cfg
from src.common import read_artifact
from src.labels import apply_thresholds, fit_thresholds

group_by, q = cfg["p5"]["group_by"], cfg["p5"]["quantile"]
table = fit_thresholds(TRAIN, group_by, q)
y_tr, y_va = apply_thresholds(TRAIN, table, group_by), apply_thresholds(VAL, table, group_by)
COLS = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",
        "ws_lag1", "wet3"]                       # no instant: it would let the trees recognise individual days
out = {"logistic_stored": read_artifact("p5")["metrics"]["roc_auc"]}
for label, kw in (("HGB default", {}), ("HGB 600 x 0.05", dict(max_iter=600, learning_rate=0.05, l2_regularization=1.0))):
    m = HistGradientBoostingClassifier(random_state=0, **kw).fit(TRAIN[COLS], y_tr)
    p = m.predict_proba(VAL[COLS])[:, 1]
    out[label] = {"roc_auc": float(roc_auc_score(y_va, p)), "pr_auc": float(average_precision_score(y_va, p))}
print(json.dumps(out, indent=1))
json.dump(out, open("exp/v2/e5_classifier.json", "w"), indent=1)
