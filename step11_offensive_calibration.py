"""Item 3 - Try to improve the weak 'offensive' class WITHOUT retraining.

Approach: posterior re-weighting (threshold calibration). We scale the cached
class posteriors by per-class weights and take argmax. Weights are TUNED ON THE
VALIDATION SET (to avoid peeking at test) and then applied to the test set. We
report two principled choices:
  (a) inverse-class-frequency weights (parameter-free), and
  (b) the val-tuned weight grid that maximises validation macro-F1.
If neither beats the uncalibrated baseline on test, we report the negative result.

Outputs: results/bert_metrics_weighted.json
"""
import json
import os
import itertools

import numpy as np
import pandas as pd

from config import VAL_CSV, TEST_CSV, BEST_BERT_DIR, RESULT_DIR, LABELS
from modeling import make_predict_proba
from utils import classification_report_dict


def metrics_with_weights(proba, y_true, w):
    pred = (proba * w).argmax(1)
    return classification_report_dict(y_true, pred, proba), pred


def main():
    val = pd.read_csv(VAL_CSV); test = pd.read_csv(TEST_CSV)
    yv = val.label_id.to_numpy(); yt = test.label_id.to_numpy()

    pp = make_predict_proba(BEST_BERT_DIR)
    print("[calib] scoring validation + test (inference only) ...")
    pv = pp(list(val.text)); pt = pp(list(test.text))

    # baseline (uniform weights) on test
    base_test, _ = metrics_with_weights(pt, yt, np.ones(3))

    # (a) inverse class frequency from TRAIN distribution (use val as proxy counts)
    counts = np.array([(yv == i).sum() for i in range(3)], dtype=float)
    inv = counts.sum() / (len(counts) * counts)          # inverse-frequency, mean~1
    inv_val, _ = metrics_with_weights(pv, yv, inv)
    inv_test, _ = metrics_with_weights(pt, yt, inv)

    # (b) grid search weights, maximise VAL macro-F1
    grid = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5]
    best_w, best_vf1 = np.ones(3), -1.0
    for wh, wo, wn in itertools.product(grid, repeat=3):
        w = np.array([wh, wo, wn])
        m, _ = metrics_with_weights(pv, yv, w)
        if m["macro_f1"] > best_vf1:
            best_vf1, best_w = m["macro_f1"], w
    tuned_val, _ = metrics_with_weights(pv, yv, best_w)
    tuned_test, _ = metrics_with_weights(pt, yt, best_w)

    def slim(m):
        return {k: round(m[k], 4) for k in ("accuracy", "macro_f1", "f1_hate", "f1_offensive", "f1_normal", "auc")}

    out = {
        "note": "Posterior re-weighting (no retraining). Weights tuned on validation, applied to test.",
        "class_order": LABELS,
        "baseline_test": slim(base_test),
        "inverse_freq": {"weights": [round(x, 3) for x in inv],
                          "val": slim(inv_val), "test": slim(inv_test)},
        "val_tuned": {"weights": [round(x, 3) for x in best_w],
                       "val": slim(tuned_val), "test": slim(tuned_test)},
    }

    # verdict: did the val-tuned scheme improve TEST macro-F1 and offensive-F1?
    d_macro = tuned_test["macro_f1"] - base_test["macro_f1"]
    d_off = tuned_test["f1_offensive"] - base_test["f1_offensive"]
    out["verdict"] = {
        "delta_macro_f1": round(d_macro, 4),
        "delta_offensive_f1": round(d_off, 4),
        "improves": bool(d_macro > 0.001),
    }
    with open(os.path.join(RESULT_DIR, "bert_metrics_weighted.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n===== ITEM 3: OFFENSIVE-CLASS CALIBRATION (no retrain) =====")
    print(f"  baseline test : macroF1={base_test['macro_f1']:.4f}  offF1={base_test['f1_offensive']:.4f}")
    print(f"  inverse-freq  : weights={[round(x,2) for x in inv]}  "
          f"test macroF1={inv_test['macro_f1']:.4f}  offF1={inv_test['f1_offensive']:.4f}")
    print(f"  val-tuned     : weights={[round(x,2) for x in best_w]}  "
          f"val macroF1={tuned_val['macro_f1']:.4f} -> test macroF1={tuned_test['macro_f1']:.4f}  "
          f"offF1={tuned_test['f1_offensive']:.4f}")
    print(f"  DELTA on test : macroF1 {d_macro:+.4f}  offensiveF1 {d_off:+.4f}  "
          f"-> {'IMPROVES' if out['verdict']['improves'] else 'NO IMPROVEMENT (cite as future work)'}")
    print("Saved: results/bert_metrics_weighted.json")


if __name__ == "__main__":
    main()
