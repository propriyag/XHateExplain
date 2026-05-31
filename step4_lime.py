"""Step 4 - LIME explanations for the fine-tuned BERT model.

Uses LimeTextExplainer with bow=False so that feature ids correspond directly to
token positions (aligned to HateXplain tokens). Produces plausibility/faithfulness
metrics, HTML visualisations for 5 posts, and a LIME-vs-SHAP comparison.
"""
import json
import os

import numpy as np
from lime.lime_text import LimeTextExplainer

from config import BEST_BERT_DIR, LABELS, RESULT_DIR, FIG_DIR
from modeling import make_predict_proba
from explain_common import get_subsample, score_and_save
from utils import save_json

LIME_SAMPLES = 1000


def main():
    predict_proba = make_predict_proba(BEST_BERT_DIR)
    explainer = LimeTextExplainer(class_names=LABELS, split_expression=r"\s+", bow=False)

    records = get_subsample()
    texts = [r["text"] for r in records]
    preds = list(predict_proba(texts).argmax(1))
    print(f"[LIME] explaining {len(records)} test posts ({LIME_SAMPLES} perturbations each) ...")

    importances, example_exps = [], []
    for i, (rec, pred) in enumerate(zip(records, preds)):
        exp = explainer.explain_instance(rec["text"], predict_proba,
                                         num_features=10, labels=(pred,),
                                         num_samples=LIME_SAMPLES)
        imp = np.zeros(len(rec["tokens"]), dtype=float)
        for fid, w in exp.as_map()[pred]:        # fid = token position (bow=False)
            if fid < len(imp):
                imp[fid] = w
        importances.append(imp)
        if i < 5:
            example_exps.append((rec, pred, exp))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(records)}")

    summary, df = score_and_save("LIME", records, importances, predict_proba, preds)

    # --- HTML visualisations for 5 example posts ---
    for j, (rec, pred, exp) in enumerate(example_exps, 1):
        path = os.path.join(FIG_DIR, f"lime_example_{j}.html")
        exp.save_to_file(path, labels=(pred,))
        print(f"  saved {path}")

    # --- LIME vs SHAP comparison ---
    shap_path = os.path.join(RESULT_DIR, "shap_summary.json")
    if os.path.exists(shap_path):
        shap_s = json.load(open(shap_path))
        cmp = {
            "LIME":  {"plausibility_iou": summary["plausibility_iou"], "token_f1": summary["token_f1"],
                      "aopc_5": summary["aopc_5"], "aopc_10": summary["aopc_10"]},
            "SHAP":  {"plausibility_iou": shap_s["plausibility_iou"], "token_f1": shap_s["token_f1"],
                      "aopc_5": shap_s["aopc_5"], "aopc_10": shap_s["aopc_10"]},
            "better_plausibility": "LIME" if summary["plausibility_iou"] > shap_s["plausibility_iou"] else "SHAP",
            "better_faithfulness": "LIME" if summary["aopc_5"] > shap_s["aopc_5"] else "SHAP",
        }
        save_json(cmp, os.path.join(RESULT_DIR, "lime_vs_shap.json"))
        print(f"  better plausibility: {cmp['better_plausibility']} | "
              f"better faithfulness: {cmp['better_faithfulness']}")
    else:
        print("  (run step3_shap.py first for the LIME-vs-SHAP comparison)")


if __name__ == "__main__":
    main()
