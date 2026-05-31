"""Step 3 - SHAP word-level explanations for the fine-tuned BERT model.

Computes per-token SHAP importance (toward the predicted class) aligned to the
HateXplain tokens, then plausibility (IOU / token-F1 vs human rationales) and
faithfulness (AOPC). Saves results to CSV.
"""
import numpy as np
import shap

from config import BEST_BERT_DIR, LABELS
from modeling import make_predict_proba
from explain_common import get_subsample, score_and_save


def align_to_tokens(words, values, n_tokens):
    """Map SHAP whitespace-words back onto the n_tokens positions (1:1 expected)."""
    imp = np.zeros(n_tokens, dtype=float)
    m = min(len(values), n_tokens)
    imp[:m] = values[:m]
    return imp


def main():
    predict_proba = make_predict_proba(BEST_BERT_DIR)
    masker = shap.maskers.Text(r"\s+")
    explainer = shap.Explainer(predict_proba, masker, output_names=LABELS)

    records = get_subsample()
    texts = [r["text"] for r in records]
    preds = list(predict_proba(texts).argmax(1))
    print(f"[SHAP] explaining {len(records)} test posts ...")

    importances = []
    for i, (rec, pred) in enumerate(zip(records, preds)):
        sv = explainer([rec["text"]], max_evals=300, silent=True)
        vals = np.asarray(sv.values[0])          # (n_words, C)
        cls_vals = vals[:, pred] if vals.ndim == 2 else vals
        importances.append(align_to_tokens(sv.data[0], cls_vals, len(rec["tokens"])))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(records)}")

    score_and_save("SHAP", records, importances, predict_proba, preds)


if __name__ == "__main__":
    main()
