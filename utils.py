"""Shared helpers: metrics, rationale handling, plausibility (IOU/token-F1) and
faithfulness (AOPC)."""
import json
import numpy as np

from config import LABELS, LABEL2ID


# --------------------------------------------------------------------------- rationales
def parse_rationale_tokens(rationales, n_tokens):
    """Aggregate per-annotator 0/1 rationale masks into a single majority mask.

    Returns a list of token indices that >=50% of the annotators (who provided a
    rationale) marked as important. Empty list for normal posts / no rationale.
    """
    if not rationales:
        return []
    masks = [r for r in rationales if len(r) == n_tokens]
    if not masks:
        return []
    avg = np.mean(np.array(masks, dtype=float), axis=0)
    return [int(i) for i, v in enumerate(avg) if v >= 0.5]


# --------------------------------------------------------------------------- plausibility
def iou(pred_idx, gold_idx):
    """Intersection-over-union between two sets of token indices."""
    a, b = set(pred_idx), set(gold_idx)
    if not a and not b:
        return float("nan")          # undefined: no rationale to match
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def token_prf(pred_idx, gold_idx):
    """Token-level precision / recall / F1 of predicted vs gold rationale indices."""
    a, b = set(pred_idx), set(gold_idx)
    if not a and not b:
        return (float("nan"),) * 3
    tp = len(a & b)
    prec = tp / len(a) if a else 0.0
    rec  = tp / len(b) if b else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


# --------------------------------------------------------------------------- faithfulness (AOPC)
def aopc(predict_proba, tokens, importance, target_class, ks):
    """Area Over the Perturbation Curve (comprehensiveness flavour).

    Mask the top-k most important tokens (highest `importance`) and measure the
    drop in the model's probability for `target_class`. AOPC@K = mean over
    k=1..K of (p_full - p_masked_k). Higher = more faithful.

    `predict_proba(list_of_texts) -> np.ndarray [n, n_classes]`.
    """
    order = np.argsort(importance)[::-1]          # most important first
    base = predict_proba([" ".join(tokens)])[0, target_class]
    drops, curve = [], {}
    for k in ks:
        keep = [t for i, t in enumerate(tokens) if i not in set(order[:k])]
        masked_text = " ".join(keep) if keep else "[PAD]"
        p = predict_proba([masked_text])[0, target_class]
        d = float(base - p)
        curve[k] = d
        drops.append(d)
    return float(np.mean(drops)), curve


# --------------------------------------------------------------------------- classification metrics
def classification_report_dict(y_true, y_pred, y_proba=None):
    """Accuracy, macro-F1, per-class F1, and (optional) one-vs-rest macro AUC."""
    from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    per = f1_score(y_true, y_pred, average=None, labels=list(range(len(LABELS))))
    for i, lab in enumerate(LABELS):
        out[f"f1_{lab}"] = float(per[i])
    if y_proba is not None:
        try:
            out["auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr",
                                             average="macro", labels=list(range(len(LABELS)))))
        except Exception:
            out["auc"] = float("nan")
    return out


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  saved -> {path}")
