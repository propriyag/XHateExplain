"""Shared scaffolding for the three explainers (SHAP / LIME / attention).

Each explainer produces, per post, a token-level *importance* vector aligned to
the HateXplain `post_tokens`. This module turns those vectors into the common
metrics: Plausibility (IOU + token-F1 vs human rationales) and Faithfulness
(AOPC). Keeping it here guarantees all three methods are scored identically.
"""
import json
import os

import numpy as np
import pandas as pd

from config import TEST_CSV, LABELS, ID2LABEL, TOPK, AOPC_KS, EXPLAIN_SUBSAMPLE, SEED, RESULT_DIR
from utils import iou, token_prf, aopc, save_json


def get_subsample(n=EXPLAIN_SUBSAMPLE):
    """Stratified subsample of the test set (deterministic). n=None -> full test set."""
    df = pd.read_csv(TEST_CSV)
    if n and n < len(df):
        # proportional stratified subsample by label_id (keeps all columns)
        df = (df.groupby("label_id", group_keys=False)
                .sample(frac=n / len(df), random_state=SEED)
                .reset_index(drop=True))
    recs = []
    for _, r in df.iterrows():
        recs.append({
            "post_id": r.post_id,
            "text": r.text,
            "tokens": json.loads(r.tokens),
            "label_id": int(r.label_id),
            "rationale_idx": json.loads(r.rationale_idx),
        })
    return recs


def score_and_save(method_name, records, importances, predict_proba, preds):
    """Compute plausibility + faithfulness for one explainer and write a CSV.

    records      : list of dicts from get_subsample
    importances  : list of np.array, each aligned to records[i]['tokens'],
                   signed importance toward the predicted class
    predict_proba: callable(list[str]) -> [n, C] (for AOPC perturbation)
    preds        : list[int] predicted class per record
    Returns a summary dict (means over the subsample).
    """
    rows = []
    for rec, imp, pred in zip(records, importances, preds):
        toks, gold = rec["tokens"], rec["rationale_idx"]
        imp = np.asarray(imp, dtype=float)
        # top-k most important token indices (by signed importance toward pred)
        topk_idx = list(np.argsort(imp)[::-1][:TOPK]) if len(imp) else []
        iou_v = iou(topk_idx, gold)
        _, _, tf1 = token_prf(topk_idx, gold)
        aopc_mean, curve = aopc(predict_proba, toks, imp, pred, AOPC_KS) if len(imp) else (np.nan, {})
        rows.append({
            "post_id": rec["post_id"], "method": method_name,
            "pred": ID2LABEL[pred], "true": ID2LABEL[rec["label_id"]],
            "top_tokens": " ".join(toks[i] for i in topk_idx if i < len(toks)),
            "top_idx": json.dumps([int(i) for i in topk_idx]),
            "gold_idx": json.dumps(gold),
            "iou": iou_v, "token_f1": tf1,
            "aopc_5": curve.get(5, np.nan), "aopc_10": curve.get(10, np.nan),
            "aopc_mean": aopc_mean,
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULT_DIR, f"{method_name.lower()}_explanations.csv")
    df.to_csv(csv_path, index=False)
    print(f"  saved per-sample explanations -> {csv_path}")

    # persist full per-token importances so the combined XHateExplain explainer
    # can be assembled later (step5) without re-running the explainers.
    dump = [{"post_id": rec["post_id"], "tokens": rec["tokens"],
             "gold_idx": rec["rationale_idx"], "pred": int(pred),
             "label_id": int(rec["label_id"]),
             "imp": [float(x) for x in np.asarray(imp, dtype=float)]}
            for rec, imp, pred in zip(records, importances, preds)]
    with open(os.path.join(RESULT_DIR, f"{method_name.lower()}_importances.json"), "w") as f:
        json.dump(dump, f)

    # summary: plausibility only over posts that HAVE a gold rationale (toxic posts)
    plaus = df.dropna(subset=["iou"])
    summary = {
        "method": method_name,
        "n_samples": len(df),
        "n_with_rationale": len(plaus),
        "plausibility_iou": float(plaus["iou"].mean()),
        "token_f1": float(plaus["token_f1"].mean()),
        "aopc_5": float(df["aopc_5"].mean()),
        "aopc_10": float(df["aopc_10"].mean()),
        "aopc_mean": float(df["aopc_mean"].mean()),
    }
    save_json(summary, os.path.join(RESULT_DIR, f"{method_name.lower()}_summary.json"))
    print(f"  {method_name}: IOU={summary['plausibility_iou']:.3f} "
          f"token-F1={summary['token_f1']:.3f} "
          f"AOPC@5={summary['aopc_5']:.3f} AOPC@10={summary['aopc_10']:.3f}")
    return summary, df


def _minmax(v):
    v = np.asarray(v, dtype=float)
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def combine_explainers(methods=("shap", "lime", "attention"), predict_proba=None):
    """XHateExplain combined explainer: mean of the min-max-normalised per-token
    importances from the three methods, scored with the same metrics."""
    dumps = {}
    for m in methods:
        p = os.path.join(RESULT_DIR, f"{m}_importances.json")
        if not os.path.exists(p):
            print(f"  [combine] missing {p}; skipping XHateExplain combined")
            return None
        dumps[m] = json.load(open(p))
    base = dumps[methods[0]]
    records = [{"post_id": d["post_id"], "text": " ".join(d["tokens"]),
                "tokens": d["tokens"], "label_id": d.get("label_id", 0),
                "rationale_idx": d["gold_idx"]}
               for d in base]
    preds = [d["pred"] for d in base]
    combined = [np.mean([_minmax(dumps[m][i]["imp"]) for m in methods], axis=0)
                for i in range(len(base))]
    if predict_proba is None:
        from modeling import make_predict_proba
        from config import BEST_BERT_DIR
        predict_proba = make_predict_proba(BEST_BERT_DIR)
    return score_and_save("XHateExplain", records, combined, predict_proba, preds)
