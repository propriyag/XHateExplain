"""Experiment 2a - Larger explainability evaluation.

Evaluates SHAP, LIME, Attention and the two best fusions (mean_zscore = current-
family best on faithfulness; learned_linear = best on plausibility) on EVERY test
post that carries a human rationale (n reported at runtime). Cached 300-post
attributions are reused where post_ids overlap; attributions for the remaining
posts are computed fresh. Metrics get 95% bootstrap CIs.

Outputs:
  results/explainability_ci_large.json
  results/large_importances_{shap,lime,attention}.json   (cache for reuse)
"""
import json
import os

import numpy as np
import pandas as pd
import torch

from config import (TEST_CSV, BEST_BERT_DIR, LABELS, MAX_LEN, RESULT_DIR, TOPK,
                    AOPC_KS, SEED, get_device)
from utils import iou, token_prf, aopc
from modeling import make_predict_proba
from step9_explain_ci import boot_mean_ci, N_BOOT
from step12_fusion_ablation import minmax, zscore

METHODS = ["shap", "lime", "attention"]


def rationale_records():
    df = pd.read_csv(TEST_CSV)
    recs = []
    for _, r in df.iterrows():
        gold = json.loads(r.rationale_idx)
        if not gold:
            continue
        recs.append({"post_id": r.post_id, "text": r.text,
                     "tokens": json.loads(r.tokens), "label_id": int(r.label_id),
                     "gold": gold})
    return recs


def load_cached():
    """post_id -> {method: imp array} from the cached 300-post importances."""
    cache = {}
    for m in METHODS:
        for rec in json.load(open(os.path.join(RESULT_DIR, f"{m}_importances.json"))):
            cache.setdefault(rec["post_id"], {})[m] = np.asarray(rec["imp"], float)
    return cache


# ----------------------------------------------------------------- fresh attributions
def compute_shap(texts, predict_proba):
    import shap
    masker = shap.maskers.Text(r"\s+")
    explainer = shap.Explainer(predict_proba, masker, output_names=LABELS)
    out = []
    for i, txt in enumerate(texts):
        sv = explainer([txt], max_evals=300, silent=True)
        vals = np.asarray(sv.values[0])
        out.append((sv.data[0], vals))
        if (i + 1) % 50 == 0:
            print(f"    shap {i+1}/{len(texts)}")
    return out


def compute_lime(texts, predict_proba, preds):
    from lime.lime_text import LimeTextExplainer
    ex = LimeTextExplainer(class_names=LABELS, split_expression=r"\s+", bow=False)
    out = []
    for i, (txt, pred) in enumerate(zip(texts, preds)):
        e = ex.explain_instance(txt, predict_proba, num_features=10, labels=(pred,), num_samples=1000)
        out.append(dict(e.as_map()[pred]))
        if (i + 1) % 50 == 0:
            print(f"    lime {i+1}/{len(texts)}")
    return out


def compute_attention(records):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = get_device()
    tok = AutoTokenizer.from_pretrained(BEST_BERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        BEST_BERT_DIR, output_attentions=True).to(device).eval()
    out = []
    for i, rec in enumerate(records):
        toks = rec["tokens"]
        enc = tok(toks, is_split_into_words=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        wid = enc.word_ids(0)
        with torch.no_grad():
            o = model(**{k: v.to(device) for k, v in enc.items()})
        att = o.attentions[-1][0].mean(0)[0].cpu().numpy()
        imp = np.zeros(len(toks)); cnt = np.zeros(len(toks))
        for pos, w in enumerate(wid):
            if w is not None and w < len(toks):
                imp[w] += att[pos]; cnt[w] += 1
        cnt[cnt == 0] = 1
        out.append((imp / cnt, int(o.logits.argmax(-1))))
        if (i + 1) % 100 == 0:
            print(f"    attention {i+1}/{len(records)}")
    return out


def per_post(records, imps, predict_proba):
    iou_a, tf_a, a5, a10 = [], [], [], []
    for rec, imp in zip(records, imps):
        gold, toks = rec["gold"], rec["tokens"]
        topk = list(np.argsort(imp)[::-1][:TOPK]) if len(imp) else []
        iou_a.append(iou(topk, gold)); tf_a.append(token_prf(topk, gold)[2])
        _, curve = aopc(predict_proba, toks, imp, rec["pred"], AOPC_KS) if len(imp) else (np.nan, {})
        a5.append(curve.get(5, np.nan)); a10.append(curve.get(10, np.nan))
    return {"iou": np.array(iou_a), "token_f1": np.array(tf_a),
            "aopc_5": np.array(a5), "aopc_10": np.array(a10)}


def main():
    records = rationale_records()
    n = len(records)
    print(f"[Exp2a] {n} test posts carry a human rationale.")
    predict_proba = make_predict_proba(BEST_BERT_DIR)
    preds = list(predict_proba([r["text"] for r in records]).argmax(1))
    for r, p in zip(records, preds):
        r["pred"] = int(p)
    cache = load_cached()
    n_cached = sum(1 for r in records if r["post_id"] in cache and len(cache[r["post_id"]]) == 3)
    print(f"  reusing cached attributions for {n_cached}/{n} posts; computing {n-n_cached} fresh.")

    # split into cached vs to-compute
    todo = [r for r in records if not (r["post_id"] in cache and len(cache[r["post_id"]]) == 3)]
    todo_texts = [r["text"] for r in todo]
    todo_preds = [r["pred"] for r in todo]

    print("  computing SHAP ..."); shap_raw = compute_shap(todo_texts, predict_proba)
    print("  computing LIME ..."); lime_raw = compute_lime(todo_texts, predict_proba, todo_preds)
    print("  computing attention ..."); attn_raw = compute_attention(todo)

    # assemble full per-method importance arrays aligned to `records`
    imps = {m: [] for m in METHODS}
    ti = 0
    todo_ids = {r["post_id"] for r in todo}
    shap_map = {}; lime_map = {}; attn_map = {}
    for r, (words, vals), lm, (aimp, _) in zip(todo, shap_raw, lime_raw, attn_raw):
        nt = len(r["tokens"])
        s = np.zeros(nt); mm = min(len(vals), nt)
        cls = vals[:, r["pred"]] if vals.ndim == 2 else vals
        s[:mm] = cls[:mm]; shap_map[r["post_id"]] = s
        l = np.zeros(nt)
        for fid, wgt in lm.items():
            if fid < nt: l[fid] = wgt
        lime_map[r["post_id"]] = l
        attn_map[r["post_id"]] = aimp
    for r in records:
        pid = r["post_id"]
        if pid in cache and len(cache[pid]) == 3:
            for m in METHODS: imps[m].append(cache[pid][m])
        else:
            imps["shap"].append(shap_map[pid]); imps["lime"].append(lime_map[pid])
            imps["attention"].append(attn_map[pid])

    # cache the freshly computed (full) importances for future reuse
    for m in METHODS:
        dump = [{"post_id": r["post_id"], "tokens": r["tokens"], "gold_idx": r["gold"],
                 "pred": r["pred"], "label_id": r["label_id"],
                 "imp": [float(x) for x in imps[m][i]]} for i, r in enumerate(records)]
        json.dump(dump, open(os.path.join(RESULT_DIR, f"large_importances_{m}.json"), "w"))

    # fusions
    def fuse_minmax(i): return np.mean([minmax(imps[m][i]) for m in METHODS], axis=0)
    def fuse_zscore(i): return np.mean([zscore(imps[m][i]) for m in METHODS], axis=0)
    w = {"shap": 0.18, "lime": 0.10, "attention": 0.72}   # learned weights from Exp1
    def fuse_learned(i): return np.sum([w[m] * minmax(imps[m][i]) for m in METHODS], axis=0)

    series = {"SHAP": [imps["shap"][i] for i in range(n)],
              "LIME": [imps["lime"][i] for i in range(n)],
              "Attention": [imps["attention"][i] for i in range(n)],
              "XHateExplain(mean_minmax)": [fuse_minmax(i) for i in range(n)],
              "Fusion(mean_zscore)": [fuse_zscore(i) for i in range(n)],
              "Fusion(learned_linear)": [fuse_learned(i) for i in range(n)]}

    rng = np.random.default_rng(SEED)
    print("  scoring all methods (AOPC forward passes) ...")
    cells = {}
    for name, imp_list in series.items():
        pp = per_post(records, imp_list, predict_proba)
        for mk in ["iou", "token_f1", "aopc_5", "aopc_10"]:
            mean, (lo, hi), nn = boot_mean_ci(pp[mk], rng)
            cells[(name, mk)] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        print(f"    {name:26} IOU={cells[(name,'iou')]['mean']:.4f} "
              f"tokF1={cells[(name,'token_f1')]['mean']:.4f} "
              f"AOPC@5={cells[(name,'aopc_5')]['mean']:.4f} AOPC@10={cells[(name,'aopc_10')]['mean']:.4f}")

    out = {"n_posts": n, "n_cached_reused": n_cached, "n_boot": N_BOOT, "seed": SEED,
           "note": "Evaluated only on test posts WITH a human rationale (clean plausibility).",
           "metrics": {f"{name}|{mk}": cells[(name, mk)]
                       for name in series for mk in ["iou", "token_f1", "aopc_5", "aopc_10"]}}
    json.dump(out, open(os.path.join(RESULT_DIR, "explainability_ci_large.json"), "w"), indent=2)
    print(f"\n===== EXP2a DONE (n={n}) ===== saved results/explainability_ci_large.json")


if __name__ == "__main__":
    main()
