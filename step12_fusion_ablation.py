"""Experiment 1 - Fusion ablation (no retraining).

Builds four fusion variants of SHAP+LIME+attention from the cached 300-post
per-token attributions, scores each with IOU / token-F1 / AOPC@5 / AOPC@10 and
95% bootstrap CIs, runs paired bootstrap tests vs attention and vs SHAP, and
reports the per-method post-normalisation L1 mass (to explain the IOU tie).

Outputs:
  results/fusion_ablation_ci.json
  figures/fusion_ablation.png
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULT_DIR, FIG_DIR, BEST_BERT_DIR, TOPK, AOPC_KS, SEED
from utils import iou, token_prf, aopc
from modeling import make_predict_proba
from step9_explain_ci import boot_mean_ci, paired_boot, N_BOOT

METHODS = ["shap", "lime", "attention"]
VARIANTS = ["mean_minmax", "mean_zscore", "rank_fusion", "learned_linear"]
K0 = 60          # reciprocal-rank-fusion constant
N_FOLDS = 5


def minmax(v):
    v = np.asarray(v, float); lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def zscore(v):
    v = np.asarray(v, float); mu, sd = v.mean(), v.std()
    return (v - mu) / sd if sd > 0 else np.zeros_like(v)


def rrf(vectors):
    """Reciprocal-rank fusion over a list of per-token importance vectors."""
    n = len(vectors[0]); score = np.zeros(n)
    for v in vectors:
        order = np.argsort(np.asarray(v, float))[::-1]      # high importance first
        rank = np.empty(n); rank[order] = np.arange(1, n + 1)
        score += 1.0 / (K0 + rank)
    return score


def load_importances():
    d = {m: json.load(open(os.path.join(RESULT_DIR, f"{m}_importances.json"))) for m in METHODS}
    base = d["shap"]
    recs = [{"post_id": r["post_id"], "tokens": r["tokens"], "gold": r["gold_idx"],
             "pred": int(r["pred"]), "label_id": int(r.get("label_id", 0))} for r in base]
    imps = {m: [np.asarray(r["imp"], float) for r in d[m]] for m in METHODS}
    return recs, imps


def fuse(variant, imps, idx, weights=None):
    """Per-post fused importance vector for one variant."""
    vs = [imps[m][idx] for m in METHODS]
    if variant == "mean_minmax":
        return np.mean([minmax(v) for v in vs], axis=0)
    if variant == "mean_zscore":
        return np.mean([zscore(v) for v in vs], axis=0)
    if variant == "rank_fusion":
        return rrf(vs)
    if variant == "learned_linear":
        w = weights if weights is not None else np.ones(len(METHODS)) / len(METHODS)
        return np.sum([w[k] * minmax(vs[k]) for k in range(len(METHODS))], axis=0)
    raise ValueError(variant)


def per_post_metrics(records, fused_list, predict_proba):
    """Return per-post arrays of iou, token_f1, aopc_5, aopc_10 (NaN where no gold)."""
    iou_a, tf_a, a5, a10 = [], [], [], []
    for rec, imp in zip(records, fused_list):
        toks, gold = rec["tokens"], rec["gold"]
        topk = list(np.argsort(imp)[::-1][:TOPK]) if len(imp) else []
        iou_a.append(iou(topk, gold))
        tf_a.append(token_prf(topk, gold)[2])
        _, curve = aopc(predict_proba, toks, imp, rec["pred"], AOPC_KS) if len(imp) else (np.nan, {})
        a5.append(curve.get(5, np.nan)); a10.append(curve.get(10, np.nan))
    return {"iou": np.array(iou_a), "token_f1": np.array(tf_a),
            "aopc_5": np.array(a5), "aopc_10": np.array(a10)}


def learn_weights_cv(records, imps, predict_proba_noop=None):
    """5-fold CV: per fold pick non-negative simplex weights (grid 0..1 step .1)
    maximising mean token-F1 on the training folds; apply to held-out fold.
    Returns (held_out_fused_per_post, mean_weights_over_folds)."""
    n = len(records)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(n)
    folds = np.array_split(order, N_FOLDS)
    # candidate non-negative weights on the simplex (sum=1), step 0.1
    grid = []
    for a in range(11):
        for b in range(11 - a):
            c = 10 - a - b
            grid.append(np.array([a, b, c]) / 10.0)
    grid = np.array(grid)

    def tokenf1_for(weights, idxs):
        vals = []
        for i in idxs:
            if not records[i]["gold"]:
                continue
            imp = fuse("learned_linear", imps, i, weights)
            topk = list(np.argsort(imp)[::-1][:TOPK])
            vals.append(token_prf(topk, records[i]["gold"])[2])
        return np.mean(vals) if vals else 0.0

    fused = [None] * n
    chosen = []
    for f in range(N_FOLDS):
        test_idx = list(folds[f])
        train_idx = [i for i in order if i not in set(test_idx)]
        scores = [tokenf1_for(w, train_idx) for w in grid]
        w_best = grid[int(np.argmax(scores))]
        chosen.append(w_best)
        for i in test_idx:
            fused[i] = fuse("learned_linear", imps, i, w_best)
    return fused, np.mean(chosen, axis=0)


def main():
    records, imps = load_importances()
    predict_proba = make_predict_proba(BEST_BERT_DIR)
    rng = np.random.default_rng(SEED)
    print(f"[Exp1] fusion ablation on {len(records)} posts ...")

    # ---- L1 mass after min-max normalisation (per method) ----
    l1 = {}
    for m in METHODS:
        masses = [np.sum(np.abs(minmax(imps[m][i]))) for i in range(len(records))]
        l1[m] = round(float(np.mean(masses)), 4)
    print("  post-minmax L1 mass:", l1)

    # ---- learned weights via CV ----
    print("  learning linear fusion weights (5-fold CV) ...")
    learned_fused, learned_w = learn_weights_cv(records, imps)
    learned_w = {m: round(float(w), 4) for m, w in zip(METHODS, learned_w)}
    print("  learned weights:", learned_w)

    # ---- build per-post fused vectors for each variant ----
    fused_by_variant = {}
    for var in VARIANTS:
        if var == "learned_linear":
            fused_by_variant[var] = learned_fused
        else:
            fused_by_variant[var] = [fuse(var, imps, i) for i in range(len(records))]

    # ---- per-post metrics for variants ----
    print("  scoring variants (AOPC needs model forward passes) ...")
    variant_metrics = {}
    for var in VARIANTS:
        variant_metrics[var] = per_post_metrics(records, fused_by_variant[var], predict_proba)
        print(f"    {var:14} done")

    # ---- baseline per-post metrics (attention, shap) from cached explanations ----
    base_pp = {}
    for m in ["attention", "shap"]:
        df = pd.read_csv(os.path.join(RESULT_DIR, f"{m}_explanations.csv"))
        base_pp[m] = {k: df[k].to_numpy() for k in ["iou", "token_f1", "aopc_5", "aopc_10"]}

    # ---- means + bootstrap CIs ----
    METRIC_KEYS = ["iou", "token_f1", "aopc_5", "aopc_10"]
    cells = {}
    for var in VARIANTS:
        for mk in METRIC_KEYS:
            mean, (lo, hi), nn = boot_mean_ci(variant_metrics[var][mk], rng)
            cells[(var, mk)] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": nn}
    # include attention + shap reference means
    for m in ["attention", "shap"]:
        for mk in METRIC_KEYS:
            mean, (lo, hi), nn = boot_mean_ci(base_pp[m][mk], rng)
            cells[(m, mk)] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": nn}

    # ---- paired tests: each variant vs attention & vs shap, on IOU and AOPC@5 ----
    tests = {}
    for var in VARIANTS:
        for ref in ["attention", "shap"]:
            for mk in ["iou", "aopc_5"]:
                diff, (lo, hi), p, npair = paired_boot(variant_metrics[var][mk], base_pp[ref][mk], rng)
                sig = bool(not (lo <= 0 <= hi))
                tests[f"{var}_vs_{ref}__{mk}"] = {
                    "mean_diff": round(diff, 4), "ci95_diff": [round(lo, 4), round(hi, 4)],
                    "p_value": round(p, 4), "significant_0.05": sig, "n_paired": npair}

    # ---- compare each variant vs current mean_minmax on iou & aopc_5 ----
    vs_current = {}
    for var in ["mean_zscore", "rank_fusion", "learned_linear"]:
        for mk in ["iou", "token_f1", "aopc_5", "aopc_10"]:
            diff, (lo, hi), p, npair = paired_boot(variant_metrics[var][mk],
                                                   variant_metrics["mean_minmax"][mk], rng)
            vs_current[f"{var}_vs_mean_minmax__{mk}"] = {
                "mean_diff": round(diff, 4), "ci95_diff": [round(lo, 4), round(hi, 4)],
                "p_value": round(p, 4), "significant_0.05": bool(not (lo <= 0 <= hi))}

    out = {
        "n_boot": N_BOOT, "seed": SEED, "n_posts": len(records),
        "k0_rrf": K0, "n_folds_learned": N_FOLDS,
        "l1_mass_post_minmax": l1,
        "learned_weights": learned_w,
        "metrics": {f"{v}|{mk}": cells[(v, mk)]
                    for v in VARIANTS + ["attention", "shap"] for mk in METRIC_KEYS},
        "paired_vs_components": tests,
        "paired_vs_current_mean": vs_current,
    }
    with open(os.path.join(RESULT_DIR, "fusion_ablation_ci.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---- figure: bars +/- CI for the 4 variants across 4 metrics ----
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3))
    colors = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad"]
    labels = {"iou": "IOU", "token_f1": "Token-F1", "aopc_5": "AOPC@5", "aopc_10": "AOPC@10"}
    for ax, mk in zip(axes, METRIC_KEYS):
        means = [cells[(v, mk)]["mean"] for v in VARIANTS]
        los = [cells[(v, mk)]["ci95"][0] for v in VARIANTS]
        his = [cells[(v, mk)]["ci95"][1] for v in VARIANTS]
        yerr = np.array([[m - l for m, l in zip(means, los)], [h - m for h, m in zip(his, means)]])
        ax.bar(range(len(VARIANTS)), means, yerr=yerr, capsize=5, color=colors,
               edgecolor="black", linewidth=0.6)
        ax.set_xticks(range(len(VARIANTS)))
        ax.set_xticklabels(VARIANTS, rotation=30, ha="right", fontsize=8)
        ax.set_title(labels[mk], fontweight="bold", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        for i, m in enumerate(means):
            ax.text(i, his[i], f"{m:.4f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Experiment 1 — Fusion variants with 95% bootstrap CIs "
                 f"({N_BOOT:,} resamples, {len(records)} posts)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG_DIR, "fusion_ablation.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- console summary ----
    print("\n===== EXP1: FUSION ABLATION =====")
    hdr = f"{'Variant':14}" + "".join(f"{labels[k]:>26}" for k in METRIC_KEYS)
    print(hdr)
    for v in VARIANTS:
        line = f"{v:14}"
        for mk in METRIC_KEYS:
            c = cells[(v, mk)]
            line += f"{c['mean']:.4f}[{c['ci95'][0]:.4f},{c['ci95'][1]:.4f}]".rjust(26)
        print(line)
    print(f"\n  L1 mass (post min-max): {l1}  -> attention mass "
          f"{'DOMINATES' if l1['attention']==max(l1.values()) else 'does not dominate'}")
    print(f"  learned linear weights: {learned_w}")
    # verdict
    best_iou = max(VARIANTS, key=lambda v: cells[(v, 'iou')]['mean'])
    best_a5 = max(VARIANTS, key=lambda v: cells[(v, 'aopc_5')]['mean'])
    sig_better = [k for k, t in vs_current.items()
                  if t["significant_0.05"] and t["mean_diff"] > 0 and ("iou" in k or "aopc_5" in k)]
    print(f"\n  best IOU variant: {best_iou} ({cells[(best_iou,'iou')]['mean']:.4f}) | "
          f"best AOPC@5 variant: {best_a5} ({cells[(best_a5,'aopc_5')]['mean']:.4f})")
    print(f"  variants significantly > current mean_minmax (IOU/AOPC@5): "
          f"{sig_better if sig_better else 'NONE — simple mean is justified as no worse'}")
    print("Saved: results/fusion_ablation_ci.json, figures/fusion_ablation.png")


if __name__ == "__main__":
    main()
