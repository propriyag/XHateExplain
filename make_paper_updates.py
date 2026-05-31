"""Consolidate Items 1-3 into a single pasteable PAPER UPDATES markdown block,
read from the saved result JSON/CSV so every number is authoritative."""
import json
import os

import pandas as pd
from config import RESULT_DIR, FIG_DIR, ROOT

ci = json.load(open(os.path.join(RESULT_DIR, "explainability_ci.json")))
sweep = json.load(open(os.path.join(RESULT_DIR, "community_sweep_summary.json")))
sweep_csv = pd.read_csv(os.path.join(RESULT_DIR, "community_threshold_sweep.csv"))
calib = json.load(open(os.path.join(RESULT_DIR, "bert_metrics_weighted.json")))

L = []
def w(s=""): L.append(s)

w("# PAPER UPDATES — XHateExplain (strengthened results)\n")

# ---- Item 1
w("## 1. Explainability metrics with 95% bootstrap CIs "
  f"({ci['n_boot']:,} resamples, seed {ci['seed']})")
w(f"IOU & Token-F1 over {ci['n_with_rationale']} posts with human rationales; "
  f"AOPC over all {ci['n_posts']} posts.\n")
w("| Method | Plausibility (IOU) | Token-F1 | AOPC@5 | AOPC@10 |")
w("|---|---|---|---|---|")
for m in ["SHAP", "LIME", "Attention", "XHateExplain"]:
    cells = []
    for metric in ["iou", "token_f1", "aopc_5", "aopc_10"]:
        c = ci["metrics"][f"{m}|{metric}"]
        cells.append(f"{c['mean']:.4f} [{c['ci95'][0]:.4f}, {c['ci95'][1]:.4f}]")
    w(f"| {m} | " + " | ".join(cells) + " |")
w("")
w("**Paired bootstrap significance tests:**\n")
w("| Comparison | Metric | Mean diff | 95% CI of diff | p | Verdict |")
w("|---|---|---|---|---|---|")
labels = {
    "XHateExplain_vs_Attention__iou": ("XHateExplain − Attention", "IOU"),
    "XHateExplain_vs_Attention__token_f1": ("XHateExplain − Attention", "Token-F1"),
    "XHateExplain_vs_SHAP__aopc_5": ("XHateExplain − SHAP", "AOPC@5"),
    "XHateExplain_vs_SHAP__aopc_10": ("XHateExplain − SHAP", "AOPC@10"),
    "Attention_vs_SHAP__aopc_5": ("Attention − SHAP", "AOPC@5"),
}
for k, t in ci["paired_tests"].items():
    comp, met = labels[k]
    verdict = "**significant**" if t["significant_0.05"] else "n.s. (tie)"
    w(f"| {comp} | {met} | {t['mean_diff']:+.4f} | "
      f"[{t['ci95_diff'][0]:+.4f}, {t['ci95_diff'][1]:+.4f}] | {t['p_value']:.4f} | {verdict} |")
w("")
plaus = ci["paired_tests"]["XHateExplain_vs_Attention__iou"]
f5 = ci["paired_tests"]["XHateExplain_vs_SHAP__aopc_5"]
f10 = ci["paired_tests"]["XHateExplain_vs_SHAP__aopc_10"]
attn_shap = ci["paired_tests"]["Attention_vs_SHAP__aopc_5"]

def sig_word(t):
    return "significant" if t["significant_0.05"] else "not significant (a tie)"

plaus_sentence = (
    f"XHateExplain's plausibility (IOU) versus Attention is **{sig_word(plaus)}** "
    f"(Δ={plaus['mean_diff']:+.4f}, p={plaus['p_value']:.2f})"
    + (" — the two share the top plausibility." if not plaus["significant_0.05"]
       else "."))

# faithfulness vs SHAP: describe each AOPC level honestly from the data
def faith_clause(t, level):
    direction = "below" if t["mean_diff"] < 0 else "above"
    if t["significant_0.05"]:
        return f"**significantly {direction} SHAP at {level}** (Δ={t['mean_diff']:+.4f}, p={t['p_value']:.3f})"
    return f"statistically indistinguishable from SHAP at {level} (Δ={t['mean_diff']:+.4f}, p={t['p_value']:.3f})"

w("**Verdicts (data-driven).** " + plaus_sentence + " On faithfulness, XHateExplain is "
  + faith_clause(f5, "AOPC@5") + " and " + faith_clause(f10, "AOPC@10")
  + f"; the gaps are small (~0.02 absolute). Attention is **significantly below SHAP** on AOPC@5 "
  f"(Δ={attn_shap['mean_diff']:+.4f}, p={attn_shap['p_value']:.3f}). "
  "Net interpretation: SHAP is the single most faithful method and Attention is the least faithful; "
  "Attention and XHateExplain are tied for the most plausible. XHateExplain is the only method that is "
  "simultaneously top-tier on plausibility AND recovers most of SHAP's faithfulness "
  f"(AOPC@5 {ci['metrics']['XHateExplain|aopc_5']['mean']:.4f} vs SHAP "
  f"{ci['metrics']['SHAP|aopc_5']['mean']:.4f}, far above Attention's "
  f"{ci['metrics']['Attention|aopc_5']['mean']:.4f}) — its contribution is the balance across both axes, "
  "not beating the best single method on either axis individually.\n")

# ---- Item 2
w("## 2. Community-graph robustness sweep (same 2,000-post sample)\n")
w("| Cosine thr | Edges | Avg degree | #Communities | Modularity Q | Hate-comm >70% | Normal-comm <10% |")
w("|---|---|---|---|---|---|---|")
for r in sweep_csv.to_dict("records"):
    star = " ✓" if r["threshold"] == sweep["chosen_threshold"] else ""
    w(f"| {r['threshold']}{star} | {r['edges']:,} | {r['avg_degree']} | {r['n_communities']} | "
      f"{r['modularity_Q']:.4f} | {r['hate_comm_gt70pct']} | {r['normal_comm_lt10pct']} |")
w("")
_rows = {r["threshold"]: r for r in sweep["sweep"]}
_chosen = _rows[sweep["chosen_threshold"]]
_qtrend = " → ".join(f"{r['modularity_Q']:.4f}" for r in sweep["sweep"])
_ctrend = " → ".join(f"{r['n_communities']}" for r in sweep["sweep"])
w(f"**Chosen threshold: {sweep['chosen_threshold']}** — it gives the **highest modularity** "
  f"(Q={_chosen['modularity_Q']:.4f}) among the three settings. Critically, raising the threshold does NOT "
  f"clean up the structure: modularity *decreases* across thresholds (Q: {_qtrend}) and the graph "
  f"*fragments* into ever more, mostly singleton, communities (#communities: {_ctrend}). The high "
  f"hate>70% / normal<10% community counts at 0.80–0.85 are therefore an artifact of fragmentation "
  f"(hundreds of tiny components that are trivially pure), not genuinely cleaner separation. The dense "
  f"0.70 graph (avg degree {_chosen['avg_degree']}) is the only setting with a non-fragmented, modular "
  f"community structure, so it remains the operating point.\n")
w(f"**Sentiment gap (holds):** {sweep['sentiment_gap']}.\n")
w("**Honest framing:** HateXplain has no author IDs, so this is *content-similarity clustering for "
  "hate-profile characterization*, not social-network community detection. The data supports characterizing "
  "what distinguishes hateful clusters (vocabulary via LDA, negative VADER sentiment, dense intra-cluster "
  "similarity) — not claims about real user communities or information flow.\n")

# ---- Item 3
bt = calib["baseline_test"]; vt = calib["val_tuned"]["test"]; inv = calib["inverse_freq"]["test"]
v = calib["verdict"]
w("## 3. Offensive-class calibration (no retraining)\n")
w("Posterior re-weighting; weights tuned on validation, applied to test.\n")
w("| Scheme | Weights (hate,off,norm) | Macro-F1 | Offensive-F1 | Hate-F1 | Normal-F1 | Acc |")
w("|---|---|---|---|---|---|---|")
w(f"| Baseline (uncalibrated) | (1, 1, 1) | {bt['macro_f1']:.4f} | {bt['f1_offensive']:.4f} | "
  f"{bt['f1_hate']:.4f} | {bt['f1_normal']:.4f} | {bt['accuracy']:.4f} |")
w(f"| Inverse-frequency | {tuple(calib['inverse_freq']['weights'])} | {inv['macro_f1']:.4f} | "
  f"{inv['f1_offensive']:.4f} | {inv['f1_hate']:.4f} | {inv['f1_normal']:.4f} | {inv['accuracy']:.4f} |")
w(f"| **Val-tuned** | {tuple(calib['val_tuned']['weights'])} | **{vt['macro_f1']:.4f}** | "
  f"**{vt['f1_offensive']:.4f}** | {vt['f1_hate']:.4f} | {vt['f1_normal']:.4f} | {vt['accuracy']:.4f} |")
w("")
_outcome = "POSITIVE" if v["improves"] else "NEGATIVE (no net gain) — cite as future work"
w(f"**Outcome: {_outcome}.** Tuning class weights on validation to maximise macro-F1 raises *validation* "
  f"macro-F1 to {calib['val_tuned']['val']['macro_f1']:.4f}, but the gain does **not transfer to test**: "
  f"test macro-F1 changes by {v['delta_macro_f1']:+.4f} ({bt['macro_f1']:.4f}→{vt['macro_f1']:.4f}) — "
  f"essentially flat. Offensive-F1 does improve ({v['delta_offensive_f1']:+.4f}, "
  f"{bt['f1_offensive']:.4f}→{vt['f1_offensive']:.4f}), but at the cost of hate-F1 "
  f"({bt['f1_hate']:.4f}→{vt['f1_hate']:.4f}), so the macro average is unchanged. AUC is unchanged "
  "(re-weighting is ranking-invariant). Conclusion: cheap posterior re-weighting cannot fix the offensive "
  "class without hurting another; a proper remedy (class-weighted/focal-loss *re-training*, or richer "
  "features) is left as future work.\n")

# ---- file manifest
w("## New / updated files")
_ct = sweep["chosen_threshold"]
new_results = ["explainability_ci.json", "table2_explainability_ci.csv",
               "community_threshold_sweep.csv", "community_sweep_summary.json",
               f"community_profiles.csv (regenerated at chosen threshold {_ct})",
               f"community_meta.json (updated; chosen threshold {_ct}, Q={_chosen['modularity_Q']:.4f})",
               "bert_metrics_weighted.json"]
new_figs = ["table2_explainability.png (regenerated: mean ± 95% CI)",
            "community_graph_0.85.png (sparsest setting, for the appendix)",
            f"community_graph_best.png (chosen threshold {_ct})"]
w("**results/**")
for f in new_results: w(f"- {f}")
w("\n**figures/**")
for f in new_figs: w(f"- {f}")

out = os.path.join(ROOT, "PAPER_UPDATES.md")
open(out, "w").write("\n".join(L))
print("WROTE", out)
