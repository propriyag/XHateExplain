# Prompt — Strengthen the XHateExplain Results

Paste this into Claude (or your coding agent) inside the `/Users/priyu/xhateexplain` project where the `step1..step8` scripts and the `results/` and `figures/` folders live. It only adds analyses; it does NOT re-run BERT fine-tuning.

---

You are working in the existing XHateExplain project (HateXplain, fine-tuned bert-base-uncased). The pipeline already produced `results/` and `figures/`. Do NOT retrain the classifier. Add the three improvements below, save new artifacts, and print a short summary of every new number so I can paste it into the paper. Use the existing 300-post stratified explainability subset and the existing per-token attribution files (`results/{shap,lime,attention,xhateexplain}_importances.json`) — do not recompute SHAP/LIME from scratch.

### 1. Bootstrap confidence intervals + significance on explainability (highest priority)
For each method (SHAP, LIME, Attention, XHateExplain) recompute IOU, token-F1, AOPC@5, AOPC@10 over the 300-post subset, but now:
- Report each metric to **4 decimal places** (the current 3-decimal table makes Attention and XHateExplain look identical at IOU 0.186 — resolve the tie).
- Compute **95% bootstrap confidence intervals** (10,000 resamples over the 300 posts, fixed seed) for every metric/method cell.
- Run **paired bootstrap significance tests**: (a) XHateExplain vs Attention on IOU and token-F1; (b) XHateExplain vs SHAP on AOPC@5 and AOPC@10; (c) Attention vs SHAP on AOPC@5. Report the mean difference, 95% CI of the difference, and a bootstrap p-value for each.
- Save `results/explainability_ci.json` and regenerate `figures/table2_*.png` showing mean ± 95% CI.
- Print a verdict sentence: is XHateExplain's plausibility advantage over Attention statistically real, or a tie? Is its faithfulness statistically indistinguishable from SHAP?

### 2. Community graph robustness + honest reframing (second priority)
The current graph (cosine ≥ 0.7) gives 224,081 edges on 2,000 nodes (avg degree ~224 — very dense, and similar hateful posts cluster by construction). Strengthen it:
- Re-run the community step at cosine thresholds **0.70 (existing), 0.80, and 0.85** on the same 2,000-post sample.
- For each threshold report: edge count, average degree, number of Louvain communities, **modularity Q**, and the hate-ratio separation (e.g., share of communities that are >70% hate vs <10% hate).
- Keep the per-community LDA topics / VADER sentiment / betweenness profile for the chosen best threshold (highest modularity with non-trivial community count).
- Save `results/community_threshold_sweep.csv` and an updated `results/community_profiles.csv` + `figures/community_graph_0.85.png`.
- Print: which threshold gives the cleanest, sparsest separation, and confirm the hate-vs-normal sentiment gap still holds. (I will reframe this module in the paper as *content-similarity clustering for hate-profile characterization*, not social-network community detection, since HateXplain has no author IDs — flag if the data supports anything stronger.)

### 3. (Optional) Offensive-class improvement
The 'offensive' F1 (~0.54) is the weakest spot. WITHOUT retraining from scratch, try a lightweight fix on the existing logits/predictions if cached, or a quick re-fine-tune ONLY if fast: apply class-weighted cross-entropy (inverse class frequency) or threshold calibration, and report whether macro-F1 and offensive-F1 improve. If it helps, save `results/bert_metrics_weighted.json`; if not, report the negative result so I can cite it as future work. Do not let this block items 1 and 2.

### Output
After running, print one consolidated block titled "PAPER UPDATES" containing: the new 4-decimal Table 2 with CIs, the significance verdicts, the threshold-sweep table, and (if attempted) the offensive-class outcome. List every new file written under `results/` and `figures/`.
