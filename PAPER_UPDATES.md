# PAPER UPDATES — XHateExplain (strengthened results)

## 1. Explainability metrics with 95% bootstrap CIs (10,000 resamples, seed 42)
IOU & Token-F1 over 300 posts with human rationales; AOPC over all 300 posts.

| Method | Plausibility (IOU) | Token-F1 | AOPC@5 | AOPC@10 |
|---|---|---|---|---|
| SHAP | 0.1783 [0.1544, 0.2038] | 0.2523 [0.2207, 0.2835] | 0.4670 [0.4341, 0.4998] | 0.4831 [0.4473, 0.5188] |
| LIME | 0.1730 [0.1498, 0.1979] | 0.2485 [0.2186, 0.2789] | 0.4340 [0.3999, 0.4684] | 0.4410 [0.4031, 0.4780] |
| Attention | 0.1863 [0.1623, 0.2119] | 0.2643 [0.2334, 0.2955] | 0.2906 [0.2449, 0.3349] | 0.3323 [0.2853, 0.3776] |
| XHateExplain | 0.1863 [0.1609, 0.2120] | 0.2625 [0.2314, 0.2950] | 0.4474 [0.4132, 0.4814] | 0.4569 [0.4201, 0.4937] |

**Paired bootstrap significance tests:**

| Comparison | Metric | Mean diff | 95% CI of diff | p | Verdict |
|---|---|---|---|---|---|
| XHateExplain − Attention | IOU | +0.0000 | [-0.0115, +0.0109] | 0.9902 | n.s. (tie) |
| XHateExplain − Attention | Token-F1 | -0.0019 | [-0.0157, +0.0116] | 0.8032 | n.s. (tie) |
| XHateExplain − SHAP | AOPC@5 | -0.0196 | [-0.0290, -0.0104] | 0.0000 | **significant** |
| XHateExplain − SHAP | AOPC@10 | -0.0262 | [-0.0367, -0.0165] | 0.0000 | **significant** |
| Attention − SHAP | AOPC@5 | -0.1764 | [-0.2043, -0.1495] | 0.0000 | **significant** |

**Verdicts (data-driven).** XHateExplain's plausibility (IOU) versus Attention is **not significant (a tie)** (Δ=+0.0000, p=0.99) — the two share the top plausibility. On faithfulness, XHateExplain is **significantly below SHAP at AOPC@5** (Δ=-0.0196, p=0.000) and **significantly below SHAP at AOPC@10** (Δ=-0.0262, p=0.000); the gaps are small (~0.02 absolute). Attention is **significantly below SHAP** on AOPC@5 (Δ=-0.1764, p=0.000). Net interpretation: SHAP is the single most faithful method and Attention is the least faithful; Attention and XHateExplain are tied for the most plausible. XHateExplain is the only method that is simultaneously top-tier on plausibility AND recovers most of SHAP's faithfulness (AOPC@5 0.4474 vs SHAP 0.4670, far above Attention's 0.2906) — its contribution is the balance across both axes, not beating the best single method on either axis individually.

## 2. Community-graph robustness sweep (same 2,000-post sample)

| Cosine thr | Edges | Avg degree | #Communities | Modularity Q | Hate-comm >70% | Normal-comm <10% |
|---|---|---|---|---|---|---|
| 0.7 ✓ | 224,081 | 224.1 | 26 | 0.5625 | 7 | 14 |
| 0.8 | 72,602 | 72.6 | 337 | 0.4605 | 64 | 269 |
| 0.85 | 33,316 | 33.3 | 864 | 0.3309 | 155 | 702 |

**Chosen threshold: 0.7** — it gives the **highest modularity** (Q=0.5625) among the three settings. Critically, raising the threshold does NOT clean up the structure: modularity *decreases* across thresholds (Q: 0.5625 → 0.4605 → 0.3309) and the graph *fragments* into ever more, mostly singleton, communities (#communities: 26 → 337 → 864). The high hate>70% / normal<10% community counts at 0.80–0.85 are therefore an artifact of fragmentation (hundreds of tiny components that are trivially pure), not genuinely cleaner separation. The dense 0.70 graph (avg degree 224.1) is the only setting with a non-fragmented, modular community structure, so it remains the operating point.

**Sentiment gap (holds):** hate-heavy mean sentiment -0.358 vs normal-heavy -0.220 (gap +0.138).

**Honest framing:** HateXplain has no author IDs, so this is *content-similarity clustering for hate-profile characterization*, not social-network community detection. The data supports characterizing what distinguishes hateful clusters (vocabulary via LDA, negative VADER sentiment, dense intra-cluster similarity) — not claims about real user communities or information flow.

## 3. Offensive-class calibration (no retraining)

Posterior re-weighting; weights tuned on validation, applied to test.

| Scheme | Weights (hate,off,norm) | Macro-F1 | Offensive-F1 | Hate-F1 | Normal-F1 | Acc |
|---|---|---|---|---|---|---|
| Baseline (uncalibrated) | (1, 1, 1) | 0.6831 | 0.5392 | 0.7671 | 0.7431 | 0.6932 |
| Inverse-frequency | (1.081, 1.17, 0.82) | 0.6811 | 0.5536 | 0.7592 | 0.7304 | 0.6849 |
| **Val-tuned** | (1.0, 1.8, 2.0) | **0.6828** | **0.5534** | 0.7491 | 0.7460 | 0.6906 |

**Outcome: NEGATIVE (no net gain) — cite as future work.** Tuning class weights on validation to maximise macro-F1 raises *validation* macro-F1 to 0.7017, but the gain does **not transfer to test**: test macro-F1 changes by -0.0003 (0.6831→0.6828) — essentially flat. Offensive-F1 does improve (+0.0141, 0.5392→0.5534), but at the cost of hate-F1 (0.7671→0.7491), so the macro average is unchanged. AUC is unchanged (re-weighting is ranking-invariant). Conclusion: cheap posterior re-weighting cannot fix the offensive class without hurting another; a proper remedy (class-weighted/focal-loss *re-training*, or richer features) is left as future work.

## New / updated files
**results/**
- explainability_ci.json
- table2_explainability_ci.csv
- community_threshold_sweep.csv
- community_sweep_summary.json
- community_profiles.csv (regenerated at chosen threshold 0.7)
- community_meta.json (updated; chosen threshold 0.7, Q=0.5625)
- bert_metrics_weighted.json

**figures/**
- table2_explainability.png (regenerated: mean ± 95% CI)
- community_graph_0.85.png (sparsest setting, for the appendix)
- community_graph_best.png (chosen threshold 0.7)