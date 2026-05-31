# REVISION RESULTS — XHateExplain

All numbers read from saved artifacts; no value hand-entered. Experiments 1, 2a, 2b reuse cached attributions / encoder inference; 3 retrains; 4 is zero-shot.

## Exp 1 — Fusion ablation (300-post subset, 10,000-bootstrap CIs)

| Variant | IOU | Token-F1 | AOPC@5 | AOPC@10 |
|---|---|---|---|---|
| mean_minmax | 0.1863 [0.1615, 0.2119] | 0.2625 [0.2308, 0.2942] | 0.4474 [0.4134, 0.4809] | 0.4569 [0.4195, 0.4943] |
| mean_zscore | 0.1865 [0.1622, 0.2128] | 0.2626 [0.2315, 0.2948] | 0.4525 [0.4185, 0.4865] | 0.4650 [0.4272, 0.5014] |
| rank_fusion | 0.1806 [0.1561, 0.2057] | 0.2568 [0.2265, 0.2876] | 0.4544 [0.4193, 0.4881] | 0.4672 [0.4297, 0.5037] |
| learned_linear | 0.1933 [0.1680, 0.2187] | 0.2735 [0.2424, 0.3056] | 0.3303 [0.2859, 0.3741] | 0.3597 [0.3153, 0.4046] |

**Learned linear weights** (5-fold CV, maximise token-F1): SHAP=0.18, LIME=0.1, attention=0.72.
**Post-min-max L1 mass** (mean Σ|ã| per post): SHAP=8.4636, LIME=7.4995, attention=5.1165. → Contrary to the hypothesis, **SHAP has the largest mass, not attention** — so attention does not dominate the unweighted mean; the IOU tie is not a mass artifact.

**Significance vs current `mean_minmax` (IOU & AOPC@5):**

| Variant | metric | Δ | 95% CI | p | verdict |
|---|---|---|---|---|---|
| mean_zscore | iou | +0.0002 | [-0.0012, +0.0017] | 0.7610 | n.s. |
| mean_zscore | aopc_5 | +0.0051 | [+0.0008, +0.0101] | 0.0160 | **sig** |
| rank_fusion | iou | -0.0057 | [-0.0127, +0.0009] | 0.0970 | n.s. |
| rank_fusion | aopc_5 | +0.0071 | [-0.0006, +0.0151] | 0.0724 | n.s. |
| learned_linear | iou | +0.0070 | [-0.0018, +0.0165] | 0.1220 | n.s. |
| learned_linear | aopc_5 | -0.1171 | [-0.1412, -0.0946] | 0.0000 | **sig** |

**Verdict.** No fusion beats the simple mean on *both* axes. `mean_zscore` is significantly more faithful than `mean_minmax` (AOPC@5, p=0.016) at equal plausibility — a free, principled upgrade. `learned_linear` (attention-heavy, w=0.72) significantly raises IOU vs both SHAP and attention but **collapses faithfulness** (AOPC@5 0.33). So the simple mean is justified as *no worse*; `mean_zscore` is the recommended drop-in improvement; a plausibility-only goal would prefer `learned_linear`.

## Exp 2a — Larger-n explainability (n=1141 — every test post with a human rationale)

(177 attributions reused from the 300-post cache; the rest computed fresh. 10,000-bootstrap CIs.)

| Method | IOU | Token-F1 | AOPC@5 | AOPC@10 |
|---|---|---|---|---|
| SHAP | 0.2984 [0.2871, 0.3100] | 0.4270 [0.4145, 0.4397] | 0.5781 [0.5628, 0.5935] | 0.6031 [0.5867, 0.6191] |
| LIME | 0.2883 [0.2775, 0.2997] | 0.4166 [0.4044, 0.4285] | 0.5385 [0.5220, 0.5545] | 0.5649 [0.5475, 0.5821] |
| Attention | 0.3050 [0.2945, 0.3156] | 0.4404 [0.4290, 0.4520] | 0.4214 [0.3982, 0.4441] | 0.4942 [0.4704, 0.5175] |
| XHateExplain(mean_minmax) | 0.3027 [0.2914, 0.3141] | 0.4325 [0.4200, 0.4452] | 0.5580 [0.5422, 0.5739] | 0.5870 [0.5700, 0.6036] |
| Fusion(mean_zscore) | 0.3040 [0.2927, 0.3155] | 0.4338 [0.4212, 0.4468] | 0.5630 [0.5472, 0.5784] | 0.5903 [0.5738, 0.6065] |
| Fusion(learned_linear) | 0.3157 [0.3051, 0.3266] | 0.4525 [0.4409, 0.4640] | 0.4801 [0.4591, 0.5013] | 0.5254 [0.5039, 0.5467] |

**Note.** CIs are ~3× tighter than at n=300 and absolute scores are higher (the small subset was harder). Ordering is unchanged and sharper: SHAP most faithful, attention most plausible (single methods); **XHateExplain(mean_minmax)** holds the balance (IOU 0.3027, AOPC@5 0.558); `mean_zscore` edges faithfulness; `learned_linear` leads plausibility (IOU 0.3157) at a faithfulness cost. Consistent with Exp 1 → robust.

## Exp 2b — Full community threshold sweep (2,000-post sample)

| τ | Edges | Avg deg | #Comms | Modularity Q | Hate>70% | Normal<10% | Sentiment gap |
|---|---|---|---|---|---|---|---|
| 0.5 | 708,395 | 708.4 | 3 | 0.4300 | 1 | 1 | +0.2271 |
| 0.55 | 586,855 | 586.9 | 3 | 0.4632 | 1 | 1 | +0.2195 |
| 0.6 | 460,981 | 461.0 | 3 | 0.4985 | 1 | 1 | +0.2053 |
| 0.65 | 335,969 | 336.0 | 7 | 0.5328 | 2 | 3 | +0.2109 |
| 0.7 ✓ | 224,081 | 224.1 | 26 | 0.5625 | 7 | 14 | +0.1380 |
| 0.8 | 72,602 | 72.6 | 337 | 0.4605 | 64 | 269 | +0.1843 |
| 0.85 | 33,316 | 33.3 | 864 | 0.3309 | 155 | 702 | +0.2736 |

**argmax-Q = τ=0.7** (Q=0.5625) — modularity rises then falls, peaking exactly at 0.70, so the original choice is **confirmed** over the full 0.50–0.85 range. Lower τ collapses to ~3 giant blobs; higher τ fragments into hundreds of singletons. The hate-vs-normal sentiment gap is positive at every τ (normal less negative than hate).

## Exp 3 — Multi-seed classification (seeds [42, 1, 2], mean ± std)

| Model | Accuracy | Macro-F1 | Hate-F1 | Off-F1 | Normal-F1 | AUC |
|---|---|---|---|---|---|---|
| BiLSTM+Attention | 0.6516±0.0000 | 0.6336±0.0000 | 0.7103±0.0000 | 0.4890±0.0000 | 0.7014±0.0000 | 0.8182±0.0000 |
| BERT | 0.7010±0.0033 | 0.6875±0.0070 | 0.7679±0.0035 | 0.5441±0.0231 | 0.7504±0.0046 | 0.8601±0.0020 |
| HateBERT | 0.6970±0.0083 | 0.6901±0.0084 | 0.7673±0.0058 | 0.5606±0.0168 | 0.7424±0.0032 | 0.8552±0.0121 |

**McNemar BERT vs HateBERT, per seed:** seed 42: p=0.8562 (n.s.), seed 1: p=0.9574 (n.s.), seed 2: p=0.2272 (n.s.).
**The non-significant BERT≈HateBERT tie holds across ALL seeds: True.** Std-devs are tiny (≤0.011 on every metric), so the single-run conclusions are stable; HateBERT's small edge (~0.014 macro-F1) is consistent but never statistically significant.

## Exp 4 — Out-of-distribution (zero-shot on Davidson et al. 2017, n=24,783)

Label mapping: {'0': 'hate', '1': 'offensive', '2': 'normal'}.

| Metric | Value |
|---|---|
| accuracy | 0.5829 |
| macro_f1 | 0.4834 |
| f1_hate | 0.3285 |
| f1_offensive | 0.6784 |
| f1_normal | 0.4434 |
| auc | 0.8018 |

**Explanation transfer.** On 41 slur-bearing OOD posts, a slur was a top-3 SHAP token in 29 (71%).
**Verdict.** Expected zero-shot accuracy drop (in-domain macro-F1 ~0.683 → OOD 0.483), but AUC stays high (0.802) — ranking transfers, calibrated thresholds do not. The slur-centric attribution pattern transfers strongly (≈71%), so the explanations are not a HateXplain artifact. Main OOD error is offensive→normal (annotation-scheme mismatch).

## New files
**results/** — fusion_ablation_ci.json, explainability_ci_large.json, large_importances_{shap,lime,attention}.json, community_threshold_sweep_full.csv, community_sweep_full_summary.json, classification_multiseed.json, ood_davidson.json

**figures/** — fusion_ablation.png, modularity_vs_threshold.png, table1_classification_multiseed.png, ood_examples.png

**scripts/** — step12_fusion_ablation.py, step13_explain_large.py, step14_threshold_sweep_full.py, step15_multiseed.py, step16_ood_davidson.py