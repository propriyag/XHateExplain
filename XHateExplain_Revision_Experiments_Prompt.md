# Claude Code Prompt — XHateExplain Major-Revision Experiments

Paste this into Claude Code, opened in the `/Users/priyu/xhateexplain` project. It addresses the reviewer's three hardest concerns. Run the experiments in the order given (1 and 2 are cheap and need NO retraining; 3 and 4 need GPU retraining). After each, print a short block I can paste into the paper, and save the listed artifacts. Do not delete or overwrite existing results — write new files with the names below.

Context you can rely on: the pipeline already produced `results/` and `figures/`; the fine-tuned model is in `models/`; cached per-token attributions are in `results/{shap,lime,attention,xhateexplain}_importances.json`; the 300-post explainability subset and `bert_test_predictions.csv` already exist; community code is `step6_community.py` / `step10_community_sweep.py`. Reuse these — do not recompute SHAP/LIME from scratch unless a step explicitly needs it.

---

## Experiment 1 — Fusion ablation (HIGHEST PRIORITY, no retraining)
Reviewer concern: the unweighted min–max mean has no principled justification and is statistically tied/worse than its components. Test whether a better fusion exists.

Using the cached per-token attributions for the same 300-post subset, build and evaluate **four fusion variants** of SHAP+LIME+attention:
1. `mean_minmax` (current method — reproduce as the baseline)
2. `mean_zscore` (z-score normalise each method's per-post vector, then average)
3. `rank_fusion` (reciprocal-rank fusion: score token t = Σ_m 1/(k0 + rank_m(t)), k0=60)
4. `learned_linear` (fit non-negative weights w_SHAP, w_LIME, w_ATT on a 5-fold CV over the 300 posts to maximise token-F1 against human rationales; report the mean learned weights)

For each variant compute IOU, token-F1, AOPC@5, AOPC@10 with **95% bootstrap CIs (10,000 resamples, seed 42)**, exactly as in `step9_explain_ci.py`. Run paired bootstrap tests of each variant vs. (a) attention and (b) SHAP on IOU and AOPC@5.

Also, to explain the existing IOU tie, **report the per-component post-normalisation L1 mass**: for each method, the mean (over posts) of Σ_t |ã_m(t)| after min–max normalisation, so we can show attention dominates the unweighted mean.

Save `results/fusion_ablation_ci.json` and `figures/fusion_ablation.png` (bars ± CI). Print: which fusion, if any, is significantly better than the current mean on either axis, the learned weights, and the L1-mass table. Verdict sentence: does a smarter fusion beat the simple mean, or is the simple mean justified as "no worse"?

## Experiment 2 — Bigger explanation eval + lower-τ sweep (no retraining)
Reviewer concern: n=300 is small; τ grid starts too high.
- Re-run the explainability metrics (all methods + best fusion from Exp. 1) on a **larger subset: every test post that carries a human rationale** (report the resulting n), or 1,000 stratified posts if that is faster — state which. Reuse cached attributions where available; only compute attributions for the additional posts. Report the same metrics with 95% CIs. Save `results/explainability_ci_large.json`.
- Extend the community threshold sweep to **τ ∈ {0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.85}** (add the lower values to the existing three). For each: edges, avg degree, #communities, modularity Q, hate>70% / normal<10% counts, and the hate-heavy vs normal-heavy mean-sentiment gap. Save `results/community_threshold_sweep_full.csv` and `figures/modularity_vs_threshold.png`. Print which τ maximises modularity over the full range (confirm or revise the choice of 0.70).

## Experiment 3 — Multi-seed classification (needs GPU retraining)
Reviewer concern: the McNemar "BERT ≈ HateBERT" tie rests on a single run.
Re-train **BERT, HateBERT and the BiLSTM+attention baseline with 3 seeds each** (e.g. 42, 1, 2), same hyperparameters as `config.py`/`step2_train_bert.py`. For every Table-2 metric (accuracy, macro-F1, per-class F1, AUC) report **mean ± standard deviation across seeds**. Re-run the McNemar BERT-vs-HateBERT test per seed and report whether the non-significant result holds across all three. Save `results/classification_multiseed.json` and a regenerated `figures/table1_classification_multiseed.png`. Print the mean ± std table and the per-seed McNemar p-values. (If GPU time is tight, do BERT + HateBERT only and note the baseline stays single-seed.)

## Experiment 4 — Out-of-distribution check (needs data + light retraining)
Reviewer concern: single-dataset evaluation.
Take the **HateXplain-trained BERT as-is** (no retraining) and evaluate it zero-shot on one external dataset — **Davidson et al. 2017** (preferred; 3-class-mappable) or the **Stormfront** set [de Gibert 2018]. Map labels to {hate, offensive, normal} as closely as possible and document the mapping. Report accuracy and macro-F1 on the OOD set, and run SHAP/attention on 100 OOD posts to check whether the same slur-centric attribution pattern holds. Save `results/ood_davidson.json` (or `ood_stormfront.json`) and `figures/ood_examples.png`. Print the OOD metrics and one sentence on whether explanations transfer. (This is the lowest-priority item — skip if data download or label mapping is non-trivial.)

---

## Output
Print one consolidated **"REVISION RESULTS"** block containing: the fusion-ablation table + verdict + learned weights + L1-mass, the larger-n explainability table, the full τ-sweep table with the argmax-Q threshold, the multi-seed mean ± std table with per-seed McNemar p-values, and (if run) the OOD metrics. List every new file written under `results/` and `figures/`. Keep prose minimal — I will paste numbers into the manuscript.
