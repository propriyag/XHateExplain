# XHateExplain

**A Multi-Level Interpretability Framework for Hate Speech Detection in Social Networks**

XHateExplain fine-tunes BERT for 3-class hate-speech classification on
[HateXplain](https://github.com/hate-alert/HateXplain) and explains its decisions at
three granularities:

- **Word level** — which tokens triggered the decision (SHAP, LIME, attention, and a fused *XHateExplain* explainer)
- **Post level** — rationale extraction scored against human annotations
- **Community level** — content-similarity graph + Louvain communities profiled by LDA topics, VADER sentiment, and centrality

Explanations are evaluated for **plausibility** (agreement with human rationales: IOU, token-F1)
and **faithfulness** (AOPC — probability drop when the top tokens are masked), with
95% bootstrap confidence intervals throughout.

---

## Results at a glance

**Classification (HateXplain test, 3 seeds, mean ± std):**

| Model | Macro-F1 | Hate-F1 | Offensive-F1 | Normal-F1 | AUC |
|-------|----------|---------|--------------|-----------|-----|
| BiLSTM+Attention | 0.634 | 0.710 | 0.489 | 0.701 | 0.818 |
| BERT | 0.688 ± 0.007 | 0.768 | 0.544 | 0.750 | 0.860 |
| HateBERT | 0.690 ± 0.008 | 0.767 | 0.561 | 0.742 | 0.855 |

BERT and HateBERT are statistically tied (McNemar p = 0.86 / 0.96 / 0.23 across seeds).

**Explainability (n = 1,141 rationale posts, 95% bootstrap CI):**

| Method | IOU (plausibility) | AOPC@5 (faithfulness) |
|--------|--------------------|------------------------|
| SHAP | 0.298 | **0.578** |
| Attention | **0.305** | 0.421 |
| XHateExplain (fused) | 0.303 | 0.558 |

SHAP is most faithful, attention most plausible; the fused explainer balances both.

**Out-of-distribution (zero-shot on Davidson 2017):** macro-F1 0.483, AUC 0.802; the
slur-centric attribution pattern transfers to ~71% of slur-bearing posts.

See [`results of experiment/Reviewer_Result.docx`](results%20of%20experiment/) and
`REVISION_RESULTS.md` for the full tables, graphs, and discussion.

---

## Installation

```bash
git clone https://github.com/<your-username>/XHateExplain.git
cd XHateExplain
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Runs on CPU, CUDA, or Apple-Silicon MPS (auto-detected in `config.py`).
The HateXplain dataset and GloVe/Davidson files are **downloaded automatically** on
first run — they are not committed to the repo.

## Usage

Run the full pipeline in order:

```bash
bash run_all.sh
```

…or run any stage individually:

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `step1_data.py` | Download + parse HateXplain, 80/10/10 split |
| 2 | `step2_train_bert.py` | Fine-tune `bert-base-uncased` (3-class) |
| 3 | `step3_shap.py` | SHAP word-level explanations |
| 4 | `step4_lime.py` | LIME explanations + HTML examples |
| 5 | `step5_attention.py` | Attention explanations + fused XHateExplain |
| 6 | `step6_community.py` | Similarity graph + Louvain + LDA + VADER |
| 7 | `step7_tables_figures.py` | Publication tables & figures |
| 8 | `step8_baselines.py` | BiLSTM+GloVe, HateBERT, McNemar |

**Revision / robustness experiments** (reuse cached artifacts — no retraining unless noted):

| Step | Script | Experiment |
|------|--------|-----------|
| 9  | `step9_explain_ci.py` | Bootstrap CIs + paired significance |
| 10 | `step10_community_sweep.py` | Community threshold robustness |
| 11 | `step11_offensive_calibration.py` | Offensive-class calibration (negative result) |
| 12 | `step12_fusion_ablation.py` | Fusion ablation (4 variants) |
| 13 | `step13_explain_large.py` | Larger-n explainability (n=1,141) |
| 14 | `step14_threshold_sweep_full.py` | Full τ-sweep 0.50–0.85 |
| 15 | `step15_multiseed.py` | Multi-seed classification (GPU) |
| 16 | `step16_ood_davidson.py` | Out-of-distribution check |

## Repository layout

```
config.py              # paths, hyperparameters, device selection
modeling.py            # transformer train/eval helpers
explain_common.py      # shared plausibility / faithfulness scoring
utils.py               # metrics (IOU, token-F1, AOPC), rationale parsing
step1..step16_*.py     # pipeline + revision experiments
make_*.py              # Word/Markdown report generators
run_*.sh               # orchestration scripts
results/               # metrics JSON/CSV (generated)
figures/               # 300-DPI figures (generated)
data/ models/          # downloaded data + trained weights (gitignored)
```

## Reproducibility

All randomness is seeded (`SEED = 42` in `config.py`); bootstrap uses 10,000 resamples.
Every number in the reports is read directly from the saved `results/*.json|csv` files.

## Citation

```bibtex
@article{xhateexplain2026,
  title   = {XHateExplain: A Multi-Level Interpretability Framework for
             Hate Speech Detection in Social Networks},
  author  = {Gupta, Priya},
  year    = {2026},
  note    = {Under review}
}
```

## Acknowledgements

- HateXplain dataset — Mathew et al., 2021
- Davidson et al., 2017 (OOD evaluation)
- SHAP (Lundberg & Lee, 2017), LIME (Ribeiro et al., 2016), Louvain (Blondel et al., 2008), VADER (Hutto & Gilbert, 2014)

## License

Released under the [MIT License](LICENSE).
