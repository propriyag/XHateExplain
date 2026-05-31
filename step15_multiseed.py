"""Experiment 3 - Multi-seed classification (GPU retraining).

Retrains BERT, HateBERT and BiLSTM+Attention with seeds 42, 1, 2 (same
hyperparameters as config.py), reports mean +/- std of every Table-1 metric, and
re-runs the McNemar BERT-vs-HateBERT test per seed.

Models are trained to temp dirs and NOT allowed to overwrite models/bert_best.

Outputs:
  results/classification_multiseed.json
  figures/table1_classification_multiseed.png
"""
import json
import os
import shutil

import numpy as np
import pandas as pd

from config import (TRAIN_CSV, VAL_CSV, TEST_CSV, BERT_MODEL, HATEBERT_MODEL,
                    MODEL_DIR, RESULT_DIR, FIG_DIR, LABELS)
from modeling import train_transformer, evaluate
from step8_baselines import train_bilstm, mcnemar

SEEDS = [42, 1, 2]
METRIC_KEYS = ["accuracy", "macro_f1", "f1_hate", "f1_offensive", "f1_normal", "auc"]


def agg(metric_dicts):
    """mean+/-std across seeds for each metric key."""
    out = {}
    for k in METRIC_KEYS:
        vals = [m[k] for m in metric_dicts]
        out[k] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}
    return out


def main():
    train = pd.read_csv(TRAIN_CSV); val = pd.read_csv(VAL_CSV); test = pd.read_csv(TEST_CSV)
    y_true = test.label_id.to_numpy()

    per_seed = {"BERT": [], "HateBERT": [], "BiLSTM+Attention": []}
    preds = {"BERT": {}, "HateBERT": {}}
    mcnemar_per_seed = {}

    for seed in SEEDS:
        print(f"\n########## SEED {seed} ##########")
        # ---- BERT ----
        d = os.path.join(MODEL_DIR, f"_ms_bert_{seed}")
        train_transformer(BERT_MODEL, d, train, val, seed=seed)
        m, pdf = evaluate(d, test)
        per_seed["BERT"].append(m); preds["BERT"][seed] = pdf.pred.to_numpy()
        print(f"  BERT seed {seed}: macroF1={m['macro_f1']:.4f}")
        shutil.rmtree(d, ignore_errors=True)

        # ---- HateBERT ----
        d = os.path.join(MODEL_DIR, f"_ms_hatebert_{seed}")
        train_transformer(HATEBERT_MODEL, d, train, val, seed=seed)
        m, pdf = evaluate(d, test)
        per_seed["HateBERT"].append(m); preds["HateBERT"][seed] = pdf.pred.to_numpy()
        print(f"  HateBERT seed {seed}: macroF1={m['macro_f1']:.4f}")
        shutil.rmtree(d, ignore_errors=True)

        # ---- BiLSTM+Attention (its own seeding via set_seed inside) ----
        from modeling import set_seed
        set_seed(seed)
        bl_m, _ = train_bilstm(train, val, test)
        per_seed["BiLSTM+Attention"].append(bl_m)
        print(f"  BiLSTM seed {seed}: macroF1={bl_m['macro_f1']:.4f}")

        # ---- McNemar BERT vs HateBERT for this seed ----
        mc = mcnemar(y_true, preds["BERT"][seed], preds["HateBERT"][seed])
        mcnemar_per_seed[seed] = mc
        print(f"  McNemar BERT vs HateBERT (seed {seed}): chi2={mc['chi2']} p={mc['p_value']} "
              f"sig={mc['significant_0.05']}")

    summary = {
        "seeds": SEEDS,
        "BERT": agg(per_seed["BERT"]),
        "HateBERT": agg(per_seed["HateBERT"]),
        "BiLSTM+Attention": agg(per_seed["BiLSTM+Attention"]),
        "mcnemar_per_seed": mcnemar_per_seed,
        "mcnemar_holds_all_seeds": all(not mc["significant_0.05"] for mc in mcnemar_per_seed.values()),
    }
    json.dump(summary, open(os.path.join(RESULT_DIR, "classification_multiseed.json"), "w"), indent=2)

    # ---- figure: grouped bars mean+/-std for macro-F1 + per-class F1 ----
    import matplotlib.pyplot as plt
    models = ["BiLSTM+Attention", "BERT", "HateBERT"]
    show_keys = ["macro_f1", "f1_hate", "f1_offensive", "f1_normal", "auc"]
    klabels = ["Macro-F1", "Hate-F1", "Off-F1", "Normal-F1", "AUC"]
    x = np.arange(len(show_keys)); wbar = 0.25
    colors = {"BiLSTM+Attention": "#95a5a6", "BERT": "#2980b9", "HateBERT": "#c0392b"}
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, mdl in enumerate(models):
        means = [summary[mdl][k]["mean"] for k in show_keys]
        stds = [summary[mdl][k]["std"] for k in show_keys]
        ax.bar(x + (i - 1) * wbar, means, wbar, yerr=stds, capsize=4, label=mdl,
               color=colors[mdl], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(klabels)
    ax.set_ylabel("score"); ax.set_ylim(0, 1)
    ax.set_title(f"Table 1 (multi-seed) — mean ± std over seeds {SEEDS}", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "table1_classification_multiseed.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- console ----
    print("\n===== EXP3: MULTI-SEED (mean +/- std) =====")
    hdr = f"{'Model':18}" + "".join(f"{k:>16}" for k in METRIC_KEYS)
    print(hdr)
    for mdl in models:
        line = f"{mdl:18}"
        for k in METRIC_KEYS:
            line += f"{summary[mdl][k]['mean']:.3f}+/-{summary[mdl][k]['std']:.3f}".rjust(16)
        print(line)
    print("\n  McNemar BERT vs HateBERT per seed:")
    for s, mc in mcnemar_per_seed.items():
        print(f"    seed {s}: chi2={mc['chi2']:.3f} p={mc['p_value']:.4f} "
              f"-> {'significant' if mc['significant_0.05'] else 'n.s.'}")
    print(f"  Non-significant tie holds across ALL seeds: {summary['mcnemar_holds_all_seeds']}")
    print("Saved: results/classification_multiseed.json, figures/table1_classification_multiseed.png")


if __name__ == "__main__":
    main()
