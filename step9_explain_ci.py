"""Item 1 - Bootstrap confidence intervals + paired significance tests on the
explainability metrics, using the cached per-post values in
results/{method}_explanations.csv. No model is loaded; nothing is recomputed
from scratch.

Outputs:
  results/explainability_ci.json
  results/table2_explainability_ci.csv
  figures/table2_explainability.png      (regenerated: mean +/- 95% CI)
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULT_DIR, FIG_DIR

METHODS = ["SHAP", "LIME", "Attention", "XHateExplain"]
METRICS = ["iou", "token_f1", "aopc_5", "aopc_10"]
METRIC_LABEL = {"iou": "Plausibility (IOU)", "token_f1": "Token-F1",
                "aopc_5": "AOPC@5", "aopc_10": "AOPC@10"}
N_BOOT = 10000
SEED = 42


def load_per_post():
    """Return {method: DataFrame indexed by post_id} on a common post ordering."""
    frames = {}
    for m in METHODS:
        df = pd.read_csv(os.path.join(RESULT_DIR, f"{m.lower()}_explanations.csv"))
        frames[m] = df.set_index("post_id")
    order = list(frames[METHODS[0]].index)        # common, identical order across methods
    return frames, order


def boot_mean_ci(values, rng, n_boot=N_BOOT):
    """Mean and 95% percentile CI of a 1-D array (NaNs dropped)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return float("nan"), (float("nan"), float("nan")), 0
    idx = rng.integers(0, n, size=(n_boot, n))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), (float(lo), float(hi)), n


def paired_boot(a, b, rng, n_boot=N_BOOT):
    """Paired bootstrap of mean(a-b) over posts where BOTH are non-NaN.
    Returns mean diff, 95% CI of diff, two-sided bootstrap p-value, n."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mask], b[mask]
    d = a - b
    n = len(d)
    if n == 0:
        return float("nan"), (float("nan"), float("nan")), float("nan"), 0
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # two-sided percentile bootstrap p-value
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    p = float(min(1.0, p))
    return float(d.mean()), (float(lo), float(hi)), p, n


def main():
    frames, order = load_per_post()
    rng = np.random.default_rng(SEED)

    # ----- per method/metric: mean + 95% CI -----
    cells = {}
    table_rows = []
    for m in METHODS:
        df = frames[m]
        row = {"method": m}
        for metric in METRICS:
            mean, (lo, hi), n = boot_mean_ci(df[metric].values, rng)
            cells[(m, metric)] = {"mean": round(mean, 4),
                                  "ci95": [round(lo, 4), round(hi, 4)], "n": n}
            row[metric] = f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"
        table_rows.append(row)
    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(os.path.join(RESULT_DIR, "table2_explainability_ci.csv"), index=False)

    # ----- paired significance tests -----
    def col(m, metric):
        return frames[m].loc[order, metric].values
    tests = {}
    spec = [
        ("XHateExplain_vs_Attention__iou",     "XHateExplain", "Attention", "iou"),
        ("XHateExplain_vs_Attention__token_f1","XHateExplain", "Attention", "token_f1"),
        ("XHateExplain_vs_SHAP__aopc_5",       "XHateExplain", "SHAP",      "aopc_5"),
        ("XHateExplain_vs_SHAP__aopc_10",      "XHateExplain", "SHAP",      "aopc_10"),
        ("Attention_vs_SHAP__aopc_5",          "Attention",    "SHAP",      "aopc_5"),
    ]
    for name, ma, mb, metric in spec:
        diff, (lo, hi), p, n = paired_boot(col(ma, metric), col(mb, metric), rng)
        sig = bool(not (lo <= 0 <= hi))      # CI excludes 0
        tests[name] = {"metric": metric, "method_a": ma, "method_b": mb,
                       "mean_diff": round(diff, 4), "ci95_diff": [round(lo, 4), round(hi, 4)],
                       "p_value": round(p, 4), "n_paired": n, "significant_0.05": sig}

    out = {"n_boot": N_BOOT, "seed": SEED,
           "n_posts": len(order), "n_with_rationale": cells[("SHAP", "iou")]["n"],
           "metrics": {f"{m}|{metric}": cells[(m, metric)] for m in METHODS for metric in METRICS},
           "paired_tests": tests}
    with open(os.path.join(RESULT_DIR, "explainability_ci.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ----- regenerate Table 2 figure: mean +/- 95% CI grouped bars -----
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    colors = ["#f1c40f", "#e67e22", "#3498db", "#c0392b"]
    for ax, metric in zip(axes, METRICS):
        means = [cells[(m, metric)]["mean"] for m in METHODS]
        los = [cells[(m, metric)]["ci95"][0] for m in METHODS]
        his = [cells[(m, metric)]["ci95"][1] for m in METHODS]
        yerr = np.array([[mean - lo for mean, lo in zip(means, los)],
                         [hi - mean for hi, mean in zip(his, means)]])
        ax.bar(range(len(METHODS)), means, yerr=yerr, capsize=5,
               color=colors, edgecolor="black", linewidth=0.6)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(METHODS, rotation=30, ha="right", fontsize=8)
        ax.set_title(METRIC_LABEL[metric], fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for i, mean in enumerate(means):
            ax.text(i, his[i], f"{mean:.4f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Table 2 — Explainability metrics with 95% bootstrap CIs "
                 f"(10,000 resamples; IOU/Token-F1 over {out['n_with_rationale']} rationale posts, "
                 f"AOPC over {len(order)} posts)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG_DIR, "table2_explainability.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ----- console summary -----
    print("\n===== ITEM 1: EXPLAINABILITY CIs (4 dp) =====")
    hdr = f"{'Method':13}" + "".join(f"{METRIC_LABEL[x]:>30}" for x in METRICS)
    print(hdr)
    for m in METHODS:
        line = f"{m:13}"
        for metric in METRICS:
            c = cells[(m, metric)]
            line += f"{c['mean']:.4f} [{c['ci95'][0]:.4f},{c['ci95'][1]:.4f}]".rjust(30)
        print(line)

    print("\n----- paired bootstrap significance -----")
    for name, t in tests.items():
        verdict = "SIGNIFICANT" if t["significant_0.05"] else "n.s. (tie)"
        print(f"  {name:38} diff={t['mean_diff']:+.4f} "
              f"CI[{t['ci95_diff'][0]:+.4f},{t['ci95_diff'][1]:+.4f}] "
              f"p={t['p_value']:.4f}  -> {verdict}")

    # verdict sentences
    plaus = tests["XHateExplain_vs_Attention__iou"]
    faith5 = tests["XHateExplain_vs_SHAP__aopc_5"]
    faith10 = tests["XHateExplain_vs_SHAP__aopc_10"]
    print("\n----- VERDICTS -----")
    print("  Plausibility (XHateExplain vs Attention, IOU): "
          + ("a real advantage." if plaus["significant_0.05"] and plaus["mean_diff"] > 0
             else "statistically a TIE (no significant difference)."))
    print("  Faithfulness (XHateExplain vs SHAP, AOPC@5/@10): "
          + ("statistically INDISTINGUISHABLE from SHAP."
             if not faith5["significant_0.05"] and not faith10["significant_0.05"]
             else "significantly different from SHAP (see signs above)."))
    print("\nSaved: results/explainability_ci.json, results/table2_explainability_ci.csv, "
          "figures/table2_explainability.png")


if __name__ == "__main__":
    main()
