"""Experiment 4 - Out-of-distribution check on Davidson et al. 2017.

Evaluates the HateXplain-trained BERT zero-shot (NO retraining) on Davidson, and
runs SHAP + attention on 100 OOD posts to check whether the slur-centric
attribution pattern transfers.

Label mapping (Davidson 'class' -> our labels):
  0 hate      -> hate
  1 offensive -> offensive
  2 neither   -> normal

Outputs:
  results/ood_davidson.json
  figures/ood_examples.png
"""
import json
import os
import re

import numpy as np
import pandas as pd

from config import DATA_DIR, BEST_BERT_DIR, LABELS, LABEL2ID, RESULT_DIR, FIG_DIR, SEED
from modeling import make_predict_proba
from utils import classification_report_dict

DAVIDSON_CSV = os.path.join(DATA_DIR, "davidson_labeled_data.csv")
MAP = {0: "hate", 1: "offensive", 2: "normal"}
N_OOD_ATTR = 100


def clean(t):
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"@\w+", "user", t)
    t = re.sub(r"&amp;|&gt;|&lt;", " ", t)
    t = re.sub(r"\bRT\b", "", t)
    t = re.sub(r"[^\w\s']", " ", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def main():
    df = pd.read_csv(DAVIDSON_CSV)
    df["label"] = df["class"].map(MAP)
    df["text"] = df["tweet"].astype(str).map(clean)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    df["label_id"] = df["label"].map(LABEL2ID)
    n = len(df)
    print(f"[Exp4] Davidson OOD: {n} posts | dist={dict(df.label.value_counts())}")

    predict_proba = make_predict_proba(BEST_BERT_DIR)
    print("  scoring zero-shot (inference only) ...")
    proba = predict_proba(list(df.text))
    y_pred = proba.argmax(1); y_true = df.label_id.to_numpy()
    metrics = classification_report_dict(y_true, y_pred, proba)
    print(f"  OOD accuracy={metrics['accuracy']:.4f} macroF1={metrics['macro_f1']:.4f} "
          f"AUC={metrics.get('auc', float('nan')):.4f}")

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    # ---- SHAP + attention on 100 stratified OOD posts ----
    rng = np.random.default_rng(SEED)
    sub = (df.groupby("label_id", group_keys=False)
             .sample(frac=min(1.0, N_OOD_ATTR / n), random_state=SEED)
             .reset_index(drop=True))
    sub = sub.head(N_OOD_ATTR)
    print(f"  running SHAP + attention on {len(sub)} OOD posts ...")

    import shap
    masker = shap.maskers.Text(r"\s+")
    explainer = shap.Explainer(predict_proba, masker, output_names=LABELS)
    sub_pred = list(predict_proba(list(sub.text)).argmax(1))

    # simple slur lexicon to test "slur-centric" attribution transfer
    SLURS = {"nigger", "nigga", "niger", "faggot", "fag", "kike", "spic", "retard",
             "bitch", "cunt", "whore", "coon", "wetback", "tranny", "dyke", "chink",
             "raghead", "beaner", "gook", "sheboon"}
    top_is_slur = 0; n_with_slur = 0; examples = []
    for i, (txt, pred) in enumerate(zip(sub.text, sub_pred)):
        toks = txt.split()
        if not any(t in SLURS for t in toks):
            continue
        n_with_slur += 1
        sv = explainer([txt], max_evals=200, silent=True)
        vals = np.asarray(sv.values[0])
        cls = vals[:, pred] if vals.ndim == 2 else vals
        words = sv.data[0]
        order = np.argsort(cls)[::-1]
        top_words = [words[j].strip().lower() for j in order[:3] if j < len(words)]
        if any(w in SLURS for w in top_words):
            top_is_slur += 1
        if len(examples) < 5 and pred == LABEL2ID["hate"]:
            examples.append({"text": txt, "pred": LABELS[pred],
                             "top_words": top_words,
                             "imp": [float(cls[j]) for j in range(min(len(cls), len(words)))],
                             "words": [str(w) for w in words]})
    slur_rate = (top_is_slur / n_with_slur) if n_with_slur else float("nan")
    print(f"  slur-centric transfer: in {n_with_slur} slur-bearing posts, a slur was top-3 "
          f"attributed in {top_is_slur} ({slur_rate*100:.0f}%)")

    out = {
        "dataset": "Davidson et al. 2017", "n_posts": n,
        "label_mapping": {str(k): v for k, v in MAP.items()},
        "class_distribution": {k: int(v) for k, v in df.label.value_counts().items()},
        "zero_shot_metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "confusion_matrix_rows_true_cols_pred": cm, "cm_label_order": LABELS,
        "attribution_transfer": {
            "n_attr_posts": int(len(sub)), "n_with_slur": int(n_with_slur),
            "top3_contains_slur": int(top_is_slur),
            "slur_top_rate": round(float(slur_rate), 4) if slur_rate == slur_rate else None},
    }
    json.dump(out, open(os.path.join(RESULT_DIR, "ood_davidson.json"), "w"), indent=2)

    # ---- figure: up to 5 example attribution bars ----
    import matplotlib.pyplot as plt
    ex = examples[:5]
    if ex:
        fig, axes = plt.subplots(len(ex), 1, figsize=(11, 1.5 * len(ex)))
        for ax, e in zip(np.atleast_1d(axes), ex):
            words, imp = e["words"], np.asarray(e["imp"])
            m = min(len(words), len(imp), 18)
            colors = ["#c0392b" if imp[j] > 0 else "#2980b9" for j in range(m)]
            ax.bar(range(m), imp[:m], color=colors)
            ax.set_xticks(range(m)); ax.set_xticklabels(words[:m], rotation=60, ha="right", fontsize=7)
            ax.axhline(0, color="k", lw=0.5); ax.set_ylabel(e["pred"], fontsize=8)
        fig.suptitle("Figure — SHAP attributions on Davidson (OOD) example posts", fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(FIG_DIR, "ood_examples.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("  saved figures/ood_examples.png")

    print("\n===== EXP4: OOD (Davidson) =====")
    print(f"  zero-shot accuracy={metrics['accuracy']:.4f}  macro-F1={metrics['macro_f1']:.4f}")
    print(f"  per-class F1: hate={metrics['f1_hate']:.4f} off={metrics['f1_offensive']:.4f} "
          f"normal={metrics['f1_normal']:.4f}")
    print(f"  explanation transfer: slur is top-3 in {slur_rate*100:.0f}% of slur-bearing posts")
    print("Saved: results/ood_davidson.json, figures/ood_examples.png")


if __name__ == "__main__":
    main()
