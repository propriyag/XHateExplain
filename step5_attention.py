"""Step 5 - Attention-based explanations.

Extracts last-layer attention, averages heads, and uses the [CLS] token's
attention to each token as the importance score. Subword attentions are
aggregated back to HateXplain word tokens via word_ids(). Produces the same
plausibility/faithfulness metrics, a heatmap figure for 5 posts, and an
attention-vs-SHAP-vs-LIME comparison.
"""
import json
import os

import numpy as np
import torch

from config import BEST_BERT_DIR, MAX_LEN, RESULT_DIR, FIG_DIR, get_device
from modeling import make_predict_proba
from explain_common import get_subsample, score_and_save, combine_explainers
from utils import save_json


def word_attention(model, tok, tokens, device):
    """CLS->token attention (last layer, mean heads) aggregated to word tokens."""
    enc = tok(tokens, is_split_into_words=True, truncation=True, max_length=MAX_LEN,
              return_tensors="pt")
    word_ids = enc.word_ids(0)
    with torch.no_grad():
        out = model(**{k: v.to(device) for k, v in enc.items()}, output_attentions=True)
    pred = int(out.logits.argmax(-1))
    att = out.attentions[-1][0].mean(0)          # (seq, seq) mean over heads
    cls_att = att[0].cpu().numpy()               # attention from [CLS] to each subword
    imp = np.zeros(len(tokens), dtype=float)
    counts = np.zeros(len(tokens), dtype=float)
    for pos, wid in enumerate(word_ids):
        if wid is not None and wid < len(tokens):
            imp[wid] += cls_att[pos]; counts[wid] += 1
    counts[counts == 0] = 1
    return imp / counts, pred


def main():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = get_device()
    tok = AutoTokenizer.from_pretrained(BEST_BERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        BEST_BERT_DIR, output_attentions=True).to(device).eval()
    predict_proba = make_predict_proba(BEST_BERT_DIR)

    records = get_subsample()
    print(f"[Attention] explaining {len(records)} test posts ...")
    importances, preds = [], []
    for i, rec in enumerate(records):
        imp, pred = word_attention(model, tok, rec["tokens"], device)
        importances.append(imp); preds.append(pred)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(records)}")

    summary, df = score_and_save("Attention", records, importances, predict_proba, preds)

    # --- heatmap for 5 example posts ---
    import matplotlib.pyplot as plt
    ex = records[:5]; ex_imp = importances[:5]
    fig, axes = plt.subplots(len(ex), 1, figsize=(12, 1.4 * len(ex)))
    for ax, rec, imp in zip(np.atleast_1d(axes), ex, ex_imp):
        toks = rec["tokens"]
        ax.imshow(imp[None, :], cmap="Reds", aspect="auto")
        ax.set_xticks(range(len(toks)))
        ax.set_xticklabels(toks, rotation=60, ha="right", fontsize=7)
        ax.set_yticks([]); ax.set_ylabel(rec["post_id"][:8], fontsize=7)
    fig.suptitle("Last-layer [CLS] attention (averaged heads)")
    fig.tight_layout()
    hp = os.path.join(FIG_DIR, "attention_heatmaps.png")
    fig.savefig(hp, dpi=300); plt.close(fig)
    print(f"  saved {hp}")

    # --- attention vs SHAP vs LIME ---
    cmp = {"Attention": {k: summary[k] for k in ("plausibility_iou", "token_f1", "aopc_5", "aopc_10")}}
    for other in ("shap", "lime"):
        p = os.path.join(RESULT_DIR, f"{other}_summary.json")
        if os.path.exists(p):
            s = json.load(open(p))
            cmp[other.upper()] = {k: s[k] for k in ("plausibility_iou", "token_f1", "aopc_5", "aopc_10")}
    if len(cmp) > 1:
        best = max(cmp.items(), key=lambda kv: kv[1]["plausibility_iou"])[0]
        cmp["best_plausibility"] = best
        save_json(cmp, os.path.join(RESULT_DIR, "attention_vs_shap_vs_lime.json"))
        print(f"  best plausibility among available methods: {best}")

    # --- XHateExplain combined explainer (SHAP + LIME + Attention) ---
    print("\n[XHateExplain] combining SHAP + LIME + Attention ...")
    combine_explainers(predict_proba=predict_proba)


if __name__ == "__main__":
    main()
