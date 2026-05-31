"""Step 7 - Assemble publication-ready tables and figures from the results of the
previous steps. Every table is saved as CSV and as a 300-DPI PNG; every figure
is saved as a 300-DPI PNG. Missing inputs are skipped gracefully.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import RESULT_DIR, FIG_DIR, LABELS

plt.rcParams.update({"font.size": 9})


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def render_table(df, path_png, title):
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(df.columns), 0.6 + 0.45 * (len(df) + 1)))
    ax.axis("off")
    ax.set_title(title, fontweight="bold", pad=12)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    for j in range(len(df.columns)):
        tbl[0, j].set_facecolor("#34495e"); tbl[0, j].set_text_props(color="w", fontweight="bold")
    fig.tight_layout(); fig.savefig(path_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path_png}")


# --------------------------------------------------------------------------- Table 1
def table1_classification():
    bert = load(os.path.join(RESULT_DIR, "bert_metrics.json"))
    base = load(os.path.join(RESULT_DIR, "baselines.json"))
    if not bert:
        print("  [T1] bert_metrics.json missing — skip"); return
    rows = {}
    if base:
        rows["BiLSTM+Attention"] = base["results"].get("BiLSTM+Attention")
        rows["HateBERT"] = base["results"].get("HateBERT")
    rows["BERT"] = bert["test"]
    rows["XHateExplain"] = bert["test"]    # same fine-tuned classifier; framework adds explainability
    cols = [("Accuracy", "accuracy"), ("Macro-F1", "macro_f1"), ("Hate-F1", "f1_hate"),
            ("Offensive-F1", "f1_offensive"), ("Normal-F1", "f1_normal"), ("AUC", "auc")]
    data = []
    for name, m in rows.items():
        if not m:
            continue
        data.append([name] + [f"{m.get(k, float('nan')):.3f}" for _, k in cols])
    df = pd.DataFrame(data, columns=["Model"] + [c for c, _ in cols])
    df.to_csv(os.path.join(RESULT_DIR, "table1_classification.csv"), index=False)
    render_table(df, os.path.join(FIG_DIR, "table1_classification.png"),
                 "Table 1 — Classification Performance")


# --------------------------------------------------------------------------- Table 2
def table2_explainability():
    rows = []
    for m in ("SHAP", "LIME", "Attention", "XHateExplain"):
        s = load(os.path.join(RESULT_DIR, f"{m.lower()}_summary.json"))
        if s:
            rows.append([m, f"{s['plausibility_iou']:.3f}", f"{s['aopc_5']:.3f}",
                         f"{s['aopc_10']:.3f}", f"{s['token_f1']:.3f}"])
    if not rows:
        print("  [T2] no explainer summaries — skip"); return
    df = pd.DataFrame(rows, columns=["Method", "Plausibility (IOU)", "AOPC@5", "AOPC@10", "Token-F1"])
    df.to_csv(os.path.join(RESULT_DIR, "table2_explainability.csv"), index=False)
    render_table(df, os.path.join(FIG_DIR, "table2_explainability.png"),
                 "Table 2 — Explainability Comparison")


# --------------------------------------------------------------------------- Table 3
def table3_communities():
    p = os.path.join(RESULT_DIR, "community_profiles.csv")
    if not os.path.exists(p):
        print("  [T3] community_profiles.csv missing — skip"); return
    df = pd.read_csv(p).head(5).copy()
    df["Hate%"] = (df["hate_ratio"] * 100).round(1)
    show = df[["community", "size", "Hate%", "avg_sentiment", "top_topics", "key_posts"]].copy()
    show.columns = ["Comm", "Size", "Hate%", "Sentiment", "Top Topics", "Key Posts"]
    show["Top Topics"] = show["Top Topics"].str.slice(0, 40)
    show["Key Posts"] = show["Key Posts"].str.slice(0, 28)
    show.to_csv(os.path.join(RESULT_DIR, "table3_communities.csv"), index=False)
    render_table(show, os.path.join(FIG_DIR, "table3_communities.png"),
                 "Table 3 — Top-5 Community Profiles")


# --------------------------------------------------------------------------- Figure 1
def figure1_architecture():
    fig, ax = plt.subplots(figsize=(11, 6)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)

    def box(x, y, w, h, text, color):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                     fc=color, ec="black", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.3))

    box(0.2, 2.5, 1.6, 1.0, "HateXplain\nposts + rationales", "#d6eaf8")
    box(2.2, 2.5, 1.6, 1.0, "BERT encoder\n(fine-tuned 3-class)", "#aed6f1")
    box(4.2, 4.2, 1.7, 0.9, "SHAP", "#f9e79f")
    box(4.2, 3.0, 1.7, 0.9, "LIME", "#f9e79f")
    box(4.2, 1.8, 1.7, 0.9, "Attention", "#f9e79f")
    box(6.4, 3.0, 1.7, 1.0, "XHateExplain\ncombined explainer", "#f5b7b1")
    box(6.4, 0.4, 1.7, 1.0, "Community analysis\n(graph+Louvain+LDA)", "#abebc6")
    box(8.4, 3.0, 1.4, 1.0, "Plausibility +\nFaithfulness", "#d2b4de")
    box(8.4, 0.4, 1.4, 1.0, "Community\nprofiles", "#d2b4de")

    arrow(1.8, 3.0, 2.2, 3.0)
    for y in (4.6, 3.4, 2.2):
        arrow(3.8, 3.0, 4.2, y)
    for y in (4.6, 3.4, 2.2):
        arrow(5.9, y, 6.4, 3.5)
    arrow(3.8, 2.7, 6.4, 1.0)
    arrow(8.1, 3.5, 8.4, 3.5)
    arrow(8.1, 0.9, 8.4, 0.9)
    ax.set_title("Figure 1 — XHateExplain framework architecture", fontweight="bold")
    fig.savefig(os.path.join(FIG_DIR, "figure1_architecture.png"), dpi=300, bbox_inches="tight")
    plt.close(fig); print("  saved figure1_architecture.png")


# --------------------------------------------------------------------------- helpers for fig 2/3
def _load_imp(method):
    return load(os.path.join(RESULT_DIR, f"{method}_importances.json"))


def _pick_hate_post(dump):
    for d in dump:
        if d["pred"] == LABELS.index("hate"):
            return d
    return dump[0]


# --------------------------------------------------------------------------- Figure 2
def figure2_shap_waterfall():
    dump = _load_imp("shap")
    if not dump:
        print("  [F2] shap_importances.json missing — skip"); return
    d = _pick_hate_post(dump)
    toks, imp = d["tokens"], np.asarray(d["imp"])
    order = np.argsort(np.abs(imp))[::-1][:12][::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#c0392b" if imp[i] > 0 else "#2980b9" for i in order]
    ax.barh([toks[i] for i in order], [imp[i] for i in order], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("SHAP value (contribution to 'hate' prediction)")
    ax.set_title("Figure 2 — SHAP explanation for an example hate post")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "figure2_shap_waterfall.png"), dpi=300)
    plt.close(fig); print("  saved figure2_shap_waterfall.png")


# --------------------------------------------------------------------------- Figure 3
def figure3_method_comparison():
    dumps = {m: _load_imp(m) for m in ("shap", "lime", "attention")}
    if any(v is None for v in dumps.values()):
        print("  [F3] need shap/lime/attention importances — skip"); return
    target = _pick_hate_post(dumps["shap"])["post_id"]
    def get(method):
        for d in dumps[method]:
            if d["post_id"] == target:
                return d
        return None
    recs = {m: get(m) for m in dumps}
    toks = recs["shap"]["tokens"]
    def norm(v):
        v = np.abs(np.asarray(v)); return v / v.max() if v.max() > 0 else v
    x = np.arange(len(toks)); w = 0.27
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(toks)), 4.5))
    ax.bar(x - w, norm(recs["shap"]["imp"]), w, label="SHAP", color="#f1c40f")
    ax.bar(x,     norm(recs["lime"]["imp"]), w, label="LIME", color="#e67e22")
    ax.bar(x + w, norm(recs["attention"]["imp"]), w, label="Attention", color="#3498db")
    ax.set_xticks(x); ax.set_xticklabels(toks, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("normalised importance"); ax.legend()
    ax.set_title("Figure 3 — SHAP vs LIME vs Attention on the same post")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "figure3_method_comparison.png"), dpi=300)
    plt.close(fig); print("  saved figure3_method_comparison.png")


def figure4_note():
    p = os.path.join(FIG_DIR, "community_graph.png")
    print(f"  Figure 4 = {p}" if os.path.exists(p)
          else "  [F4] community_graph.png missing (run step6) — skip")


def main():
    print("[Step7] tables ...")
    table1_classification(); table2_explainability(); table3_communities()
    print("[Step7] figures ...")
    figure1_architecture(); figure2_shap_waterfall(); figure3_method_comparison(); figure4_note()
    print("[Step7] done.")


if __name__ == "__main__":
    main()
