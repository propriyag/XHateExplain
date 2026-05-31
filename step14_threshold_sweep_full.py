"""Experiment 2b - Full community threshold sweep.

Extends the sweep to tau in {0.50,0.55,0.60,0.65,0.70,0.80,0.85} on the same
2,000-post sample. For each tau: edges, avg degree, #communities, modularity Q,
hate>70% / normal<10% counts, and the hate-heavy vs normal-heavy sentiment gap.

Outputs:
  results/community_threshold_sweep_full.csv
  figures/modularity_vs_threshold.png
"""
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import community as community_louvain
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import TRAIN_CSV, COMMUNITY_SUBSAMPLE, SEED, RESULT_DIR, FIG_DIR
from step6_community import bert_embeddings

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.85]


def build_graph(sim, thr, n):
    G = nx.Graph(); G.add_nodes_from(range(n))
    iu = np.triu_indices(n, k=1); w = sim[iu]; mask = w >= thr
    for i, j, weight in zip(iu[0][mask], iu[1][mask], w[mask]):
        G.add_edge(int(i), int(j), weight=float(weight))
    return G


def main():
    df = pd.read_csv(TRAIN_CSV)
    if COMMUNITY_SUBSAMPLE and COMMUNITY_SUBSAMPLE < len(df):
        df = df.sample(COMMUNITY_SUBSAMPLE, random_state=SEED).reset_index(drop=True)
    n = len(df)
    print(f"[Exp2b] {n} posts -> embeddings (inference only) ...")
    emb = bert_embeddings(list(df.text))
    sim = cosine_similarity(emb); np.fill_diagonal(sim, 0.0)

    vader = SentimentIntensityAnalyzer()
    sent = np.array([vader.polarity_scores(t)["compound"] for t in df.text])
    is_hate = (df.label == "hate").to_numpy()

    rows = []
    for thr in THRESHOLDS:
        G = build_graph(sim, thr, n)
        ne = G.number_of_edges(); avg_deg = 2 * ne / n
        part = community_louvain.best_partition(G, random_state=SEED)
        Q = community_louvain.modularity(part, G)
        members = defaultdict(list)
        for node, c in part.items():
            members[c].append(node)
        hate_ratio = {c: is_hate[m].mean() for c, m in members.items()}
        hi = sum(1 for c in members if hate_ratio[c] > 0.70)
        lo = sum(1 for c in members if hate_ratio[c] < 0.10)
        # sentiment gap among communities of >=10 posts
        big = {c: m for c, m in members.items() if len(m) >= 10}
        hs = [sent[m].mean() for c, m in big.items() if hate_ratio[c] > 0.5]
        ns = [sent[m].mean() for c, m in big.items() if hate_ratio[c] < 0.1]
        gap = (np.mean(ns) - np.mean(hs)) if (hs and ns) else float("nan")
        rows.append({
            "threshold": thr, "edges": ne, "avg_degree": round(avg_deg, 1),
            "n_communities": len(members), "modularity_Q": round(Q, 4),
            "hate_comm_gt70pct": hi, "normal_comm_lt10pct": lo,
            "sentiment_gap_norm_minus_hate": round(float(gap), 4) if gap == gap else None,
        })
        print(f"  tau={thr:.2f}: edges={ne:6d} avgdeg={avg_deg:6.1f} comms={len(members):4d} "
              f"Q={Q:.4f} | hate>70%:{hi:3d} normal<10%:{lo:3d} | sentgap={gap:+.3f}")

    sweep = pd.DataFrame(rows)
    sweep.to_csv(os.path.join(RESULT_DIR, "community_threshold_sweep_full.csv"), index=False)
    best = sweep.loc[sweep["modularity_Q"].idxmax()]

    # ---- modularity vs threshold figure (twin axis with #communities) ----
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(sweep["threshold"], sweep["modularity_Q"], "o-", color="#c0392b", lw=2, label="Modularity Q")
    ax1.set_xlabel("Cosine similarity threshold τ"); ax1.set_ylabel("Modularity Q", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b"); ax1.grid(alpha=0.3)
    ax1.axvline(best["threshold"], ls="--", color="gray", alpha=0.7)
    ax1.annotate(f"max Q @ τ={best['threshold']}", (best["threshold"], best["modularity_Q"]),
                 textcoords="offset points", xytext=(8, -4), fontsize=9)
    ax2 = ax1.twinx()
    ax2.plot(sweep["threshold"], sweep["n_communities"], "s--", color="#2980b9", lw=1.5,
             label="# communities")
    ax2.set_ylabel("# communities", color="#2980b9"); ax2.tick_params(axis="y", labelcolor="#2980b9")
    fig.suptitle("Experiment 2 — Modularity & community count vs similarity threshold", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG_DIR, "modularity_vs_threshold.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n===== EXP2b: FULL THRESHOLD SWEEP =====")
    print(sweep.to_string(index=False))
    print(f"\n  argmax-Q threshold = {best['threshold']} (Q={best['modularity_Q']:.4f}) "
          f"-> {'CONFIRMS' if best['threshold']==0.70 else 'REVISES'} the choice of 0.70")
    with open(os.path.join(RESULT_DIR, "community_sweep_full_summary.json"), "w") as f:
        json.dump({"argmax_Q_threshold": float(best["threshold"]),
                   "argmax_Q": float(best["modularity_Q"]),
                   "rows": rows}, f, indent=2)
    print("Saved: results/community_threshold_sweep_full.csv, figures/modularity_vs_threshold.png")


if __name__ == "__main__":
    main()
