"""Item 2 - Community-graph robustness sweep.

Rebuilds the post-similarity graph on the SAME 2,000-post sample at cosine
thresholds 0.70 / 0.80 / 0.85, and for each reports edges, average degree,
#communities, modularity Q, and hate-ratio separation. The encoder is used for
embeddings only (inference) - the classifier is NOT retrained.

Outputs:
  results/community_threshold_sweep.csv
  results/community_profiles.csv          (overwritten: profiles at best threshold)
  results/community_meta.json             (overwritten: best-threshold meta)
  figures/community_graph_0.85.png
  figures/community_graph_best.png
"""
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import community as community_louvain
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import (TRAIN_CSV, SIM_THRESHOLD, COMMUNITY_SUBSAMPLE, SEED,
                    RESULT_DIR, FIG_DIR)
from step6_community import bert_embeddings, lda_keywords

THRESHOLDS = [0.70, 0.80, 0.85]


def build_graph(sim, thr, n):
    G = nx.Graph(); G.add_nodes_from(range(n))
    iu = np.triu_indices(n, k=1)
    w = sim[iu]
    mask = w >= thr
    for i, j, weight in zip(iu[0][mask], iu[1][mask], w[mask]):
        G.add_edge(int(i), int(j), weight=float(weight))
    return G


def hate_separation(df, partition):
    members = defaultdict(list)
    for node, c in partition.items():
        members[c].append(node)
    hi, lo, sizes = 0, 0, []
    hate_ratios = {}
    for c, nodes in members.items():
        hr = float((df.iloc[nodes].label == "hate").mean())
        hate_ratios[c] = hr
        sizes.append(len(nodes))
        if hr > 0.70:
            hi += 1
        elif hr < 0.10:
            lo += 1
    return len(members), hi, lo, hate_ratios, members


def profile_communities(df, G, partition, members, hate_ratios):
    vader = SentimentIntensityAnalyzer()
    rows = []
    for c, nodes in members.items():
        sub = df.iloc[nodes]
        sent = float(np.mean([vader.polarity_scores(t)["compound"] for t in sub.text]))
        topics = lda_keywords(list(sub.text))
        subG = G.subgraph(nodes)
        if subG.number_of_edges() > 0:
            bc = nx.betweenness_centrality(subG)
            key_nodes = sorted(bc, key=bc.get, reverse=True)[:3]
        else:
            key_nodes = nodes[:3]
        rows.append({
            "community": int(c), "size": len(nodes),
            "hate_ratio": round(hate_ratios[c], 3),
            "toxic_ratio": round(float((sub.label != "normal").mean()), 3),
            "avg_sentiment": round(sent, 3),
            "top_topics": " | ".join(topics[:2]) if topics else "(too small for LDA)",
            "key_posts": ", ".join(p[:14] for p in (df.iloc[n].post_id for n in key_nodes)),
        })
    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)


def draw_graph(df, G, partition, hate_ratios, thr, path, title):
    import matplotlib.pyplot as plt
    node_color = [hate_ratios[partition[i]] for i in range(len(df))]
    pos = nx.spring_layout(G, seed=SEED, k=0.15, iterations=30)
    fig, ax = plt.subplots(figsize=(11, 9))
    nx.draw_networkx_edges(G, pos, alpha=0.05, ax=ax)
    sc = nx.draw_networkx_nodes(G, pos, node_size=18, node_color=node_color,
                                cmap="Reds", vmin=0, vmax=1, ax=ax)
    plt.colorbar(sc, ax=ax, label="community hate ratio")
    ax.set_title(title); ax.axis("off"); fig.tight_layout()
    fig.savefig(path, dpi=300); plt.close(fig)


def main():
    df = pd.read_csv(TRAIN_CSV)
    if COMMUNITY_SUBSAMPLE and COMMUNITY_SUBSAMPLE < len(df):
        df = df.sample(COMMUNITY_SUBSAMPLE, random_state=SEED).reset_index(drop=True)
    n = len(df)
    print(f"[sweep] {n} posts -> embeddings (encoder inference only) ...")
    emb = bert_embeddings(list(df.text))
    sim = cosine_similarity(emb); np.fill_diagonal(sim, 0.0)

    sweep_rows, kept = [], {}
    for thr in THRESHOLDS:
        G = build_graph(sim, thr, n)
        ne = G.number_of_edges()
        avg_deg = 2 * ne / n
        partition = community_louvain.best_partition(G, random_state=SEED)
        Q = community_louvain.modularity(partition, G)
        n_comm, hi, lo, hate_ratios, members = hate_separation(df, partition)
        # consider only communities with >=10 posts for "clean separation" reporting
        big = {c: m for c, m in members.items() if len(m) >= 10}
        big_hi = sum(1 for c in big if hate_ratios[c] > 0.70)
        big_lo = sum(1 for c in big if hate_ratios[c] < 0.10)
        sweep_rows.append({
            "threshold": thr, "edges": ne, "avg_degree": round(avg_deg, 1),
            "n_communities": n_comm, "modularity_Q": round(Q, 4),
            "comm_ge10": len(big),
            "hate_comm_gt70pct": hi, "normal_comm_lt10pct": lo,
            "big_hate_gt70pct": big_hi, "big_normal_lt10pct": big_lo,
        })
        kept[thr] = (G, partition, hate_ratios, members)
        print(f"  thr={thr}: edges={ne} avgdeg={avg_deg:.1f} comms={n_comm} "
              f"Q={Q:.4f} | hate>70%:{hi} normal<10%:{lo}")

    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(os.path.join(RESULT_DIR, "community_threshold_sweep.csv"), index=False)

    # ----- choose best: highest modularity among thresholds with a non-trivial,
    #       not-fragmented community count (>=3 communities of size >=10) -----
    elig = sweep[sweep["comm_ge10"] >= 3]
    chosen = (elig if len(elig) else sweep).sort_values("modularity_Q", ascending=False).iloc[0]
    best_thr = float(chosen["threshold"])
    print(f"\n[sweep] chosen threshold = {best_thr} (modularity Q={chosen['modularity_Q']})")

    G, partition, hate_ratios, members = kept[best_thr]
    prof = profile_communities(df, G, partition, members, hate_ratios)
    prof.to_csv(os.path.join(RESULT_DIR, "community_profiles.csv"), index=False)
    save_meta = {"chosen_threshold": best_thr, "n_nodes": n,
                 "n_edges": int(G.number_of_edges()),
                 "n_communities": int(len(members)),
                 "modularity_Q": float(chosen["modularity_Q"])}
    with open(os.path.join(RESULT_DIR, "community_meta.json"), "w") as f:
        json.dump(save_meta, f, indent=2)

    # ----- sentiment gap (hate-heavy vs normal-heavy communities, size>=10) -----
    big = {c: m for c, m in members.items() if len(m) >= 10}
    hate_sent = [np.mean([SentimentIntensityAnalyzer().polarity_scores(t)["compound"]
                          for t in df.iloc[m].text]) for c, m in big.items() if hate_ratios[c] > 0.5]
    norm_sent = [np.mean([SentimentIntensityAnalyzer().polarity_scores(t)["compound"]
                          for t in df.iloc[m].text]) for c, m in big.items() if hate_ratios[c] < 0.1]
    gap_msg = "n/a"
    if hate_sent and norm_sent:
        gap_msg = (f"hate-heavy mean sentiment {np.mean(hate_sent):+.3f} vs "
                   f"normal-heavy {np.mean(norm_sent):+.3f} "
                   f"(gap {np.mean(norm_sent)-np.mean(hate_sent):+.3f})")

    # ----- figures: always 0.85, plus the chosen best -----
    if 0.85 in kept:
        G85, p85, hr85, _ = kept[0.85]
        draw_graph(df, G85, p85, hr85, 0.85,
                   os.path.join(FIG_DIR, "community_graph_0.85.png"),
                   f"Post-similarity graph @ cosine>=0.85 — {len(set(p85.values()))} communities")
    draw_graph(df, G, partition, hate_ratios, best_thr,
               os.path.join(FIG_DIR, "community_graph_best.png"),
               f"Post-similarity graph @ cosine>={best_thr} (chosen) — {len(members)} communities")

    print("\n===== ITEM 2: COMMUNITY THRESHOLD SWEEP =====")
    print(sweep.to_string(index=False))
    print(f"\nchosen threshold: {best_thr}")
    print(f"sentiment gap @ chosen: {gap_msg}")
    print("Saved: results/community_threshold_sweep.csv, results/community_profiles.csv, "
          "results/community_meta.json, figures/community_graph_0.85.png, figures/community_graph_best.png")
    # stash sentiment gap for the consolidated summary
    with open(os.path.join(RESULT_DIR, "community_sweep_summary.json"), "w") as f:
        json.dump({"chosen_threshold": best_thr, "sentiment_gap": gap_msg,
                   "sweep": sweep_rows}, f, indent=2)


if __name__ == "__main__":
    main()
