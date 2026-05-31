"""Step 6 - Community-level analysis.

NOTE: HateXplain contains no user/author field (only posts, tokens, annotator
labels, targets and rationales). We therefore build a POST-level interaction
graph: each post is a node, and two posts are linked when the cosine similarity
of their fine-tuned-BERT embeddings >= SIM_THRESHOLD. Louvain communities are
detected on this graph; "key users" become the most central posts. Everything
else (LDA topics, VADER sentiment, betweenness, hate ratio) is exactly as
specified.
"""
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

import community as community_louvain
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import (TRAIN_CSV, BEST_BERT_DIR, MAX_LEN, SIM_THRESHOLD, LDA_TOPICS,
                    COMMUNITY_SUBSAMPLE, SEED, RESULT_DIR, FIG_DIR, get_device)
from utils import save_json


def bert_embeddings(texts, batch_size=64):
    """Mean-pooled last-hidden-state embeddings from the fine-tuned encoder."""
    from transformers import AutoTokenizer, AutoModel
    device = get_device()
    tok = AutoTokenizer.from_pretrained(BEST_BERT_DIR)
    enc_model = AutoModel.from_pretrained(BEST_BERT_DIR).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(list(texts[i:i + batch_size]), truncation=True, padding=True,
                      max_length=MAX_LEN, return_tensors="pt").to(device)
            h = enc_model(**enc).last_hidden_state            # (b, t, d)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(emb.cpu().numpy())
    return np.concatenate(out, 0)


def lda_keywords(texts, n_topics=LDA_TOPICS, n_top=8):
    if len(texts) < max(3, n_topics):
        return []
    cv = CountVectorizer(max_df=0.95, min_df=1, stop_words="english")
    try:
        X = cv.fit_transform(texts)
    except ValueError:
        return []
    if X.shape[1] < n_topics:
        return []
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=SEED, max_iter=15)
    lda.fit(X)
    vocab = np.array(cv.get_feature_names_out())
    topics = []
    for comp in lda.components_:
        topics.append(", ".join(vocab[np.argsort(comp)[::-1][:n_top]]))
    return topics


def main():
    df = pd.read_csv(TRAIN_CSV)
    if COMMUNITY_SUBSAMPLE and COMMUNITY_SUBSAMPLE < len(df):
        df = df.sample(COMMUNITY_SUBSAMPLE, random_state=SEED).reset_index(drop=True)
    print(f"[Community] {len(df)} posts -> embeddings ...")
    emb = bert_embeddings(list(df.text))

    # --- similarity graph ---
    print(f"[Community] building graph (cosine >= {SIM_THRESHOLD}) ...")
    sim = cosine_similarity(emb)
    np.fill_diagonal(sim, 0.0)
    G = nx.Graph()
    G.add_nodes_from(range(len(df)))
    iu = np.triu_indices(len(df), k=1)
    for i, j in zip(*iu):
        if sim[i, j] >= SIM_THRESHOLD:
            G.add_edge(int(i), int(j), weight=float(sim[i, j]))
    print(f"  nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    # --- Louvain ---
    partition = community_louvain.best_partition(G, random_state=SEED)
    df["community"] = [partition[i] for i in range(len(df))]
    n_comm = len(set(partition.values()))
    print(f"  detected {n_comm} communities")

    # --- per-community profiles ---
    vader = SentimentIntensityAnalyzer()
    members = defaultdict(list)
    for node, c in partition.items():
        members[c].append(node)

    profiles = []
    for c, nodes in members.items():
        sub = df.iloc[nodes]
        hate_ratio = float((sub.label == "hate").mean())
        sentiment = float(np.mean([vader.polarity_scores(t)["compound"] for t in sub.text]))
        topics = lda_keywords(list(sub.text))
        # betweenness centrality within this community's subgraph -> key posts
        subG = G.subgraph(nodes)
        if subG.number_of_edges() > 0:
            bc = nx.betweenness_centrality(subG)
            key_nodes = sorted(bc, key=bc.get, reverse=True)[:3]
        else:
            key_nodes = nodes[:3]
        key_posts = [df.iloc[n].post_id for n in key_nodes]
        profiles.append({
            "community": int(c), "size": len(nodes),
            "hate_ratio": round(hate_ratio, 3),
            "toxic_ratio": round(float((sub.label != "normal").mean()), 3),
            "avg_sentiment": round(sentiment, 3),
            "top_topics": " | ".join(topics[:2]) if topics else "(too small for LDA)",
            "key_posts": ", ".join(p[:14] for p in key_posts),
        })

    prof = pd.DataFrame(profiles).sort_values("size", ascending=False).reset_index(drop=True)
    csv_path = os.path.join(RESULT_DIR, "community_profiles.csv")
    prof.to_csv(csv_path, index=False)
    print(f"  saved {csv_path}")
    print(prof.head(5).to_string(index=False))
    save_json({"n_communities": n_comm, "n_nodes": G.number_of_nodes(),
               "n_edges": G.number_of_edges()},
              os.path.join(RESULT_DIR, "community_meta.json"))

    # --- graph visualisation coloured by hate ratio ---
    import matplotlib.pyplot as plt
    print("[Community] rendering graph ...")
    hate_of_comm = {p["community"]: p["hate_ratio"] for p in profiles}
    node_color = [hate_of_comm[partition[i]] for i in range(len(df))]
    pos = nx.spring_layout(G, seed=SEED, k=0.15, iterations=30)
    fig, ax = plt.subplots(figsize=(11, 9))
    nx.draw_networkx_edges(G, pos, alpha=0.05, ax=ax)
    sc = nx.draw_networkx_nodes(G, pos, node_size=18, node_color=node_color,
                                cmap="Reds", vmin=0, vmax=1, ax=ax)
    plt.colorbar(sc, ax=ax, label="community hate ratio")
    ax.set_title(f"Post-similarity graph — {n_comm} Louvain communities "
                 f"(coloured by hate ratio)")
    ax.axis("off"); fig.tight_layout()
    fp = os.path.join(FIG_DIR, "community_graph.png")
    fig.savefig(fp, dpi=300); plt.close(fig)
    print(f"  saved {fp}")


if __name__ == "__main__":
    main()
