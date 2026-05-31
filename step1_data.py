"""Step 1 - Download, parse, split and profile the HateXplain dataset.

Produces data/{train,val,test}.csv with columns:
    post_id, text, tokens (json), label, label_id,
    rationale_idx (json list of gold token indices), targets (json list)
"""
import json
import os
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd

from config import (HATEXPLAIN_URL, RAW_JSON, TRAIN_CSV, VAL_CSV, TEST_CSV,
                    LABELS, LABEL2ID, RAW2LABEL, SEED)
from utils import parse_rationale_tokens


def download():
    if os.path.exists(RAW_JSON):
        print(f"dataset.json already present ({os.path.getsize(RAW_JSON)/1e6:.1f} MB)")
        return
    print(f"Downloading HateXplain from {HATEXPLAIN_URL} ...")
    urllib.request.urlretrieve(HATEXPLAIN_URL, RAW_JSON)
    print(f"  saved {os.path.getsize(RAW_JSON)/1e6:.1f} MB -> {RAW_JSON}")


def majority_label(annotators):
    """Majority vote over 3 annotator labels. Returns None on a 3-way tie (undecided)."""
    votes = Counter(RAW2LABEL[a["label"]] for a in annotators)
    top, n = votes.most_common(1)[0]
    if n == 1:                       # all three differ -> undecided, drop
        return None
    return top


def aggregate_targets(annotators):
    """Union of non-'None' target groups across annotators."""
    t = set()
    for a in annotators:
        for g in a.get("target", []):
            if g and g.lower() != "none":
                t.add(g)
    return sorted(t)


def parse():
    with open(RAW_JSON) as f:
        raw = json.load(f)
    rows = []
    for pid, rec in raw.items():
        label = majority_label(rec["annotators"])
        if label is None:
            continue
        tokens = rec["post_tokens"]
        rows.append({
            "post_id": pid,
            "text": " ".join(tokens),
            "tokens": json.dumps(tokens),
            "label": label,
            "label_id": LABEL2ID[label],
            "rationale_idx": json.dumps(parse_rationale_tokens(rec.get("rationales", []), len(tokens))),
            "targets": json.dumps(aggregate_targets(rec["annotators"])),
        })
    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} posts (dropped 3-way-tie/undecided).")
    return df


def split(df):
    from sklearn.model_selection import train_test_split
    train, tmp = train_test_split(df, test_size=0.20, random_state=SEED,
                                  stratify=df["label_id"])
    val, test = train_test_split(tmp, test_size=0.50, random_state=SEED,
                                 stratify=tmp["label_id"])
    for d, p, name in [(train, TRAIN_CSV, "train"), (val, VAL_CSV, "val"), (test, TEST_CSV, "test")]:
        d.to_csv(p, index=False)
        print(f"  {name:5} {len(d):6} -> {p}")
    return train, val, test


def stats(name, df):
    print(f"\n=== {name} statistics (n={len(df)}) ===")
    dist = df["label"].value_counts()
    for lab in LABELS:
        c = int(dist.get(lab, 0))
        print(f"  {lab:10} {c:6}  ({100*c/len(df):5.1f}%)")
    lens = df["tokens"].apply(lambda t: len(json.loads(t)))
    print(f"  avg text length : {lens.mean():.1f} tokens (max {lens.max()})")
    has_rat = df["rationale_idx"].apply(lambda r: len(json.loads(r)) > 0)
    print(f"  rationale coverage: {100*has_rat.mean():.1f}% of posts have a gold rationale")
    toxic = df[df["label"] != "normal"]
    if len(toxic):
        tox_rat = toxic["rationale_idx"].apply(lambda r: len(json.loads(r)) > 0)
        print(f"     among toxic posts: {100*tox_rat.mean():.1f}% have a rationale")


def main():
    download()
    df = parse()
    stats("FULL", df)
    train, val, test = split(df)
    for name, d in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
        stats(name, d)


if __name__ == "__main__":
    main()
