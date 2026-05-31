"""Step 8 - Baseline comparisons.

1. BiLSTM + Attention  (GloVe 300d embeddings)
2. HateBERT            (GroNLP/hateBERT, fine-tuned identically to BERT)
3. Combined comparison table
4. McNemar's test between the main model (BERT) and the best baseline.
"""
import json
import os
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (TRAIN_CSV, VAL_CSV, TEST_CSV, DATA_DIR, HATEBERT_MODEL,
                    BEST_HATEBERT_DIR, BEST_BERT_DIR, BILSTM_PATH, RESULT_DIR,
                    LABELS, MAX_LEN, SEED, get_device)
from modeling import train_transformer, evaluate, make_predict_proba, set_seed
from utils import classification_report_dict, save_json

GLOVE_ZIP = os.path.join(DATA_DIR, "glove.6B.zip")
GLOVE_TXT = os.path.join(DATA_DIR, "glove.6B.300d.txt")
EMB_DIM, HIDDEN, BL_EPOCHS, BL_BS, BL_LR = 300, 256, 8, 64, 1e-3


# ----------------------------------------------------------------- GloVe / vocab
def ensure_glove():
    if os.path.exists(GLOVE_TXT):
        return
    print("Extracting glove.6B.300d.txt ...")
    with zipfile.ZipFile(GLOVE_ZIP) as z:
        z.extract("glove.6B.300d.txt", DATA_DIR)


def build_vocab(token_lists, min_freq=1):
    from collections import Counter
    c = Counter(t for toks in token_lists for t in toks)
    itos = ["<pad>", "<unk>"] + [w for w, f in c.items() if f >= min_freq]
    return {w: i for i, w in enumerate(itos)}, itos


def load_glove_matrix(stoi):
    """Build [V, 300] matrix, loading only vectors for words in our vocab."""
    ensure_glove()
    mat = np.random.normal(0, 0.1, (len(stoi), EMB_DIM)).astype(np.float32)
    mat[0] = 0.0
    found = 0
    with open(GLOVE_TXT, encoding="utf8") as f:
        for line in f:
            w, *vec = line.rstrip().split(" ")
            if w in stoi:
                mat[stoi[w]] = np.asarray(vec, dtype=np.float32); found += 1
    print(f"  GloVe coverage: {found}/{len(stoi)} vocab words")
    return torch.tensor(mat)


def encode(token_lists, stoi, max_len=MAX_LEN):
    out = np.zeros((len(token_lists), max_len), dtype=np.int64)
    for i, toks in enumerate(token_lists):
        for j, t in enumerate(toks[:max_len]):
            out[i, j] = stoi.get(t, 1)            # <unk>=1
    return torch.tensor(out)


# ----------------------------------------------------------------- BiLSTM + Attention
class BiLSTMAttn(nn.Module):
    def __init__(self, emb_matrix, hidden=HIDDEN, n_classes=len(LABELS)):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(emb_matrix, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(EMB_DIM, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * hidden, 1)
        self.fc = nn.Linear(2 * hidden, n_classes)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)             # (b, t, 1)
        h, _ = self.lstm(self.emb(x))             # (b, t, 2H)
        scores = self.attn(h).masked_fill(~mask, -1e9)
        alpha = torch.softmax(scores, dim=1)      # (b, t, 1)
        ctx = (alpha * h).sum(1)                  # (b, 2H)
        return self.fc(self.drop(ctx))


def train_bilstm(train, val, test):
    set_seed(); device = get_device()
    tok_tr = [json.loads(t) for t in train.tokens]
    stoi, _ = build_vocab(tok_tr)
    emb = load_glove_matrix(stoi)
    Xtr, Xva, Xte = (encode([json.loads(t) for t in d.tokens], stoi) for d in (train, val, test))
    ytr = torch.tensor(train.label_id.values)
    yva = torch.tensor(val.label_id.values)

    model = BiLSTMAttn(emb).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=BL_LR)
    crit = nn.CrossEntropyLoss()
    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BL_BS, shuffle=True)

    from sklearn.metrics import f1_score
    best_f1 = -1
    print(f"[BiLSTM+Attn] training on {device} ...")
    for ep in range(1, BL_EPOCHS + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xva.to(device)).argmax(1).cpu().numpy()
        vf1 = f1_score(yva.numpy(), vp, average="macro")
        print(f"  epoch {ep}: val macro-F1 {vf1:.4f}" + ("  <-- best" if vf1 > best_f1 else ""))
        if vf1 > best_f1:
            best_f1 = vf1
            torch.save({"state": model.state_dict(), "stoi": stoi}, BILSTM_PATH)

    # test eval with best
    ck = torch.load(BILSTM_PATH, weights_only=False)
    model.load_state_dict(ck["state"]); model.eval()
    with torch.no_grad():
        logits = model(Xte.to(device)).cpu()
        proba = torch.softmax(logits, -1).numpy()
    y_pred = proba.argmax(1); y_true = test.label_id.to_numpy()
    metrics = classification_report_dict(y_true, y_pred, proba)
    return metrics, y_pred


# ----------------------------------------------------------------- McNemar
def mcnemar(y_true, pred_a, pred_b):
    """McNemar's test on the two models' correctness vectors."""
    from scipy.stats import chi2
    a_ok = pred_a == y_true; b_ok = pred_b == y_true
    n01 = int(np.sum(a_ok & ~b_ok))      # A right, B wrong
    n10 = int(np.sum(~a_ok & b_ok))      # A wrong, B right
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if (n01 + n10) else 0.0
    p = float(chi2.sf(stat, 1))
    return {"n01_only_A_right": n01, "n10_only_B_right": n10,
            "chi2": round(float(stat), 4), "p_value": round(p, 5),
            "significant_0.05": p < 0.05}


def main():
    train, val, test = (pd.read_csv(p) for p in (TRAIN_CSV, VAL_CSV, TEST_CSV))
    y_true = test.label_id.to_numpy()
    results = {}

    # 1) BiLSTM + Attention
    bl_metrics, bl_pred = train_bilstm(train, val, test)
    results["BiLSTM+Attention"] = bl_metrics
    print("  BiLSTM test:", {k: round(v, 3) for k, v in bl_metrics.items()})

    # 2) HateBERT
    print("\n[HateBERT] fine-tuning ...")
    train_transformer(HATEBERT_MODEL, BEST_HATEBERT_DIR, train, val)
    hb_metrics, hb_pred = evaluate(BEST_HATEBERT_DIR, test)
    hb_pred = hb_pred.pred.to_numpy()
    results["HateBERT"] = hb_metrics
    print("  HateBERT test:", {k: round(v, 3) for k, v in hb_metrics.items()})

    # 3) main model (BERT) predictions for comparison + McNemar
    bert_pred = pd.read_csv(os.path.join(RESULT_DIR, "bert_test_predictions.csv")).pred.to_numpy()
    bert_metrics = json.load(open(os.path.join(RESULT_DIR, "bert_metrics.json")))["test"]
    results["BERT (main)"] = bert_metrics

    # best baseline by macro-F1
    baselines = {"BiLSTM+Attention": (bl_metrics, bl_pred), "HateBERT": (hb_metrics, hb_pred)}
    best_name = max(baselines, key=lambda k: baselines[k][0]["macro_f1"])
    mc = mcnemar(y_true, bert_pred, baselines[best_name][1])
    print(f"\nMcNemar BERT vs best baseline ({best_name}): {mc}")

    save_json({"results": results, "best_baseline": best_name, "mcnemar": mc},
              os.path.join(RESULT_DIR, "baselines.json"))


if __name__ == "__main__":
    main()
