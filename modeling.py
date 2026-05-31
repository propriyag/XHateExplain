"""Shared transformer training / evaluation utilities used by Step 2 (BERT) and
Step 8 (HateBERT)."""
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import (MAX_LEN, BATCH_SIZE, LR, EPOCHS, SEED, LABELS,
                    ID2LABEL, LABEL2ID, get_device)
from utils import classification_report_dict


def set_seed(seed=SEED):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.enc = tokenizer(list(texts), truncation=True, padding="max_length",
                             max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def _run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss, preds, golds = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot_loss += loss.item() * batch["labels"].size(0)
            preds.extend(out.logits.argmax(-1).detach().cpu().tolist())
            golds.extend(batch["labels"].detach().cpu().tolist())
    from sklearn.metrics import accuracy_score, f1_score
    n = len(golds)
    return (tot_loss / n, accuracy_score(golds, preds),
            f1_score(golds, preds, average="macro"))


def train_transformer(model_name, out_dir, train_df, val_df,
                      epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, seed=SEED):
    """Fine-tune a HF sequence-classification model; save best by val macro-F1."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    set_seed(seed)
    device = get_device()
    print(f"[train] {model_name} on {device} | epochs={epochs} bs={batch_size} lr={lr}")

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)

    tr = DataLoader(TextDataset(train_df.text, train_df.label_id, tok), batch_size=batch_size, shuffle=True)
    va = DataLoader(TextDataset(val_df.text,   val_df.label_id,   tok), batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history, best_f1 = [], -1.0
    for ep in range(1, epochs + 1):
        trl, tra, trf = _run_epoch(model, tr, device, optimizer)
        val, vaa, vaf = _run_epoch(model, va, device)
        history.append({"epoch": ep, "train_loss": trl, "train_acc": tra, "train_f1": trf,
                        "val_loss": val, "val_acc": vaa, "val_f1": vaf})
        print(f"  epoch {ep}: train loss {trl:.4f} acc {tra:.4f} f1 {trf:.4f} | "
              f"val loss {val:.4f} acc {vaa:.4f} f1 {vaf:.4f}"
              + ("  <-- best" if vaf > best_f1 else ""))
        if vaf > best_f1:
            best_f1 = vaf
            os.makedirs(out_dir, exist_ok=True)
            model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
            with open(os.path.join(out_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)
    print(f"[train] best val macro-F1 = {best_f1:.4f} -> {out_dir}")
    return history


@torch.no_grad()
def predict_logits(model_dir, texts, batch_size=64, output_attentions=False):
    """Return logits (and optionally attentions) for a list of texts."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = get_device()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, output_attentions=output_attentions).to(device).eval()
    all_logits, all_att = [], []
    for i in range(0, len(texts), batch_size):
        chunk = list(texts[i:i + batch_size])
        enc = tok(chunk, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        out = model(**enc)
        all_logits.append(out.logits.cpu())
        if output_attentions:
            all_att.append((enc, out.attentions))
    logits = torch.cat(all_logits).numpy()
    return (logits, all_att) if output_attentions else logits


def make_predict_proba(model_dir, batch_size=64):
    """Return a `predict_proba(list[str]) -> np.ndarray[n, C]` callable (softmax)."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = get_device()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    @torch.no_grad()
    def predict_proba(texts):
        texts = list(texts)
        out = []
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], truncation=True, padding=True,
                      max_length=MAX_LEN, return_tensors="pt").to(device)
            out.append(torch.softmax(model(**enc).logits, -1).cpu().numpy())
        return np.concatenate(out, 0)
    return predict_proba


def evaluate(model_dir, df):
    """Full test evaluation -> (metrics dict, dataframe of predictions+probabilities)."""
    proba = make_predict_proba(model_dir)(list(df.text))
    y_pred = proba.argmax(1)
    y_true = df.label_id.to_numpy()
    metrics = classification_report_dict(y_true, y_pred, proba)
    pred_df = pd.DataFrame({
        "post_id": df.post_id.values, "text": df.text.values,
        "true": y_true, "pred": y_pred,
        **{f"p_{l}": proba[:, i] for i, l in enumerate(LABELS)},
    })
    return metrics, pred_df
