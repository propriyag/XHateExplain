"""Step 2 - Fine-tune bert-base-uncased for 3-class hate/offensive/normal
classification, evaluate on the test set, and save predictions + probabilities
for downstream explainability analysis.
"""
import os
import pandas as pd

from config import (TRAIN_CSV, VAL_CSV, TEST_CSV, BERT_MODEL, BEST_BERT_DIR, RESULT_DIR)
from modeling import train_transformer, evaluate
from utils import save_json


def main():
    train = pd.read_csv(TRAIN_CSV)
    val   = pd.read_csv(VAL_CSV)
    test  = pd.read_csv(TEST_CSV)

    history = train_transformer(BERT_MODEL, BEST_BERT_DIR, train, val)

    print("\n[eval] test set with best model ...")
    metrics, preds = evaluate(BEST_BERT_DIR, test)
    print("  test metrics:")
    for k, v in metrics.items():
        print(f"    {k:12} {v:.4f}")

    save_json({"model": "BERT", "history": history, "test": metrics},
              os.path.join(RESULT_DIR, "bert_metrics.json"))
    pred_path = os.path.join(RESULT_DIR, "bert_test_predictions.csv")
    preds.to_csv(pred_path, index=False)
    print(f"  saved predictions+probabilities -> {pred_path}")


if __name__ == "__main__":
    main()
