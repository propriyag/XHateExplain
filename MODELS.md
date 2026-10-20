# Pre-trained models

The two fine-tuned BERT checkpoints that produced every classification number
in the paper are too large for GitHub (418 MB each) and are hosted separately.

## Checkpoints

| Model | Size | HuggingFace Hub |
|-------|------|-----------------|
| `bert_best` — `bert-base-uncased` fine-tuned on HateXplain (seed 42, best val macro-F1) | 418 MB | `priyankagupta/xhateexplain-bert` |
| `hatebert_best` — `GroNLP/hateBERT` fine-tuned on HateXplain (seed 42) | 418 MB | `priyankagupta/xhateexplain-hatebert` |

> Replace `priyankagupta` with your actual HuggingFace username after upload.

## Downloading

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="priyankagupta/xhateexplain-bert",
    local_dir="models/bert_best",
    local_dir_use_symlinks=False,
)
snapshot_download(
    repo_id="priyankagupta/xhateexplain-hatebert",
    local_dir="models/hatebert_best",
    local_dir_use_symlinks=False,
)
```

Or with the CLI:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download priyankagupta/xhateexplain-bert   --local-dir models/bert_best
huggingface-cli download priyankagupta/xhateexplain-hatebert --local-dir models/hatebert_best
```

## Loading in Python

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("models/bert_best")
model = AutoModelForSequenceClassification.from_pretrained("models/bert_best")
```

## Exact numbers these checkpoints produce

Loading `models/bert_best` and running `step7_tables_figures.py` yields the
Table 2 numbers verbatim:

| Metric | BERT | HateBERT |
|--------|------|----------|
| Accuracy | 0.701 ± 0.003 | 0.697 ± 0.008 |
| Macro-F1 | 0.688 ± 0.007 | 0.690 ± 0.008 |
| Hate-F1 | 0.768 | 0.767 |
| Offensive-F1 | 0.544 | 0.561 |
| Normal-F1 | 0.750 | 0.742 |
| AUC | 0.860 ± 0.002 | 0.855 ± 0.012 |

McNemar BERT vs HateBERT across three seeds: χ² = 0.033 / 0.003 / 1.458,
p = 0.86 / 0.96 / 0.23.

## Uploading (for the authors — one-time)

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login              # paste a WRITE token from huggingface.co/settings/tokens

huggingface-cli repo create xhateexplain-bert     --type model
huggingface-cli repo create xhateexplain-hatebert --type model

huggingface-cli upload priyankagupta/xhateexplain-bert     models/bert_best     .
huggingface-cli upload priyankagupta/xhateexplain-hatebert models/hatebert_best .
```

Each upload takes ~3–10 min depending on connection speed.
