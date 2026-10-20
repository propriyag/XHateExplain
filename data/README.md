# Datasets

Datasets are **not** committed to this repository because of their licence terms.
Download them manually and place the files under this directory.

## HateXplain

- Source: https://github.com/hate-alert/HateXplain
- Expected path: `data/dataset.json`

```bash
mkdir -p data
curl -L -o data/dataset.json \
  https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json
```

`step1_data.py` consumes `data/dataset.json` directly and writes
`data/train.csv`, `data/val.csv`, `data/test.csv` locally (these are also
gitignored).

## Davidson 2017

- Source: https://github.com/t-davidson/hate-speech-and-offensive-language
- Expected path: `data/davidson_labeled_data.csv`

```bash
curl -L -o data/davidson_labeled_data.csv \
  https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv
```

Davidson is only used by `step16_ood_davidson.py` for the out-of-distribution check.

## Verification

After both downloads:

```bash
wc -l data/davidson_labeled_data.csv            # ~24,784
python -c "import json; print(len(json.load(open('data/dataset.json'))))"  # 20,148
```

## Ethics

Both corpora contain slurs. Store under restricted access (`chmod 700 data/`)
and never redistribute. Davidson has documented annotation bias against
African-American English (Sap et al., 2019).
