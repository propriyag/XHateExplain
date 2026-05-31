"""Central configuration for the XHateExplain pipeline."""
import os

# ----------------------------------------------------------------------------- paths
ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(ROOT, "data")
MODEL_DIR  = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")
FIG_DIR    = os.path.join(ROOT, "figures")
for _d in (DATA_DIR, MODEL_DIR, RESULT_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)

# raw dataset
HATEXPLAIN_URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
RAW_JSON       = os.path.join(DATA_DIR, "dataset.json")

# split csvs (produced by step1)
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV   = os.path.join(DATA_DIR, "val.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")

# ----------------------------------------------------------------------------- labels
LABELS    = ["hate", "offensive", "normal"]          # index 0,1,2
LABEL2ID  = {l: i for i, l in enumerate(LABELS)}
ID2LABEL  = {i: l for i, l in enumerate(LABELS)}
# raw annotator label -> our canonical label
RAW2LABEL = {"hatespeech": "hate", "offensive": "offensive", "normal": "normal"}

# ----------------------------------------------------------------------------- model / training
BERT_MODEL    = "bert-base-uncased"
HATEBERT_MODEL = "GroNLP/hateBERT"
MAX_LEN       = 128
BATCH_SIZE    = 32
LR            = 2e-5
EPOCHS        = 5
SEED          = 42

BEST_BERT_DIR     = os.path.join(MODEL_DIR, "bert_best")
BEST_HATEBERT_DIR = os.path.join(MODEL_DIR, "hatebert_best")
BILSTM_PATH       = os.path.join(MODEL_DIR, "bilstm_attn.pt")

# ----------------------------------------------------------------------------- explainability
# Full SHAP/LIME over the whole test set is hours of compute. We evaluate on a
# representative stratified subsample (set to None to run the entire test set).
EXPLAIN_SUBSAMPLE = 300
TOPK = 5                       # top-k tokens used for plausibility / AOPC
AOPC_KS = [1, 2, 3, 4, 5, 10]  # k values for AOPC curve

# ----------------------------------------------------------------------------- community analysis
SIM_THRESHOLD   = 0.7          # cosine-similarity edge threshold
LDA_TOPICS      = 5
COMMUNITY_SUBSAMPLE = 2000     # posts used to build the graph (full set is ~20k -> 400M-edge dense sim)


def get_device():
    """Return the best available torch device (cuda > mps > cpu)."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
