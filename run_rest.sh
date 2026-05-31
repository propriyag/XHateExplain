#!/usr/bin/env bash
# Runs steps 3-8 after BERT training, each logging to results/. Stops on first error.
set -e
cd "$(dirname "$0")"
PY=".venv/bin/python -u"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
run () { echo "===== $1 ====="; $PY "$1" 2>&1 | tee "results/log_${1%.py}.txt"; }
run step3_shap.py
run step4_lime.py
run step5_attention.py
run step6_community.py
run step8_baselines.py
run step7_tables_figures.py
echo "ALL_REMAINING_DONE"
