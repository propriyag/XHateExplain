#!/usr/bin/env bash
# Re-run the three explainers (now that the get_subsample bug is fixed) plus step7.
cd "$(dirname "$0")"
set -o pipefail
PY=".venv/bin/python -u"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
run () {
  echo "===== $1 ($(date +%H:%M:%S)) ====="
  $PY "$1" 2>&1 | tee "results/log_${1%.py}.txt"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then echo "[rerun] $1 FAILED (rc=$rc) — aborting"; exit "$rc"; fi
}
run step3_shap.py
run step4_lime.py
run step5_attention.py     # also builds the combined XHateExplain explainer
run step7_tables_figures.py
echo "RERUN_COMPLETE $(date)"
