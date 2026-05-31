#!/usr/bin/env bash
# Run the full XHateExplain pipeline in order. Each step writes to results/ and figures/.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
echo "== Step 1: data ==";        $PY step1_data.py
echo "== Step 2: BERT ==";        $PY step2_train_bert.py
echo "== Step 3: SHAP ==";        $PY step3_shap.py
echo "== Step 4: LIME ==";        $PY step4_lime.py
echo "== Step 5: Attention ==";   $PY step5_attention.py
echo "== Step 6: Community ==";    $PY step6_community.py
echo "== Step 8: Baselines ==";    $PY step8_baselines.py
echo "== Step 7: Tables/Figs ==";  $PY step7_tables_figures.py
echo "DONE — see results/ and figures/"
