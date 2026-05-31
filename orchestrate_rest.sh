#!/usr/bin/env bash
# Wait for the running BERT training (step2) to finish, then run steps 3-8.
cd "$(dirname "$0")"
BERT_PID=92412
echo "[orchestrate] waiting for BERT training (PID $BERT_PID) to finish ..."
while kill -0 "$BERT_PID" 2>/dev/null; do sleep 20; done
echo "[orchestrate] BERT process exited at $(date)"

# Confirm step2 produced its outputs before continuing.
if [ ! -f results/bert_metrics.json ] || [ ! -f results/bert_test_predictions.csv ]; then
  echo "[orchestrate] ERROR: step2 outputs missing (bert_metrics.json / bert_test_predictions.csv)."
  echo "[orchestrate] Training may have crashed before the test-eval stage. Aborting."
  ls -la results/
  exit 1
fi
echo "[orchestrate] step2 outputs present. Launching steps 3-8 ..."
bash run_rest.sh
echo "[orchestrate] PIPELINE COMPLETE at $(date)"
