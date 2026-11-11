#!/bin/bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=src

python -u -m qa_baseline.dev_score \
  --exports exports/v1/dev_edges \
  --table exports/v1/dev_edges/table.npz \
  --text-features exports/v1/dev_edges/text_features.npz \
  --dev data/raw/dev-v1.1.json \
  --final exports/v1/final_edges \
  --output-dir exports/v1/final_edges/dev

for m in baseline length-penalty frozen-head-plus-mix; do
  python -u -m qa_baseline.evaluate \
    --data data/raw/dev-v1.1.json \
    --predictions "exports/v1/final_edges/dev/predictions-${m}.json" \
    --output "exports/v1/final_edges/dev/metrics-${m}.json"
done
