#!/bin/bash
set -euo pipefail
export PYTHONPATH=src

python -u -m qa_baseline.dev_score \
  --exports exports/v1/dev \
  --table exports/v1/dev/table.npz \
  --text-features exports/v1/dev/text_features.npz \
  --dev data/raw/dev-v1.1.json \
  --final exports/v1/final \
  --output-dir exports/v1/final/dev

for m in baseline length-penalty frozen-head-plus-mix; do
  python -u -m qa_baseline.evaluate \
    --data data/raw/dev-v1.1.json \
    --predictions "exports/v1/final/dev/predictions-$m.json" \
    --output "exports/v1/final/dev/metrics-$m.json"
done
