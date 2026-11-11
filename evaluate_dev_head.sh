#!/bin/bash
set -euo pipefail
export PYTHONPATH=src

python -u -m qa_baseline.evaluate_final_head \
  --exports exports/v1/dev-final \
  --head exports/v1/heldout/final/head_final.npz \
  --output exports/v1/dev-final/predictions-head-mix.json

python -u -m qa_baseline.evaluate \
  --data data/raw/dev-v1.1.json \
  --predictions exports/v1/dev-final/predictions.json \
  --output exports/v1/dev-final/metrics-baseline.json

python -u -m qa_baseline.evaluate \
  --data data/raw/dev-v1.1.json \
  --predictions exports/v1/dev-final/predictions-head-mix.json \
  --output exports/v1/dev-final/metrics-head-mix.json
