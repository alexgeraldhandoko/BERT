#!/bin/bash
set -euo pipefail
export PYTHONPATH=src

python -u -m qa_baseline.cv_rerank \
  --exports exports/v1/heldout \
  --table exports/v1/heldout/table.npz \
  --validation runs/baseline/validation-v1.1.json \
  --text-features exports/v1/heldout/text_features.npz \
  --configs scalars,scalars+span,scalars+text,scalars+text+span \
  --l2-scalars 0.00001,0.0001,0.001,0.01 \
  --l2-text 0.001,0.01,0.1 \
  --folds 5 \
  --steps 1000 \
  --lr 0.02 \
  --output-dir exports/v1/heldout/cv_all_features
