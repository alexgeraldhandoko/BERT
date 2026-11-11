#!/bin/bash
set -euo pipefail
export PYTHONPATH=src

python -u -m qa_baseline.cv_rerank \
  --exports exports/v1/heldout_edges \
  --table exports/v1/heldout_edges/table.npz \
  --validation runs/baseline/validation-v1.1.json \
  --text-features exports/v1/heldout_edges/text_features.npz \
  --configs scalars+text+span,scalars+text+span+edges \
  --output-dir exports/v1/heldout_edges/cv_start_end_l2wide
