#!/bin/bash
set -e
export PYTHONPATH=src
python -u -m qa_baseline.final_head --exports exports/v1/heldout \
  --table exports/v1/heldout/table.npz --text-features exports/v1/heldout/text_features.npz \
  --validation runs/baseline/validation-v1.1.json --output-dir exports/v1/final
