#!/bin/bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=src

python -u -m qa_baseline.dev_table \
  --exports exports/v1/dev_edges \
  --dev data/raw/dev-v1.1.json \
  --output-dir exports/v1/dev_edges
