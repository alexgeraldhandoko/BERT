#!/bin/bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=src

python -u -m qa_baseline.export_candidates \
  --data data/raw/dev-v1.1.json \
  --model runs/baseline/model \
  --output-dir exports/v1/dev_edges \
  --reference runs/baseline/predictions.json
