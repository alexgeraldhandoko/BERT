#!/bin/bash
set -euo pipefail
export PYTHONPATH=src

python -u -m qa_baseline.export_candidates \
  --data runs/baseline/validation-v1.1.json \
  --model runs/baseline/model \
  --output-dir exports/v1/heldout_edges \
  --reference runs/baseline/validation-predictions.json
