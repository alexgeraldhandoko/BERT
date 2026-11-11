#!/bin/bash
set -e
export PYTHONPATH=src
python -m qa_baseline.export_candidates --data data/raw/dev-v1.1.json --model runs/baseline/model \
  --output-dir exports/v1/dev --reference runs/baseline/predictions.json
