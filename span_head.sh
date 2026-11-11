#!/bin/bash
set -e
export PYTHONPATH=src
python -m qa_baseline.span_head --exports exports/v1/heldout --table exports/v1/heldout/table.npz \
  --output-dir exports/v1/heldout/heads
