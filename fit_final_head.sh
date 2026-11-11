#!/bin/bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=src
mkdir -p exports/v1/heldout/final

~/miniconda3/envs/qa/bin/python -u -m qa_baseline.fit_final_head \
  --exports exports/v1/heldout \
  --table exports/v1/heldout/table.npz \
  --output exports/v1/heldout/final/head_final.npz \
  --l2 0.03 \
  --steps 250 \
  --lr 0.02 \
  --alpha 0.55
