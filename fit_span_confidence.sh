#!/bin/bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

~/miniconda3/envs/qa/bin/python -u -m qa_baseline.fit_span_confidence \
  --examples exports/v1/heldout_edges/confidence_examples.npz \
  --output-dir exports/v1/final_confidence \
  --l2 0.01 --steps 500 --lr 0.01
