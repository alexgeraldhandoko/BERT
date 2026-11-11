#!/bin/bash
#SBATCH --job-name=fit_final_edges
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

LOG_DIR="$PWD/logs/slurm"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/${SLURM_JOB_NAME:-fit_final_edges}-${SLURM_JOB_ID:-$$}.out" 2>&1

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate qa
set -u

export PYTHONPATH=src
python -u -m qa_baseline.final_head \
  --exports exports/v1/heldout_edges \
  --table exports/v1/heldout_edges/table.npz \
  --text-features exports/v1/heldout_edges/text_features.npz \
  --validation runs/baseline/validation-v1.1.json \
  --output-dir exports/v1/final_edges \
  --config scalars+text+span+edges \
  --l2 0.03,0.1,0.3,1,3 \
  --steps 1000 \
  --lr 0.02
