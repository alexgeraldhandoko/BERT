#!/bin/bash
# Predict on dev (10,570 questions) -> predictions.json + dev-scores.npz.
# One-shot: dev is reported on, never fitted on (HANDOFF gotcha 10).
#SBATCH --job-name=cs4248-predict-dev
#SBATCH --partition=gpu-long
#SBATCH --time=02:00:00
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20000

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="$REPO_ROOT/logs/slurm"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-$$}.out" 2>&1
cd "$REPO_ROOT"

set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate bert4248
set -u

echo "Job started on $(hostname) at $(date)"
nvidia-smi || echo "WARNING: no GPU visible"

PYTHONPATH=src python -m qa_baseline.predict \
  --config configs/baseline.json \
  --data data/raw/dev-v1.1.json \
  --model runs/baseline/model \
  --output runs/baseline/predictions.json

echo "Dev prediction finished at $(date)"
