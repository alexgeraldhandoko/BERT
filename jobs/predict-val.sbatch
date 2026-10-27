#!/bin/bash
# Predict on the validation partition (8,741 questions) -> val-predictions.json
# + val-scores.npz. T for temperature scaling is fitted on these scores.
#SBATCH --job-name=cs4248-predict-val
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
  --data runs/baseline/validation-v1.1.json \
  --model runs/baseline/model \
  --output runs/baseline/val-predictions.json

# Score with the unmodified course scorer -> val-metrics.json (assertion 1 baseline).
PYTHONPATH=src python -m qa_baseline.evaluate \
  --data runs/baseline/validation-v1.1.json \
  --predictions runs/baseline/val-predictions.json \
  --output runs/baseline/val-metrics.json

echo "Validation prediction finished at $(date)"
