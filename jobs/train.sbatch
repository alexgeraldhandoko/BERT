#!/bin/bash
# Fine-tune the BERT QA baseline on SQuAD 1.1 (78,858-question split, no --full-train:
# the 8,741 validation rows must stay out of training — see HANDOFF decision 4).
#SBATCH --job-name=cs4248-train
#SBATCH --partition=gpu-long
#SBATCH --time=08:00:00
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
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

PYTHONPATH=src python -m qa_baseline.train \
  --config configs/baseline.json \
  --data data/raw/train-v1.1.json \
  --output runs/baseline

echo "Training finished at $(date)"
ls -la runs/baseline/model
