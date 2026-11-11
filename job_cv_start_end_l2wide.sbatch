#!/bin/bash
#SBATCH --job-name=cv_start_end_l2wide
#SBATCH --partition=normal
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH --output=logs/slurm/cv_start_end_l2wide-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=src

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-qa}"
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

bash cv_start_end_l2wide.sh
