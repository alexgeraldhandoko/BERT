#!/bin/bash
#SBATCH --job-name=score_conf
#SBATCH --partition=normal
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/slurm/score_span_confidence-%j.out

set -euxo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

echo "Scoring frozen confidence head on dev in $(pwd)"
~/miniconda3/envs/qa/bin/python -u -m qa_baseline.score_span_confidence \
  --exports exports/v1/dev_edges \
  --head exports/v1/final_confidence/span_confidence_head.npz \
  --output-dir exports/v1/final_confidence/dev
