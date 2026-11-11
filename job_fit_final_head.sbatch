#!/bin/bash

# Template for running a pipeline script as a Slurm batch job.
#
# Submit this file from the repository root:
#   sbatch path/to/job_template.sbatch
#
# Extra arguments supplied to sbatch are forwarded to the pipeline script:
#   sbatch path/to/job_template.sbatch \
#       --num_shards 4 --shard_id 0

# Name displayed by Slurm commands such as squeue.
#SBATCH --job-name=fit_final_head

# Queue in which this job should run.
#SBATCH --partition=normal

# Stop the job after five mins if it has not finished.
#SBATCH --time=01:00:00

# CPU-only NumPy job; no GPU requested.

# Reserve four CPU cores for the task.
#SBATCH --cpus-per-task=4

# Request 8,000 MB per CPU core (32,000 MB total).
#SBATCH --mem-per-cpu=8000

# Exit on errors, unset variables, or failures within a pipeline.
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration: change these values for your project.
# ---------------------------------------------------------------------------

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-qa}"

PIPELINE_SCRIPT="fit_final_head.sh"

# ---------------------------------------------------------------------------
# Repository and logging.
# ---------------------------------------------------------------------------

# Slurm runs a spooled copy of this script from a temporary directory.
# SLURM_SUBMIT_DIR is the directory from which sbatch was invoked.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"

LOG_DIR="$REPO_ROOT/logs/slurm"
mkdir -p "$LOG_DIR"

# Write standard output and standard error to the same job-specific log file.
exec > "$LOG_DIR/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-$$}.out" 2>&1

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Environment setup.
# ---------------------------------------------------------------------------

# Non-interactive Slurm jobs do not normally source ~/.bashrc, so initialize
# Conda explicitly before activating the environment. Some environments refer
# to unset variables during activation, so temporarily disable `set -u`.
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

# ---------------------------------------------------------------------------
# Run the project pipeline and forward all additional sbatch arguments.
# ---------------------------------------------------------------------------

bash "$PIPELINE_SCRIPT" "$@"
