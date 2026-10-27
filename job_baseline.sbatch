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
#SBATCH --job-name=baseline

# Queue in which this job should run.
#SBATCH --partition=gpu-long

# Stop the job after five mins if it has not finished.
#SBATCH --time=03:00:00

# Request one NVIDIA H100 GPU with 47 GB of memory.
#SBATCH --gpus=h100-47:1

# Reserve eight CPU cores for the task.
#SBATCH --cpus-per-task=4

# Request 20,000 MB of memory for each reserved CPU core.
#SBATCH --mem-per-cpu=8000

# Exit on errors, unset variables, or failures within a pipeline.
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration: change these values for your project.
# ---------------------------------------------------------------------------

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-qa}"

PIPELINE_SCRIPT="train_baseline.sh"

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
