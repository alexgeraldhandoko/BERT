#!/bin/bash
# One-paste pipeline launcher: submit train, then chain the two prediction
# jobs so they run automatically (and in order) after training succeeds.
# Slurm keeps the jobs running even after the terminal is closed.
#
# Usage (from anywhere; repo lives at ~/BERT):
#   bash ~/BERT/jobs/run_all.sh
set -euo pipefail

REPO="${REPO:-$HOME/BERT}"
cd "$REPO"

git pull --ff-only
echo "--- submitting pipeline ---"

TRAIN_JOB=$(sbatch --parsable jobs/train.sbatch)
echo "train:        $TRAIN_JOB"

VAL_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB jobs/predict-val.sbatch)
echo "predict-val:  $VAL_JOB (afterok:$TRAIN_JOB)"

DEV_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB jobs/predict-dev.sbatch)
echo "predict-dev:  $DEV_JOB (afterok:$TRAIN_JOB)"

echo "--- done ---"
echo "Watch:  squeue --me"
echo "Logs:   tail -f $REPO/logs/slurm/cs4248-train-${TRAIN_JOB}.out"
echo "The two prediction jobs start automatically once training completes."
