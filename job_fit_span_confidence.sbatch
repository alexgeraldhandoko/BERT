#!/bin/bash
#SBATCH --job-name=span_conf
#SBATCH --partition=normal
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/slurm/fit_span_confidence-%j.out

bash fit_span_confidence.sh
