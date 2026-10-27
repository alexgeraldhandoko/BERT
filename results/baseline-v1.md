# baseline-v1: bert-base-uncased
- Held-out validation: EM 69.72886, F1 82.64001, 8741 questions
- Official dev: EM 79.66, F1 87.57, 10570 questions
- Config: configs/baseline.json (seed inside)
- Commit: c761a352fd02623b3a4eaf1b552b87945c923832
- Hardware: h100-47 x1 via Slurm (job 1015181)
- Note: pyarrow<21 was installed manually in the qa env to work with numpy 1.26.4
- Note: held-out validation questions have a single reference answer, so EM/F1 are lower than on official dev (3+ references per question).
- Commands: sbatch job_baseline.sbatch (runs train_baseline.sh)
