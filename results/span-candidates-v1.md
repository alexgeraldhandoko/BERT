# span-candidates-v1: candidate export on held-out validation
- Code: tag span-candidates-v1 (commit e8c666a)
- Model: runs/baseline/model (bert-base-uncased, trained on the 90% training partition)
- Data: runs/baseline/validation-v1.1.json (8741 questions, one reference answer each)
- Command: sbatch job_export_heldout.sbatch (job 1044505), elapsed <00:01:14>
- Output: 8741 questions, 8869 windows, 174703 candidates (top 20 per question)
- Check: rank-1 EM 69.73 / F1 82.64; rank-1 answers differing from baseline predictions: 0 of 8741

| | EM | F1 |
|---|---|---|
| Oracle@1 | 69.73 | 82.64 |
| Oracle@5 | 88.79 | 93.52 |
| Oracle@10 | 92.47 | 95.65 |
| Oracle@20 | 95.09 | 97.20 |

Notes: candidates are deduplicated by character span; logp_all normalizes over all window positions, logp_ctx over context tokens only.