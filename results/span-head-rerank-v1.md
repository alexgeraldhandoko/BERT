# Span-head reranking v1

- Branch: `span-head`
- Code commit: <fill with the revision used for this run>
- Hypothesis: A learned head using scalar candidate features and mean-pooled span vectors can rerank BERT's top-20 candidates.
- Data: SQuAD v1.1. Held-out set: 8,741 questions, article split seed 42; 5-fold CV across 45 articles. Final dev check: 10,570 questions.
- Baseline model: `runs/baseline/model`
- Baseline config: `configs/baseline.json`
- Final head checkpoint: `exports/v1/heldout/final/head_final.npz` (kept outside Git)
- Head settings: `scalars+span`, L2 0.03, 250 steps, learning rate 0.02, alpha 0.55.
- Scorer: `vendor/evaluate-v2.0.py`

## Held-out cross-validation

| System               |    EM |    F1 | Change in EM | Change in F1 |
| -------------------- | ----: | ----: | -----------: | -----------: |
| Baseline             | 69.73 | 82.64 |            — |            — |
| Scalars + span + mix | 70.75 | 82.87 |        +1.02 |        +0.23 |

Article-bootstrap intervals: ΔEM `[+0.54, +1.48]`; ΔF1 `[-0.11, +0.53]`.
The F1 interval includes zero.

## Final dev comparison

| System          |      EM |      F1 |
| --------------- | ------: | ------: |
| Baseline        | 79.6594 | 87.5653 |
| Frozen head mix | 80.9650 | 88.1851 |
| Change          | +1.3056 | +0.6198 |

The baseline result matched the previously reported dev score. Dev was used for this final comparison and was not used to tune the head or alpha.

## Fit and diagnostic details

- Final fit: 174,703 candidate rows from 8,736 questions; 5 questions had no candidates.
- Final-fit training loss: 0.720599 (training diagnostic, not EM/F1).
- Final-fit Slurm job: 1060361; `normal`, 4 CPUs, 24 seconds, peak RSS about 2 GB.
- Boundary diagnostic: 1,666 rank-1 errors had a correct candidate within the top 5. This is an oracle opportunity count, not a measured reranking gain.

## Reproduction

- Final fit: `sbatch job_fit_final_head.sbatch` (job 1060361)
- Dev evaluation: `sbatch job_evaluate_dev_head.sbatch` (job 1064318)
- Python/package versions: <fill from the `qa` environment>
