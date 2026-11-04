# Frozen reranking results

## Setup

- Baseline: BERT QA model at `runs/baseline/model`.
- Held-out data: 8,741 questions, split by article; five-fold cross-validation.
- Dev data: 10,570 questions, 48 articles.
- Frozen settings, tagged before dev scoring: `final-frozen-v1` (`0d8f794`).
- Head: `scalars+text+span`, L2 0.03, step 125, alpha 0.7.
- Length penalty: lambda 0.08.
- Dev metrics were checked with the official SQuAD scorer.

## Held-out cross-validation

| Method                      |    EM |    F1 | Change vs baseline                              |
| --------------------------- | ----: | ----: | ----------------------------------------------- |
| Baseline                    | 69.73 | 82.64 | —                                               |
| Length penalty              | 70.44 | 82.64 | dEM +0.71 [0.08, 1.28]; dF1 -0.00 [-0.56, 0.41] |
| Scalars + text + span + mix | 70.76 | 82.78 | dEM +1.03 [0.45, 1.66]; dF1 +0.14 [-0.32, 0.61] |

Intervals are 95% article-level bootstrap intervals. The head met the predeclared rule against the baseline in cross-validation.

## Dev comparison

| Method            |    EM |    F1 | Change vs baseline |
| ----------------- | ----: | ----: | ------------------ |
| Baseline          | 79.66 | 87.57 | —                  |
| Length penalty    | 81.00 | 88.13 | +1.34 EM, +0.57 F1 |
| Frozen head + mix | 80.92 | 88.02 | +1.26 EM, +0.45 F1 |

Head minus length penalty: dEM -0.09 [-0.50, 0.30]; dF1 -0.12 [-0.46, 0.20]. These intervals include zero, so the dev results do not show that the head beats the simpler penalty.

## Commands

- Cross-validation: `job_cv_all_features.sbatch`
- Freeze settings: `job_freeze.sbatch`
- Score dev and run the official scorer: `job_dev_score.sbatch`

The dev comparison uses frozen settings.
