### Post-hoc span-conditioned confidence head

- **Branch:** `exp/span-conditioned-confidence`
- **Baseline:** frozen BERT QA checkpoint; no BERT fine-tuning in this experiment.
- **Features:** `[CLS; mean-pooled rank-1 span; candidate-present flag]` (1,537 values).
- **Target:** whether the baseline rank-1 answer is an exact match.
- **Data:** train confidence head on 5,845 held-out questions (4,114 exact matches); fit temperature on the disjoint 2,896-question calibration set (1,981 exact matches).
- **Head settings:** logistic head, L2 = 0.01, 500 steps, learning rate = 0.01.
- **Fitted temperature:** 1.5402.
- **Dev:** 10,570 questions. Rank-1 EM/F1 remained 79.66 / 87.57 because confidence scoring does not change answers.

| Dev confidence score             | AUROC ↑ |  NLL ↓ | Brier ↓ | ECE ↓ (15 bins) |
| -------------------------------- | ------: | -----: | ------: | --------------: |
| Raw endpoint probability product |  0.8354 | 0.4582 |  0.1535 |          0.1549 |
| Span head before temperature     |  0.7756 | 0.5030 |  0.1641 |          0.1311 |
| Span head after temperature      |  0.7756 | 0.4973 |  0.1635 |          0.1592 |

**Interpretation:** The raw endpoint score had better AUROC, NLL, and Brier score. The uncalibrated span head had lower ECE, but after temperature scaling its dev ECE was higher. Overall, this run does not show a consistent confidence-metric advantage for the span head. The temperature was fitted on the calibration set, not dev.

**Comparison limitation:** The raw endpoint product was not temperature-scaled in this run.
