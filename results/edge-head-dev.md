# Edge-aware reranker: dev evaluation

## Frozen experiment

- Branch: `exp/start-end-readouts`
- Frozen tag: `final-edge-frozen-v1`
- Dataset: SQuAD v1.1; dev has 10,570 questions from 48 articles.
- Baseline checkpoint: `runs/baseline/model`
- Head: linear listwise reranker using 14 scalar, 98 text, 768 pooled-span, and 1,536 start/end vector features (2,416 total).
- Fit protocol: five article folds, seed 42; train on 7,100 held-out questions and select settings on calibration fold 0 (1,641 questions).
- Frozen settings: L2 0.3, step 550, alpha 0.5, length-penalty lambda 0.08.

## Official dev results

| Method                 |    EM |    F1 | Change vs baseline |
| ---------------------- | ----: | ----: | -----------------: |
| Baseline               | 79.66 | 87.57 |                  — |
| Length penalty         | 81.00 | 88.13 | +1.34 EM, +0.57 F1 |
| Frozen edge head + mix | 81.14 | 88.22 | +1.49 EM, +0.66 F1 |

Official scorer outputs (`vendor/evaluate-v2.0.py`, via `qa_baseline.evaluate`):

- Baseline: exact 79.6594, F1 87.5653, total 10,570.
- Length penalty: exact 81.0028, F1 88.1331, total 10,570.
- Frozen edge head + mix: exact 81.1447, F1 88.2232, total 10,570.

Article-bootstrap comparison:

- Edge head + mix vs baseline: ΔEM +1.49 [1.14, 1.84]; ΔF1 +0.66 [0.38, 0.93].
- Edge head + mix vs length penalty: ΔEM +0.14 [-0.18, 0.45]; ΔF1 +0.09 [-0.15, 0.31].

## Interpretation

The frozen edge head improves over baseline on dev. Its point estimates are slightly above the length penalty, but the paired intervals include zero, so this experiment does not establish that the head is better than the simpler penalty. Under the declared decision rule, prefer the length penalty. This dev comparison is exploratory because other variants were previously evaluated on dev; no settings were tuned on dev.

## Reproducibility

- Dev edge export: 10,570 questions, 10,784 windows, 211,400 candidates; rank-1 mismatch vs baseline: 0.
- Export job 1081102: elapsed 1:57; MaxRSS 5,164 MB.
- Dev table job 1081150: 10,570 questions, 211,400 candidates, 98 text features; table rank-1 EM/F1 matched the export.
- Environment: Python 3.12.15, NumPy 1.26.4, PyTorch 2.6.0, Transformers 4.52.4, Datasets 3.6.0, PyArrow 20.0.0.
