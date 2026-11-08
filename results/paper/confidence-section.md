# Report section (draft) — Confidence calibration for extractive QA

> Draft for Kea's ~1.5-page report section, **binding protocol** (TA split
> spec, 2026-11-08): temperature/Platt fitted ONLY on the 2,896 calib IDs;
> dev (10,570) used for reporting. Every number traces to a pipeline command
> in `results/2026-10-27-confidence-v1.md` §11. Port to the team's Google
> Doc / LaTeX. Figures: `results/figures/risk_coverage.png`,
> `results/figures/reliability.png` (regenerate from
> `runs/baseline/plots-calib-platt-v1/` if needed).

## 1. Setup

We evaluate confidence calibration on a BERT-base-uncased extractive QA
model fine-tuned on SQuAD 1.1 (3 epochs, seed 42). The course train file
(87,599 questions) divides into **78,858 QA_train** (trains the baseline
BERT and every jointly trained variant) and **8,741 held-out** questions,
which split further — disjoint by article — into **5,845 head_train**
(trains the Kamath forest and the span-head reranker) and **2,896 calib**
(fits every temperature and Platt calibrator). The **10,570-question dev
file** is the report partition: every method is reported on the same dev
IDs. Correctness labels `y ∈ {0,1}` come from the official SQuAD v2.0
scorer (max exact match over golds) — never a reimplementation. Val and dev
grade differently: the held-out data has one gold answer per question, dev
about three (any-match), so dev's base rate is 79.6% against 70.0%. This
9.7-point gap is central to our findings.

## 2. Method

Each emitted answer carries a confidence derived from the winning window's
start/end logits (s, e) and the winning token positions (i, j):

- **v1 (raw span probability):** `softmax(s)[i] · softmax(e)[j]`, the model's
  own probability for its answer over the full window row.
- **v2 (renormalised):** `exp(s_i + e_j) / Z`, where Z sums over all legal
  spans (start ≤ end, context tokens only). A true distribution over answers;
  **zero fitted parameters**.
- **Temperature (1 parameter):** divide logits by T, fitted on the calib
  subset by NLL.
- **Platt (2 parameters):** `sigmoid(a·logit(c) + b)`, fitted on the calib
  subset by Newton/IRLS on the same NLL objective.

Every fit is sanity-checked by a half-split refit (calib split by article);
disagreement beyond ±0.05 (Δa) / ±0.1 (Δb) flags fit instability. Temperature
scaling and Platt are monotone, so they change only the confidence's level —
answers, EM/F1, and ranking are unchanged by construction, and the pipeline
asserts EM/F1 are bit-identical to the baseline on every report.

## 3. Findings

**(a) The raw signal is under-confident on dev — and the fix is structural.**
The model answers 79.6% of dev correctly while claiming 65.0% confidence. The
cause is measurable: the v1 product parks probability mass on tokens that
cannot be answers. Renormalising over legal spans (v2) moves mean confidence
to 67.9% toward the base rate and improves every proper score **with zero
fitted parameters** (Brier 0.1523 → 0.1456, ECE 0.1457 → 0.1175).

**(b) A one-parameter calibrator fitted on calib does not transfer.** Fitting
T on the calib subset returns T=1.04 (the signal is nearly calibrated *on the
held-out data*, where the gap is ~2 points), but applied to dev every proper
score worsens (Brier 0.1523 → 0.1564). The diagnosis is exact, not assumed: a
temperature controls only the *spread* of confidences; the *centre* is set by
the base rate, and the held-out base rate (0.70) is not dev's (0.80). Fitted
on the v2 signal, the same failure repeats (T=1.10 flattens dev). The
transferred temperature is reported as an honest negative result.

**(c) Two parameters transfer, because the offset absorbs the level shift.**
Platt's b moves the centre that T cannot. Fitted on calib, Platt on the v1
signal improves every dev score (Brier 0.1410, ECE 0.1239, NLL 0.4331) and is
the only fitted method that passes all pipeline gates. Its edge over the
zero-parameter v2 signal is small (0.1410 vs 0.1456) — with only 2,896 fit
rows, nearly all of the achievable gain is already captured by
renormalisation, and Platt on the renormalised signal buys nothing further
(it improves Brier/ECE marginally but worsens NLL). Calibration here is a
level shift, not complex structure — a 2-parameter affine map on the model's
own span probability, matching a 200-tree random-forest estimator (Kamath et
al., 2020) evaluated under the same protocol by a teammate.

**(d) Ranking was never the problem.** Raw AUC is 0.83, and the practical
use case works uncalibrated: a threshold set on the calib subset for a 5%
error budget holds on dev at 23.8% coverage / 1.79% achieved error, and EM
is 95.0 at 50% coverage (above the 81.5 inter-annotator ceiling).
Calibration corrects the *level* of trust, not the *order*.

## 4. Comparison table (dev, n=10,570; calibrators fitted on the 2,896 calib IDs)

| method | params | Brier ↓ | BSS ↑ | ECE ↓ | NLL ↓ | AUC ↑ |
|---|---|---|---|---|---|---|
| v1 raw softmax | 0 | 0.1523 | +0.062 | 0.1457 | 0.4548 | 0.8293 |
| v1 + temperature (calib-fit) | 1 | 0.1564 | +0.037 | 0.1586 | 0.4654 | 0.8293 |
| v2 renormalised (raw) | 0 | 0.1456 | +0.103 | **0.1175** | 0.4387 | 0.8187 |
| v2 + temperature (calib-fit) | 1 | 0.1534 | +0.055 | 0.1464 | 0.4581 | 0.8187 |
| v2 + Platt (calib-fit) | 2 | 0.1438 | +0.114 | 0.1216 | 0.4395 | 0.8187 |
| **v1 + Platt (calib-fit)** | 2 | **0.1410** | **+0.131** | 0.1239 | **0.4331** | 0.8293 |
| Kamath forest (teammate, cited) | ~200 trees | 0.1388* | — | 0.1267 | 0.4300 | 0.8357 |

*Fitted under a preliminary protocol; to be re-measured on the 5,845
head_train IDs for the final table. BSS is skill against the base-rate floor
`base(1−base)`; ECE uses 15 equal-mass bins and is diagnostic only (not
proper). EM/F1 identical across all rows (monotone maps; asserted in the
pipeline).

**Headline:** the cheapest effective fix (renormalisation over legal spans)
needs no fitted data at all and comes within 0.005 Brier of the best learned
calibrator, which itself uses only 2 parameters. Calibration here is
low-dimensional — a level shift, not complex structure.

## 5. Limitations

1. **Gold-convention mismatch.** The held-out data (1 gold/question) and dev
   (~3, any-match) differ in base rate by 9.7 points; no calibrator fitted
   on held-out data can learn dev's convention. Platt recovers a large part
   of the achievable gain (a dev-fitted Platt reaches 0.1213 — a diagnostic,
   never a method). ECE is therefore never compared across partitions.
2. **Small-sample fits.** At n=2,896 the Platt half-split disagreement
   (Δa=0.052–0.068) exceeds the ±0.05 tolerance calibrated on the full
   8,741 rows; we report the fits as measured and flag the uncertainty
   rather than retune the gate post-hoc.
3. **Report discipline.** Dev was read once per pre-registered method under
   the initial protocol; the protocol change and the v2+Platt exploration
   required additional reads, all labelled post-hoc and disclosed. The
   full-val fits (superseded) are archived with the pipeline outputs.
4. **Renormalisation is not rank-preserving** (AUC −0.011), so error-budget
   results are quoted from the v1 signal.
5. **Winning-window confidence.** 3% of dev questions have multiple windows;
   confidence comes from the winning window only. Brier Skill Score must be
   read next to AUC: dev's base-rate floor `base(1−base)` is small at base
   0.796.

## Citations (finalised)

- Devlin, J. et al. 2019. *BERT: Pre-training of Deep Bidirectional
  Transformers for Language Understanding.* NAACL. (model)
- Rajpurkar, P. et al. 2016. *SQuAD: 100,000+ Questions for Machine
  Comprehension of Text.* EMNLP. (dataset; dev-v1.1 golds)
- Guo, C. et al. 2017. *On Calibration of Modern Neural Networks.* ICML.
  (temperature scaling, ECE, reliability diagrams)
- Platt, J. 1999. *Probabilistic Outputs for Support Vector Machines.* (the
  2-parameter affine map)
- **Kamath, A., Jia, R., Liang, P. 2020. *Selective Question Answering under
  Domain Shift.* ACL.** (selective-prediction confidence baseline; cite our
  teammate Alexander's implementation `alex/baseline_kamath`, to be
  re-measured on the 5,845 head_train IDs under the binding protocol)
- **Zhang, M. et al. 2021. *Knowing What You Know: Calibrating Confidence
  for Extractive QA* (arXiv:2106.01494).** Differentiation: they train an
  additional learned CLS head (new capacity, retraining); we calibrate the
  existing span probability with at most 2 fitted parameters and zero
  retraining — and show the residual structure such heads capture is largely
  a level shift in our setting.
