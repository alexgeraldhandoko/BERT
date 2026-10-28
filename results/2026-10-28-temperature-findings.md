# Temperature scaling: plain-language findings

**Date.** 2026-10-28. Full technical record: `2026-10-27-confidence-v1.md`.
**One-line summary.** Our confidence numbers carry real information, but their
*level* is wrong on dev, and a one-number correction cannot fix a wrong level.
The *order* of the numbers is reliable, and the order is enough for the main
practical use.

## Words this document uses

| word | meaning here |
|---|---|
| calibrated | A confidence number is calibrated when it equals the true share of correct answers among answers with that number. |
| base rate | The share of correct answers in a dataset. |
| temperature (T) | One number. We divide the model scores by T before the softmax. T below 1 sharpens the confidence. T above 1 flattens it. |
| Brier score | Mean squared error between the stated confidence and the true 1/0 outcome. Lower is better. |
| ECE | Average gap between claimed confidence and true accuracy, grouped by confidence level. Lower is better. |
| AUC | The probability that a correct answer gets a higher confidence than an incorrect answer. 0.5 means no information. |
| Platt scaling | A two-number correction: `sigmoid(a * logit(c) + b)`. |

## Finding 1 — The model is under-confident on dev

The model answers 79.6% of dev questions correctly. On average it claims only
65.0% confidence. A gap of ~15 points means the model hedges when it should not.
This matches the handoff prediction: the raw span product is not a distribution,
so it sits too low.

## Finding 2 — The same signal is nearly calibrated on validation

On the validation partition the model is right 69.9% of the time and claims 67.5%
confidence. The gap is ~2 points. When we fit a temperature there, the fit
returns T = 1.02 — almost no change. Two independent halves of validation return
T = 1.03 and T = 1.01, so this is not noise.

This surprised us. We expected T below 1, because dev shows a clear under-confidence.

## Finding 3 — The temperature did not transfer to dev

We applied T = 1.02 to dev. Every proper score got worse:

| | Brier | ECE | NLL |
|---|---|---|---|
| no calibration | 0.1523 | 0.1457 | 0.4548 |
| with T = 1.02 | 0.1543 | 0.1522 | 0.4599 |

The answers themselves did not change. A temperature never changes the argmax;
it only rescales the number next to the answer. So the code was correct, and the
result was real.

## Finding 4 — Why: the two datasets grade differently

Validation has one gold answer per question. Dev has about three, and we accept
any of them. More accepted answers push the base rate up:

| partition | base rate |
|---|---|
| validation | 0.6995 |
| dev | 0.7960 |

A temperature controls only the spread of the confidence numbers. It cannot move
their centre. The centre is set by the base rate. We fitted T on a dataset with
base rate 0.70, so the fit centres the numbers near 0.70. Dev needs a centre near
0.80. The correction pushed dev the wrong way.

Proof: when we fit T on dev itself (diagnostic only — this is not a permitted
method), the fit returns T = 0.71, and every score improves: Brier 0.1342,
ECE 0.0718. The method works on the distribution it was fitted on. The transfer
is what fails.

## Finding 5 — Two numbers transfer half-way; one number transfers none

We fitted Platt scaling (two numbers: slope a, offset b) on validation and
applied it to dev:

| calibrator | fitted on | dev Brier |
|---|---|---|
| nothing | — | 0.1523 |
| temperature (1 number) | validation | 0.1543 |
| Platt (2 numbers) | validation | 0.1391 |
| Platt | dev (diagnostic) | 0.1213 |

The offset b absorbs part of the base-rate shift, so Platt recovers about half
the possible gain. The remaining half needs labels graded under dev's own
convention. No calibrator fitted on validation can learn that.

## Finding 6 — The order of the numbers transfers, and the order is enough

A temperature is a monotone function: it never reorders the questions. The
calibration work above therefore never touched the order. We then measured the
order on its own, with AUC:

| partition | AUC |
|---|---|
| validation | 0.800 |
| dev | 0.829 |

The ranking survived the dataset change, while the level did not (ECE went from
0.045 to 0.146). Two results use only the order:

**Selective prediction.** Sort dev by confidence and answer only the top part:

| share answered | accuracy on that share |
|---|---|
| 100% | 79.6% |
| 70% | 91.0% |
| 50% | 95.0% |

The human inter-annotator ceiling on dev is 81.5% EM. The model beats it once it
abstains on the 40% it trusts least.

**Error budget.** We chose a threshold on validation so that the retained set
has at most 5% error, then applied the same threshold to dev:

| signal | share answered | error on that share |
|---|---|---|
| span probability | 25.4% | 1.87% |
| margin | 21.2% | 1.96% |
| entropy | 26.0% | 1.71% |

All three signals beat the 5% budget. No calibration step improves these numbers,
because they depend only on the order. This is the capability a score without a
calibrated probability cannot offer, and it survived the distribution change.

## What v2 must do

1. Renormalise the span probability over legal spans. This attacks the level
   error directly. The function exists and is tested; nobody wired it into the
   report yet.
2. Fit the calibrator on data graded under the target convention, or use Platt
   and accept the measured half-transfer.
3. Keep the error-budget result as the headline claim. It is rank-based, so it
   does not depend on the base rate of the report set.

## Caveats

- We used validation twice: once for checkpoint selection during training, once
  to fit T and the thresholds. The half-split check bounds the harm.
- We reported on dev once only. We did not tune anything on dev.
- The scores cache stores the winning window only. About 3% of questions have
  several windows; their confidence comes from the winning window alone.
- ECE depends on the bin rule. We used 15 equal-mass bins. Do not compare ECE
  between val and dev.
