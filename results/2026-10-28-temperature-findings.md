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

> **Update 2026-11-07.** Platt is now a pipeline method
> (`calibrate fit --method platt`). The fitted values are a=0.7672, b=+0.2489;
> the half-split check passes (Δa=0.036, Δb=0.016); and the report reproduces
> the table above exactly (dev Brier 0.1391, ECE 0.1171, NLL 0.4280). Unlike
> the temperature, Platt **passes all three report gates** on dev. The
> "Platt fitted on dev (diagnostic)" row stays a diagnostic — never a method.

## Finding 6 — Renormalising over legal spans fixes most of the level error for free (2026-11-07)

The Oct-28 prediction: the v1 confidence parks probability mass on tokens that
cannot be answers, so putting that mass back (renormalising over legal spans)
should improve the scores. **Confirmed on dev:**

| dev | mean_c | Brier | BSS | ECE | NLL |
|---|---|---|---|---|---|
| v1 raw | 0.6503 | 0.1523 | +0.0617 | 0.1457 | 0.4548 |
| v2 raw (renormalised, no fitted parameters) | **0.6789** | **0.1456** | **+0.1032** | **0.1175** | **0.4387** |
| v2 + temperature (T=1.10, fitted on val) | 0.6496 | 0.1534 | +0.0553 | 0.1464 | 0.4581 |

One number changes sign in the story: the temperature fitted on the v2 signal
again does not transfer (T=1.10 flattens dev, same base-rate reason as Finding
4). The fix that works needs no fitted parameters at all. Platt on the v1
signal remains the best overall (Brier 0.1391); renormalisation is the best
parameter-free signal (0.1456). One cost: renormalisation is not
rank-preserving — AUC drops 0.829 → 0.819 — so the error-budget table
(Finding 6 in the older numbering, "the most persuasive result") should still
be quoted from the v1 signal.

Reproduce: `calibrate fit --method platt`, `calibrate fit --signal v2`,
`calibrate report --signal v2` (commands in the technical report §10).

**Update 2026-11-08 (binding protocol).** The TA's split spec arrived: ALL
temperature/Platt fits must use only 2,896 calib IDs (not the full 8,741 val
partition we had used). We refit everything on the calib IDs — see §11 of
the technical report. What changes: Platt's edge shrinks (v1+Platt dev
Brier 0.1391 → 0.1410) and the parameter-free renormalised signal now sits
only 0.0046 behind the best 2-parameter calibrator; v2+Platt becomes a wash
on its own signal (improves Brier/ECE, worsens NLL); the temperature still
fails to transfer, on both signals, now by a wider margin. The story is
*stronger* under the honest protocol: the cheapest fix (renormalisation,
zero fitted parameters) nearly matches the best learned calibrator.

**Update 2026-11-08.** The last empty cell in the grid — Platt fitted on the
renormalised signal — was run beforehand, labelled post-hoc (dev's one-shot
read was already spent). Under the full-val protocol it helped but did not
win: dev Brier 0.1420, better than v2-raw (0.1456) on every proper score and
the best ECE in the table (0.1152), but the pre-declared target (beat
Platt-on-v1's 0.1391) was NOT met. Under the binding calib-only refit it is
a wash (§11): with 2,896 fit rows, Platt on the renormalised signal buys
nothing the renormalisation itself didn't already buy. The plain-language
conclusion stands: renormalisation is the best fix that needs no fitted
parameters; Platt on the original signal is the best fix that uses two.