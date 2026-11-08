# Report section (draft) — Confidence calibration for extractive QA

> Draft for Kea's ~1.5-page report section. Every number traces to a pipeline
> command in `results/2026-10-27-confidence-v1.md` (§2–§10). Port to the
> team's Google Doc / LaTeX. Figures: `results/figures/risk_coverage.png`,
> `results/figures/reliability.png` (both committed).

## 1. Setup

We evaluate confidence calibration on a BERT-base-uncased extractive QA
model fine-tuned on SQuAD 1.1 (3 epochs, seed 42; 78,858 training questions).
The 8,741-row validation partition (45 articles) is the **calibration**
partition: all calibrators are fitted on it, and it is held out of training.
The 10,570-row dev partition is the **report** partition, read once per
method (the dev read for Platt-on-v2 is labelled post-hoc). Correctness
labels `y ∈ {0,1}` come from the official SQuAD v2.0 scorer (max exact match
over golds) — never a reimplementation. Val and dev grade differently: val
has one gold answer per question, dev about three (any-match), so dev's base
rate is 79.6% against val's 70.0%. This 9.7-point gap is central to our
findings.

## 2. Method

Each emitted answer carries a confidence derived from the winning window's
start/end logits (s, e) and the winning token positions (i, j):

- **v1 (raw span probability):** `softmax(s)[i] · softmax(e)[j]`, the model's
  own probability for its answer over the full window row.
- **v2 (renormalised):** `exp(s_i + e_j) / Z`, where Z sums over all legal
  spans (start ≤ end, context tokens only). A true distribution over answers;
  **zero fitted parameters**.
- **Temperature (1 parameter):** divide logits by T, fitted on val by NLL.
- **Platt (2 parameters):** `sigmoid(a·logit(c) + b)`, fitted on val by
  Newton/IRLS on the same NLL objective.

Every fit is sanity-checked by a half-split refit (val split by article);
disagreement beyond ±0.05 (Δa) / ±0.1 (Δb) would indicate overfitting.
Temperature scaling and Platt are monotone, so they change only the
confidence's level — answers, EM/F1, and ranking are unchanged by
construction, and the pipeline asserts EM/F1 are bit-identical to the
baseline on every report.

## 3. Findings

**(a) The raw signal is under-confident on dev — and the fix is structural.**
The model answers 79.6% of dev correctly while claiming 65.0% confidence. The
cause is measurable: the v1 product parks probability mass on tokens that
cannot be answers. Renormalising over legal spans (v2) moves mean confidence
to 67.9% toward the base rate and improves every proper score **with zero
fitted parameters** (Brier 0.1523 → 0.1456, ECE 0.1457 → 0.1175).

**(b) A one-parameter calibrator fitted on val does not transfer.** Fitting T
on val returns T=1.02 (the signal is nearly calibrated *on val*, where the
gap is ~2 points), but applied to dev every proper score worsens (Brier
0.1523 → 0.1543). The diagnosis is exact, not assumed: a temperature controls
only the *spread* of confidences; the *centre* is set by the base rate, and
val's base rate (0.70) is not dev's (0.80). Fitted on the v2 signal, the same
failure repeats (T=1.10 flattens dev). The transferred temperature is
reported as an honest negative result.

**(c) Two parameters transfer, because the offset absorbs the level shift.**
Platt's b moves the centre that T cannot. Fitted on val, Platt on the v1
signal improves every dev score (Brier 0.1391, ECE 0.1171, NLL 0.4280) —
matching a 200-tree random-forest confidence estimator (Kamath et al., 2020)
evaluated under the identical protocol (cited from our teammate's
implementation). The post-hoc v2+Platt cell confirms partial composition:
renormalisation plus Platt beats v2-raw on every proper score (Brier 0.1420)
and gives the best ECE in the table (0.1152), but does **not** beat
Platt-on-v1 — the v1 signal's sharper ranking (AUC 0.829 vs 0.819) is worth
more than the renormalised level. Pre-declared criterion (Brier < 0.1391),
fixed before the run: not met, recorded as a finding.

**(d) Ranking was never the problem.** Raw AUC is 0.83, and the practical
use case works uncalibrated: a threshold set on val for a 5% error budget
holds on dev at 25.4% coverage / 1.87% achieved error, and EM is 95.0 at 50%
coverage (above the 81.5 inter-annotator ceiling). Calibration corrects the
*level* of trust, not the *order*.

## 4. Comparison table (dev, n=10,570; all calibrators fitted on full val, n=8,741)

| method | params | Brier ↓ | BSS ↑ | ECE ↓ | NLL ↓ | AUC ↑ |
|---|---|---|---|---|---|---|
| v1 raw softmax | 0 | 0.1523 | +0.062 | 0.1457 | 0.4548 | 0.8293 |
| v1 + temperature (val-fit) | 1 | 0.1543 | +0.050 | 0.1522 | 0.4599 | 0.8293 |
| v2 renormalised (raw) | 0 | 0.1456 | +0.103 | 0.1175 | 0.4387 | 0.8187 |
| v2 + temperature (val-fit) | 1 | 0.1534 | +0.055 | 0.1464 | 0.4581 | 0.8187 |
| v2 + Platt (val-fit) — post-hoc | 2 | 0.1420 | +0.126 | **0.1152** | 0.4348 | 0.8187 |
| **v1 + Platt (val-fit)** | 2 | **0.1391** | **+0.143** | 0.1171 | **0.4280** | 0.8293 |
| Kamath forest (Alexander, cited) | ~200 trees | 0.1388 | — | 0.1267 | 0.4300 | 0.8357 |

BSS is skill against the base-rate floor `base(1−base)`; ECE uses 15
equal-mass bins and is diagnostic only (not proper). EM/F1 identical across
all rows (monotone maps; asserted in the pipeline).

**Headline:** a two-parameter affine map on the model's own span probability
matches a 200-tree forest, and the cheapest effective fix (renormalisation)
needs no fitted data at all. Calibration here is low-dimensional — a level
shift, not complex structure.

## 5. Limitations

1. **Gold-convention mismatch.** Val (1 gold/question) and dev (~3,
   any-match) differ in base rate by 9.7 points; no calibrator fitted on val
   can learn dev's convention. Platt recovers about half the achievable gain
   (a dev-fitted Platt reaches 0.1213 — a diagnostic, never a method). ECE is
   therefore never compared across partitions.
2. **Dev one-shot discipline.** Dev was read once per pre-registered method;
   the v2+Platt read is a labelled post-hoc single confirmation, disclosed.
3. **Renormalisation is not rank-preserving** (AUC −0.011), so error-budget
   results are quoted from the v1 signal.
4. **Winning-window confidence.** 3% of dev questions have multiple windows;
   confidence comes from the winning window only.
5. Brier Skill Score must be read next to AUC: dev's base-rate floor
   `base(1−base)` is small at base 0.796.

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
  teammate Alexander's implementation `alex/baseline_kamath`, evaluated under
  our identical val-fit/dev-report protocol)
- **Zhang, M. et al. 2021. *Knowing What You Know: Calibrating Confidence
  for Extractive QA* (arXiv:2106.01494).** Differentiation: they train an
  additional learned CLS head (new capacity, retraining); we calibrate the
  existing span probability with at most 2 fitted parameters and zero
  retraining — and show a 200-tree learned forest (Kamath) is matched by 2
  parameters, i.e. the residual structure their head captures is largely a
  level shift in our setting.
