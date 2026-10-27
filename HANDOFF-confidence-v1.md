# Handoff — Confidence output v1 (Raschka-style temperature scaling)

**Goal.** Make the SQuAD model emit a *calibrated probability that its answer is
correct*, alongside the answer — the way Jev's API returns a probability per
permitted answer. v1 is deliberately the simple, literature-standard version:

> fine-tune normally → take the model's own span probability as confidence →
> fit **one** temperature on held-out data → report it.

No new heads, no new losses, no renormalisation. Those are v2+ (see
§9 Out of scope) and the pipeline below is built so they drop in without rework.

Author state: this doc is written the day before implementation. Nothing in
`src/` has been changed yet.

---

## 1. What already exists (verified, do not re-derive)

### Data
- `data/raw/train-v1.1.json` (87,599 q / 442 articles), `data/raw/dev-v1.1.json`
  (10,570 q / 48 articles). Both gitignored, sourced from
  `rajpurkar/SQuAD-explorer` via `raw.githubusercontent.com` (see `SOURCES.md`
  convention). **Train/dev article titles are fully disjoint.**
- SQuAD 1.1 has **no unanswerable questions**. Every question has a gold span.
- `train-v1.1.json`: **exactly 1 gold answer per question**.
- `dev-v1.1.json`: **~3 gold answers per question** (`{3: 8490, 2: 136, 4: 759,
  1: 3, 5: 1147, 6: 35}`). 53.2% have >1 *distinct* answer text.
  → **never compare ECE across the two files**; the base rate of `y` differs.

### Split (`data.split_train`, seed 42, `validation_fraction: 0.1`)
| partition | questions | articles |
|---|---|---|
| train | 78,858 | 397 |
| validation | **8,741** | 45 |
| dev (`dev-v1.1.json`) | 10,570 | 48 |

`train.py` writes the validation partition to
`runs/<run>/validation-v1.1.json` (official SQuAD format). It is already
out-of-sample w.r.t. the QA model — that is exactly what is needed to fit a
temperature. **No new split work required.**

### Code
| file | relevant content |
|---|---|
| `src/qa_baseline/postprocess.py` | `best_answers(examples, features, start_logits, end_logits, n_best_size, max_answer_tokens) -> dict[qid, str]`. **Last line discards the score:** `return {qid: value[1] for qid, value in best.items()}`. |
| `src/qa_baseline/predict.py` | CLI `--config --data --model --output`; builds `prediction_features`; runs `Trainer.predict` → `start_logits, end_logits`; calls `best_answers`; writes `{qid: str}` JSON. |
| `src/qa_baseline/features.py` | `prediction_features` sets `offset_mapping[k] = None` for every non-context token (CLS, question, SEP, padding). **That `None` test is the context mask — reuse it.** |
| `src/qa_baseline/evaluate.py` | **Line 23 hard-rejects non-string prediction values.** Do not add fields to `predictions.json`. |
| `vendor/evaluate-v2.0.py` | `normalize_answer(s)`, `compute_exact(a_gold, a_pred)`, `compute_f1(a_gold, a_pred)`. **Note argument order: gold first.** Main takes `max()` over all golds. |
| `configs/baseline.json` | `model_name, seed 42, validation_fraction 0.1, max_length 384, doc_stride 128, max_answer_tokens 128, n_best_size 20, lr 2e-5, bs 8/16, epochs 3, weight_decay 0.01, warmup_ratio 0.1` |
| `tests/test_baseline.py` | 3 tests, incl. cross-window span offsets and "course scorer accepts v1 data". **Keep them green.** |

### Analysis already done
- `eda_squad.py`, `eda_squad2.py` — dataset shape/split/length stats.
- `SQUAD_SHAPE.md` — written summary of the above.
- Measured annotator-noise ceiling on dev: a human predicting with one
  annotator's span scores **81.5 EM / 91.0 F1** against the other annotators.
  Identical predictions score **69.4 EM** under single-gold convention →
  **gold convention alone moves EM by +12.2**. Only 46.8% of dev questions have
  all annotators agreeing.
  → use **81.5** as the reference line on reliability / risk-coverage plots.

---

## 2. Decisions already made (with one-line rationale)

1. **Confidence = the model's own span probability.** No new parameters.
   `c = softmax(s)[i*] · softmax(e)[j*]` — exactly slide 37 of the lecture.
2. **v1 does not renormalise** over legal spans. Deferred to v2; the design
   must make it a one-function swap.
3. **Temperature is fitted on the validation partition, reported on dev.**
   Val is already used for checkpoint selection (`load_best_model_at_end`) —
   disclose this (Raschka does the same); it is 1 parameter, so the optimism is
   negligible. Verify with the half-split check in §7.
4. **Do not use `--full-train` for the calibrated model.** It would train on the
   8,741 rows used to fit T, leaking in-sample correctness into the calibrator.
   Report the 78,858-trained model. State the ~0.5 EM cost as a deliberate
   trade.
5. **Do not touch the `{qid: str}` predictions contract.** Confidence goes in a
   separate file. `evaluate.py` will raise otherwise.
6. **Persist temperature as a model artifact** (temperature + threshold +
   package versions), mirroring `detector-config.json` in Raschka's notebook, so
   inference always applies it before the softmax.
7. **Metrics: Brier is the primary number** (see §5.1), then NLL, then ECE,
   plus a reliability diagram and a risk-coverage curve. Rationale: ECE is *not*
   a proper scoring rule — a constant base-rate predictor scores ECE ≈ 0 while
   being useless — so ECE alone cannot support a claim. Brier can. Raschka
   reports only Brier + log loss; ECE and the reliability diagram are our
   addition, so justify them.
8. **`y` must be computed with `vendor/evaluate-v2.0.py`'s own functions**,
   taking `max()` over golds. Never reimplement normalisation.

---

## 3. Deliverable shape (the Jev-API analogue)

```
# inference
answer, confidence = predict(text_or_example)     # confidence in [0,1], calibrated
```

```
runs/<run>/
  model/                     # existing: BERT checkpoint + tokenizer
  predictions.json           # existing: {qid: str}   (contract, unchanged)
  confidence.json            # NEW: {qid: {"c": float, ...}}  uncalibrated
  calibration.json           # NEW: {"temperature": float, "threshold": ...}
  scores.npz                 # NEW: cache, see §4
  val-predictions.json       # from predicting on validation-v1.1.json
  val-confidence.json
```

---

## 4. Step 1 — stop discarding the score (`postprocess.py`)

Do **not** change `best_answers`'s signature; the tests and `predict.py` use it.
Extract the search, then wrap.

```python
def best_spans(examples, features, start_logits, end_logits,
               n_best_size, max_answer_tokens):
    """Returns {qid: {"text", "feature", "start", "end", "score"}} for the
    single best legal span per question, chosen across all windows."""
    #  ... existing body of best_answers, but keep the full tuple ...


def best_answers(examples, features, start_logits, end_logits,
                 n_best_size, max_answer_tokens):
    spans = best_spans(examples, features, start_logits, end_logits,
                       n_best_size, max_answer_tokens)
    return {qid: s["text"] for qid, s in spans.items()}
```

Then, for each winning span, also record enough to recompute any confidence
signal later:

```python
# per qid, from the WINNING feature only:
s_row   # start_row of that feature, length max_length
e_row   # end_row of that feature
mask    # [off is not None for off in offsets]  ← context mask
best_i, best_j
```

Dump to `scores.npz`:

| array | dtype | shape |
|---|---|---|
| `qid` | `<U24` | (n,) |
| `s` | float32 | (n, max_length) |
| `e` | float32 | (n, max_length) |
| `mask` | bool | (n, max_length) |
| `best_i` | int32 | (n,) |
| `best_j` | int32 | (n,) |
| `n_windows` | int8 | (n,) |

Why cache: dev is ~11k windows; every later experiment (temperature variants,
margin, entropy, renormalisation) then runs in seconds without re-invoking BERT.

**Acceptance:** `predictions.json` is byte-identical to before the change, and
the 3 existing tests pass.

---

## 5. Step 2 — confidence functions (`src/qa_baseline/confidence.py`, new)

Pure functions, no torch. `numpy` only (already in `requirements.txt`).

```python
def softmax_rows(x, axis=-1):        # max-shift for stability (logits reach ±20)
def span_probability(s, e, mask, i, j, T=1.0):
    """v1 signal: softmax(s/T)[i] * softmax(e/T)[j]."""
def span_margin(s, e, mask, i, j, T=1.0):
    """top1 - top2 over legal spans in the 20x20 grid (cheap; for ablation)"""
def correctness(example, prediction):
    """y in {0,1} via vendor compute_exact; max over golds."""
def ece(c, y, n_bins=15):            # equal-mass bins — see below
def brier(c, y):
def nll(c, y):
def reliability_bins(c, y, n_bins=15):
def risk_coverage(c, y, coverages=(0.95,0.9,0.8,0.7,0.6,0.5)):
def fit_temperature(s, e, mask, best_i, best_j, y, grid=np.linspace(0.25, 5.0, 476)):
    """1-D search minimising NLL of the derived probability. Returns float."""
```

Notes:
- **Equal-mass bins** for ECE (`np.array_split` on the sorted confidences), not
  equal-width: confidences cluster near 0.8 so equal-width bins go sparse.
  Report `n_bins` alongside every ECE.
- `sklearn` has `brier_score_loss`, `log_loss` and `calibration_curve` but
  **no ECE** — write ECE by hand (~15 lines). Use sklearn for the others.
- `risk_coverage` returns EM on the retained subset at each coverage; plot it.

**Unit tests to add:** identical confidence on a hand-built 2-token example;
ECE = 0 on perfectly calibrated synthetic data; ECE = |p−base| for a constant
predictor.

### 5.1 Brier score

**Definition.** Mean squared error between the confidence you stated and what
actually happened:

$$ B = \frac{1}{N}\sum_{i=1}^{N}\left(c_i - y_i\right)^2 $$

Range `[0, 1]`. Lower is better. `0` = perfect. No clipping needed (unlike NLL),
because `c` is already in `[0,1]`.

**Why it is the primary number and ECE is secondary.** Brier is a *proper*
score — it is minimised when you report your true probability, so it cannot be
gamed. ECE can:

| predictor | ECE | Brier |
|---|---|---|
| always output the base rate (`0.74`) | **≈ 0** — looks perfect | `0.74 × 0.26 = 0.19` — correctly mediocre |
| honest and informative | small | small |

So report **both**: Brier is the claim, ECE is the diagnosis. If ECE is low but
Brier is not better than `base(1−base)`, your confidence carries no information
and you should say so.

**Interpretation aid.** `base(1−base)` is the Brier of the useless model that
just always says the base rate. On dev with `base ≈ 0.74` that is `0.19`. Your
Brier must beat that to be worth anything:

$$ \text{Brier Skill Score} = 1 - \frac{B}{base\,(1-base)} $$

`0` = no better than always guessing the base rate. `1` = perfect. One line,
unitless, immediately readable.

**What to expect on v1.** The unnormalised span product sums to well below 1 over
legal spans, so it is *systematically deflated*. Expect Brier to look poor here,
and expect v2's renormalisation to show up as a Brier drop. Write that down now —
it makes v2 a test rather than a fishing trip.

**Do not confuse with multiclass Brier.** `y` is binary per question ("was my
emitted answer correct") and `c` is the confidence in the answer actually given.
One target, one probability.

## 6. Step 3 — calibrate + report (`src/qa_baseline/calibrate.py`, new)

Two subcommands, so val and dev never get confused:

```bash
# fit T on the validation partition
PYTHONPATH=src python -m qa_baseline.calibrate fit \
  --scores runs/baseline/val-scores.npz \
  --data   runs/baseline/validation-v1.1.json \
  --output runs/baseline/calibration.json

# report on dev
PYTHONPATH=src python -m qa_baseline.calibrate report \
  --scores  runs/baseline/dev-scores.npz \
  --data    data/raw/dev-v1.1.json \
  --predictions runs/baseline/predictions.json \
  --calibration runs/baseline/calibration.json \
  --output  runs/baseline/calibration-report.json \
  --plot-dir runs/baseline/plots
```

`calibration.json`:
```json
{"method": "temperature scaling", "temperature": 1.62,
 "fitted_on": "validation-v1.1.json", "n": 8741,
 "threshold": 0.5, "package_versions": {...}}
```

`report` prints exactly this table (numbers below are illustrative placeholders):

```
              EM     F1   mean_c  base    Brier   BSS    ECE    NLL
uncalib.     ?      ?     ?       ?       ?       ?      ?      ?
+T (T=?)     ?      ?     ?       ?       ?       ?      ?      ?
```

`base` (the fraction of answers that are correct) and `Brier` are the two
numbers a reader needs to interpret everything else. `BSS` is the Brier Skill
Score from §5.1.

Plus, saved to `--plot-dir`:
1. **reliability diagram** (`sklearn.calibration.calibration_curve`), uncalibrated
   vs calibrated, with the diagonal;
2. **risk-coverage curve** (EM vs coverage) for `p_top`, with the **81.5 EM human
   ceiling** as a horizontal reference line, and the base rate as the floor;
3. **histogram of confidences**, uncalibrated vs calibrated — makes the `T`
   effect visible at a glance.

**Hard assertions to put in `report`.** Fail loudly on each:

```
1. EM and F1 IDENTICAL between the two rows          (T is a monotone map)
2. NLL DECREASED                                     (T was fitted to minimise NLL)
3. Brier and ECE both DECREASED                      (the point of the exercise)
```

Assertion 1 is the important one: it proves you changed only the number attached
to the answer, not the answer itself. If it fails, you have an offset, mask or
argmax bug.

---

## 7. Gotchas (in the order you will hit them)

1. **`evaluate.py:23` rejects non-string values.** Separate file, always.
2. **`compute_exact(a_gold, a_pred)` — gold first.** And `max()` over golds, or
   your `y` disagrees with `metrics.json` and you will chase ghosts.
3. **Logit overflow.** Subtract the row max before `exp` (logits reach ±20).
4. **`offset_mapping[k] is None` is the context mask.** Padding and question
   tokens are already `None`; don't hand-roll an attention mask.
5. **Multi-window questions (3% of dev, max 6 windows).** `best_spans` picks the
   winner across windows. Take confidence from **that window only** and document
   the choice; per-window probabilities are not strictly comparable.
6. **`n_best_size = 20` truncation.** Fine for the v1 product signal. It becomes
   wrong for v2's `Z` (which must sum over the full legal grid) — flag it in code.
7. **Train/val vs dev gold convention.** Val = 1 gold, dev = ~3. Never compare
   ECE numbers across the two files.
8. **`--full-train` is forbidden for the calibrated model** (decision 4).
9. **Half-split sanity check before trusting T** — split the val partition in
   half by article, fit T on each half; if `T_A ≈ T_B` (±0.05) there is nothing
   to overfit (1 parameter on 8.7k rows).
10. **Do not fit T on dev.** Report on dev, fit on val. One-shot on dev.

---

## 8. Definition of done for the next session

Minimum (this is v1 shipped):

- [ ] `postprocess.best_answers` behaviour unchanged; 3 existing tests green.
- [ ] `predict.py` writes `predictions.json` **unchanged in format** plus
      `confidence.json` and `scores.npz`.
- [ ] `confidence.py` with `span_probability`, `correctness`, `ece`, `brier`,
      `nll`, `risk_coverage`, `fit_temperature` + unit tests.
- [ ] `calibrate.py fit` produces `calibration.json` from the **validation**
      partition.
- [ ] `calibrate.py report` produces the §6 table + 3 plots on **dev**, and runs
      the 3 hard assertions.
- [ ] Half-split T check passes and is recorded in `results/`.
- [ ] `results/<date>-confidence-v1.md` records: config, seed, commit,
      `pip freeze`, T, the table, and the two plots.

Stretch (only if the above is green):

- [ ] `margin` and `entropy` as alternative signals (free — same `s`/`e` cache).
- [ ] separate `T_s`, `T_e` instead of one shared `T` (2 parameters).
- [ ] error-rate targeting: pick a threshold on val for a 5% target error, check
      the achieved rate on dev (uncalibrated vs calibrated). **This is the
      most persuasive single result** — it is the thing a ranking-only signal
      cannot do.

---

## 9. Explicitly out of scope for v1 (v2+)

| idea | why deferred |
|---|---|
| **Renormalise over legal spans** (`p = exp(s_i+e_j)/Z`) | one-line change once the cache exists; run as v2 and compare ECE. Motivated by: the naive product sums to <1 over legal spans (≈0.5 of the grid is illegal), so it is not a distribution over answers. |
| `[CLS]`-based learned head | needs training + `output_hidden_states=True` (not exposed by `AutoModelForQuestionAnswering`); belongs after v1 shows headroom. Precedent: Zhang et al. 2021, ACL Findings ([2106.01494](https://arxiv.org/abs/2106.01494)) — **this is close enough to the proposal's row 4/5 that it must be cited and differentiated.** |
| span-conditioned head | motivated by the start/end independence assumption baked into `L = -log Ps - log Pe`. Precedent: Read+Verify (AAAI'19), Retrospective Reader (AAAI'21). |
| group-wise temperature | 1 param × G groups; the rung between global T and a learned head. |
| Platt instead of pure temperature for a learned head | pure `T` cannot absorb a base-rate shift; the dev/val gold-convention gap is +12.2 EM. |

---

## 10. One-paragraph framing for the report

> The baseline emits `P_s(a_s)·P_e(a_e)`, which is a scoring function for
> argmax, not a calibrated probability: it is not normalised over legal spans,
> and it assumes the two boundaries are independent. v1 therefore takes the
> model's own probability as the confidence signal and applies a single
> temperature fitted on held-out data — the standard recipe (Guo et al. 2017;
> implemented in this form by Raschka, 2026). Because temperature scaling is
> monotone, EM and F1 are unchanged by construction; we therefore evaluate the
> confidence itself with the **Brier score** (a proper scoring rule, so it cannot
> be gamed) alongside ECE (which diagnoses *where* the probabilities are wrong).
> Both are reported against the trivial baseline `base(1−base)` via the Brier
> Skill Score. The measured ceiling on dev is 81.5 EM (inter-annotator
> agreement), which bounds how much any confidence signal can realistically
> discriminate.
