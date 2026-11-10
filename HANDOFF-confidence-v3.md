# Handoff — Confidence v3: results consolidation & paper sprint

**Status.** All experiments for Kea's axis are DONE and pushed. The v2
prediction is confirmed, Platt is a pipeline method, and the comparison table
is complete. What remains is **integration and writing**, against a hard
deadline: **report + slides + video due 12 Nov** (grading 60% content / 30%
presentation / 10% novelty; 4-page ACL short paper).

**Read in order:** this file → `results/2026-10-28-temperature-findings.md`
(plain-language) → `results/2026-10-27-confidence-v1.md` §11 (BINDING
protocol table) and §10 (superseded full-val table, kept as provenance).

**Addendum 2026-11-08 — PROTOCOL CHANGE (binding).** The TA's split spec
arrived after the v2+Platt run: ALL temperature/Platt fits use ONLY the
2,896 calib IDs (exactly the `calib` part of Ziying's
`splits/heldout-v1.json`, commit `f24d049`); the forest/reranker train on
the 5,845 head_train IDs; final comparison on the 10,570 dev IDs. Everything
was refit under `--part calib` (§11 of the technical report): Platt-on-v1
still best fitted row (dev Brier 0.1410, all gates pass); v2-raw now only
0.0046 behind with ZERO parameters; T still fails on both signals; both
Platt half-split gates fail at n=2,896 (recorded, not retuned). The §1
table's fitted rows are SUPERSEDED — quote §11 numbers. Alexander's Kamath
row is now binding-compliant too: he retrained on head_train @ `68fab11`
(Brier 0.1405, ECE 0.1292 10-bin, NLL 0.4371, AUC 0.8344) and added the
team-canonical split `splits/confidence-comparison-v1.json` (verified: its
calib IDs are identical to ours, so our refits stand). Still a dead heat
with our calib-fit Platt (0.1410). Ziying recorded a compliant post-hoc
confidence head (train 5,845 / temp 2,896): an honest negative — loses to
the raw signal on every metric (Brier 0.1635, AUROC 0.776 vs 0.835).

---

## 1. State of the results (all pipeline-reproduced, commit `86adbdf`)

The paper's central confidence table, dev set (10,570), all fitted on the full
val partition (8,741, seed 42, protocol identical to Alexander's Kamath run):

| method | params | dev Brier | BSS | dev ECE | dev NLL | AUC |
|---|---|---|---|---|---|---|
| raw softmax (v1) | 0 | 0.1523 | +0.062 | 0.1457 | 0.4548 | 0.8293 |
| + temperature (val-fit) | 1 | 0.1543 | +0.050 | 0.1522 | 0.4599 | 0.8293 |
| **v2 renormalised (raw)** | 0 | **0.1456** | **+0.103** | **0.1175** | **0.4387** | 0.8187 |
| v2 + temperature (val-fit) | 1 | 0.1534 | +0.055 | 0.1464 | 0.4581 | 0.8187 |
| **v2 + Platt (val-fit)** — post-hoc, single confirmation | 2 | 0.1420 | +0.126 | 0.1152 | 0.4348 | 0.8187 |
| **Platt on v1 (val-fit)** | 2 | **0.1391** | **+0.143** | 0.1171 | 0.4280 | 0.8293 |
| Kamath forest (Alexander, cited) | ~200 trees | 0.1388 | — | 0.1267 | 0.4300 | 0.8357 |

Headline sentences, ready to use:

1. **Renormalisation fixes most of the level error for free** (zero fitted
   parameters): Brier 0.1523 → 0.1456, mean_c 0.650 → 0.679 toward base 0.796.
   The Oct-28 prediction was pre-registered and CONFIRMED.
2. **Platt (2 params) matches a 200-tree forest** (0.1391 vs 0.1388) under the
   identical protocol. Calibration here is low-dimensional — a level shift,
   not complex structure.
3. **One-parameter temperature never transfers** (val base 0.6995 vs dev base
   0.7960, +9.66 points): diagnosed, measured, and the same failure repeats on
   the v2 signal (T=1.10 flattens dev).
4. **Ranking was never the problem**: AUC 0.83 raw; error budget (5% target,
   threshold from val) holds on dev at 25.4% coverage / 1.87% error; 95 EM at
   50% coverage (above the 81.5 inter-annotator ceiling).

## 2. What happened in the v2 session (2026-11-07, commit `86adbdf`)

- Platt: `calibrate fit --method platt` → a=0.7672, b=+0.2489, half-split
  Δa=0.036 / Δb=0.016 (pass). Dev row passes all three report gates. Labelled
  **post-hoc, single confirmation** (dev one-shot spent; fresh split declined).
- v2: `renormalised_probabilities` (chunked vectorised) + `fit_temperature_v2`.
  Fit on val: **T=1.10, half-split Δ=0**. Dev: v2-raw improves every proper
  score; v2+T is hollow (passes the pre-declared criterion by 0.0009, worse
  than v2-raw — recorded honestly).
- **Bug of the session**: 3 val rows have empty context masks (no legal spans)
  → NaN probabilities silently poisoned the grid search (fit returned the grid
  edge T=0.25 with a phantom half-split disagreement). `calibrate fit` now
  excludes and prints them; `fit_temperature_v2` raises on NaN. Dev has zero.
- Tests: **38 passed** (Platt recovery/dispatch, v2 chunked-vs-slow equivalence,
  distribution property, signal orientation contract incl. v2).
- **Addendum 2026-11-08 (v3 session):** ran the missing grid cell — Platt
  fitted on the v2 signal (`calibrate fit --method platt --signal v2` →
  a=0.7601, b=+0.1126, n=8,738, 3 NaN rows excluded; half-split Δa=0.0492 —
  just inside ±0.05 — Δb=0.0394, pass). Dev report (post-hoc, single
  confirmation): Brier 0.1420, BSS +0.1256, ECE 0.1152, NLL 0.4348; all three
  report gates pass; risk–coverage identical to v2-raw (Platt is monotone).
  Pre-declared criterion (Brier < 0.1391) **NOT met** — Platt-on-v1 remains
  the headline. Outputs: `runs/baseline/calibration-platt-v2.json`,
  `calibration-report-platt-v2.json`, `plots-platt-v2/`. **Experiments now
  closed for good; writing sprint starts.**

## 3. Team state (from the Nov 7 Telegram export + branches)

| person | branch | owns | status |
|---|---|---|---|
| **Kea** | `experiment/temperature-confidence` | calibration axis: raw/+T/Platt/v2, error budget, risk–coverage, all results docs | **experiments complete** |
| Alexander | `alex/baseline_kamath` (+ `baseline-results`) | Kamath forest confidence; TA suggested it as the confidence baseline | done, numbers cited in our table |
| Ziying | `span-head` → `exp/span-conditioned-confidence` | learned span-head **reranker** (accuracy: dev EM 79.66 → 81.14); offered to add a confidence output | done for reranking; confidence output optional |

All four branch lines forked from `c761a35` (initial commit); **none is
merged**; `main` is still the untouched scaffold. Kea's split-protocol answer
(full-val fit / dev report) was given in chat but not yet acknowledged by the
teammates.

## 4. What remains before 12 Nov

1. **Send the protocol message** (Kea, today): propose the §1 table as the
   report's confidence comparison; ask Ziying whether her span-head confidence
   joins it (her thirds are fine internally, disclosed); ask Alexander to
   confirm the Kamath numbers as the cited row.
2. **Write Kea's report section** (~1.5 pages): 5-part outline already approved
   in principle — Setup (splits/protocol) → Method (span probability, T,
   Platt, v2) → Findings (deflation, transfer failure, rank transfers) →
   Comparison table (§1) → Limitations. Draft in repo markdown
   (`results/paper/` suggested), port to the team's Google Doc / LaTeX.
   Figures when needed: risk–coverage + reliability
   (`results/figures/*.png` already committed).
3. **Integrate with Ziying's accuracy axis**: the combined story is "the
   reranker improves EM/F1 (+1.49 EM); the calibration layer attaches a
   confidence with a guaranteed error budget". Decide whose section goes first.
4. **Citations** (mandatory): Guo et al. 2017, Platt 1999, **Kamath et al.
   2020** (now in-repo — cite Alexander's implementation), **Zhang et al.
   2021 (2106.01494)** — differentiate (they learn a CLS head; we calibrate the
   span probability with ≤2 parameters), SQuAD, BERT, Devlin.
5. **Merge/reconcile branches** — LAST, after the report draft exists. Alex's
   branch independently fixed the `max_answer_tokens` NameError and modified
   `postprocess.py`/`predict.py`; expect conflicts with our `0cdc7d3`/sidecar
   fix. Suggested order: merge `alex/baseline_kamath` into a consolidation
   branch first (smallest), then Ziying's (3,600 lines, 14 test files), then
   ours, resolving `postprocess.py`/`predict.py` by hand. Do NOT merge through
   `main` blindly.
6. **Video** (15 min, everyone speaks): Kea's segment ≈ 3 min on the
   confidence axis — use the plain-language findings doc as the script
   skeleton; the bug-and-gate story (NameError, inverted margin, assertions)
   earns presentation marks.

## 5. Gotchas (inherited + new)

1. **Dev is burned.** Nothing more gets fitted on it; further dev reads must be
   labelled post-hoc or run on a fresh split (declined for this pass).
2. **`results/*` and `runs/*` are gitignored** — `git add -f results/...` for
   any new doc/figure.
3. **The synthetic fixture lies** (twice now): any new signal must pass the
   orientation contract on real-shaped data (`tests/test_signals.py`).
4. **`_signal` contract: higher = more confident**; NaN margins → −inf; v2
   signal added under the same contract.
5. **Rows with no legal spans produce NaN v2 probabilities** — 3 in val (excluded
   by `calibrate fit`, printed), 0 in dev. `fit_temperature_v2` raises instead
   of corrupting.
6. **ECE is not comparable across val/dev** (gold conventions differ); BSS must
   be read next to AUC (dev's floor `base(1−base)` is small at base 0.796).
7. **Local venv has no torch** — `predict.main()` can't run locally; the
   symtable test is the guard. Kamath reruns need `scikit-learn==1.6.1` (not
   installed; cite Alexander's summary instead).
8. **Hard stop passed for experiments** (10 Nov EOD was the self-imposed limit;
   it's already met — no new experiments before the deadline unless a teammate
   asks for one).

## 6. Definition of done for the v3 session

- [ ] Protocol message sent to Ziying/Alexander; their answers recorded here.
- [ ] Kea's report section drafted (repo markdown), table §1 included.
- [ ] Citations list finalised; Zhang/Kamath differentiation written.
- [ ] Video segment scripted (~3 min).
- [ ] Branch merge plan executed or explicitly scheduled with owners.
- [ ] Final numbers in the report all trace to pipeline commands (no
      chat-only numbers).
