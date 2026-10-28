# Handoff — v2 experiments (renormalisation, Platt, confirmation split)

**Status.** v1 is COMPLETE: run, fitted, reported, written up, pushed. The
session-2 work is done; what remains is the v2 experiment set and then paper
assembly. Estimated hands-on time for v2: **one afternoon, no GPU** — everything
runs locally on the cached `scores.npz`.

Read in this order: this file → `results/2026-10-28-temperature-findings.md`
(plain-language findings) → `results/2026-10-27-confidence-v1.md` (technical
record).

---

## 1. Where things stand after session 2 (2026-10-28)

### What was accomplished
| item | outcome |
|---|---|
| predict pipeline | both jobs crashed on a `NameError` in session 1; fixed (`0cdc7d3`), resubmitted as `1030408`/`1030409`, both `COMPLETED`. Model was never touched — no retrain. |
| calibration | T = 1.02 fitted on val (half-split Δ=0.02). Transfers **negatively** to dev: NLL 0.4548→0.4599, Brier 0.1523→0.1543. Diagnosed: val/dev base rates differ by +9.66 pts (1 gold vs ~3 golds per question); a monotone map cannot move an operating point. |
| oracle headroom | T* on dev = 0.71 → Brier 0.1342. Platt (2-param) fitted on **val** → 0.1391 (half the gap to oracle 0.1213). Measured, but see §3.3 — not yet in the pipeline. |
| rank-based results | AUC dev 0.829 (val 0.800). Risk–coverage: 95 EM at 50% coverage. Error budget (5%, threshold from val): prob 25.4% cov @ 1.87% err, margin 21.2% @ 1.96%, entropy 26.0% @ 1.71%. |
| bugs found & fixed | (1) unbound `max_answer_tokens` killed both predict jobs; (2) val/dev sidecar name collision (dev would have clobbered val cache); (3) **inverted `span_margin`** in `_signal` — margin reported UNREACHABLE while its AUC was 0.757; NaN margins ranked most confident. All three have regression tests; suite is 29 passing. |
| docs | `results/2026-10-27-confidence-v1.md` (technical record, incl. §3.3/§3.4 headroom) and `results/2026-10-28-temperature-findings.md` (plain-language, six findings). Plots in `results/figures/`. All force-added (results/* is gitignored). |

### Branch / commits
`experiment/temperature-confidence`, tip `ae946ce`:
`0cdc7d3` → `423a1a5` → `4a555b6` → `a908d51` → `53fb8d2` → `ae946ce`.
Cluster is on `98292c3` — **no cluster work is needed for v2**; everything runs
on the local `runs/baseline/*.npz` (already fetched, ~50 MB).

### Grading context (from `paper_sources/cs4248_project_requirements-1.pdf`)
60% content / 30% presentation / 10% novelty; 4-page ACL short paper + video +
code. Findings map: strong on soundness/quantity; the performance claim must be
framed as **selective prediction with a measured error budget** (EM/F1 are
unchanged — do not claim otherwise); novelty is the weak axis, which v2 exists
to address. Must cite Guo 2017, Platt 1999, and **Zhang et al. 2021
(2106.01494)** and differentiate (they learn a CLS head; we calibrate the span
probability with no new parameters).

---

## 2. Task A — wire v2 renormalisation into fit/report (do first)

`renormalised_span_probability` exists in `confidence.py` and is tested, but no
report reads it. Predicted result (pre-registered in the old handoff §5.1):
**Brier drops vs v1** because v1 is systematically deflated (dev mean_c 0.6503
vs base 0.7960).

Concrete steps:
1. Extend `calibrate report` (or add `--signal`) to compute the v2 signal from
   the scores cache: `exp(s_i + e_j) / Z` over the **legal grid** (mask-true,
   end ≥ start). The cache stores full window rows, so Z needs no approximation —
   do NOT truncate to the 20×20 n_best grid (that limit was fine for the argmax
   search, wrong for a normaliser; see old handoff gotcha 6).
2. Report v2 as its own row in the §3 table: mean_c, Brier, BSS, ECE, NLL on dev.
3. Gate: compare against the recorded v1 row. A Brier **increase** is a finding
   too — write it down either way; do not tune on dev to force the drop.
4. If v2 moves the *level* (mean_c toward base) but not all metrics, that is the
   expected signature: renormalisation fixes level, not ranking (AUC is identical
   up to ties).
5. Update both results docs with the v2 row; label the prediction-confirmation
   status explicitly in the text.

Acceptance: `calibrate report` prints v1 and v2 rows; tests cover Z computation
on a hand-built row; suite green.

## 3. Task B — Platt, properly (turn the diagnostic into a result)

Current state: the Platt numbers (val-fit → dev 0.1391 vs uncalibrated 0.1523,
oracle 0.1213) were computed in an ad-hoc script in session 2's chat. They are
NOT reproducible from the repo. Steps:
1. Add `--method platt` to `calibrate fit`. Newton/IRLS on `logit(c)` from the
   val cache; write `{"method": "platt", "a": ..., "b": ...}` to
   `calibration.json`. `report`/`target` must dispatch on `method`.
2. Half-split check by article, same as T got: fit `a,b` per half, require
   agreement within tolerance (pick ±0.05 on `a`, ±0.1 on `b`; two params on
   8,741 rows should pass easily — record, don't assume).
3. Unit tests: recovers a known a,b on synthetic data; `report` reads platt
   calibration; NaN/edge handling consistent with the orientation contract.
4. Re-emit the §3.4 table from the pipeline and relabel the Platt row in both
   results docs from "diagnostic" to "measured".

### 3.1 Claim discipline (important)
The original protocol ("fit one T on val, report on dev") was pre-registered;
Platt was chosen **after** dev rejected T — that makes 0.1391 post-hoc. Dev's
one-shot is SPENT (gotcha 10 of the original handoff). Options, in order of
rigour:
- **Preferred: fresh confirmation split.** Carve a new held-out set from
  *training* rows never used (we trained on 78,858 of 87,599 → 8,741 val → the
  remaining rows are all in training; so instead re-split: e.g. take 10% of the
  78,858 train rows out, retrain or reuse checkpoints only if the rows were
  truly unseen). **Do not reuse val.** If retraining is too expensive, fall back
  to the option below.
- Acceptable: keep Platt as a clearly-labelled post-hoc, single-confirmation
  analysis in the report, and pre-register the v2 comparison instead (Task A is
  genuinely pre-registered — its prediction is dated in the old handoff).

## 4. Task C — paper assembly (after A and B)

1. Story line (settled in session 2): baseline reproduction → confidence layer
   → level fails to transfer (measured, +9.66 pts) → ranking transfers →
   selective prediction with a guaranteed error budget → v2/Platt bound the
   remaining headroom.
2. Headline table = error budget (§Finding 6 of the findings doc) +
   risk–coverage with the 81.5 EM ceiling. EM/F1 tables go in setup only.
3. 4-page budget suggestion: 0.75p setup+data, 0.75p confidence+calibration
   method, 1.25p findings (level vs rank), 0.75p v2/Platt analysis,
   0.5p limitations (val double-use, winning-window, ECE binning).
4. Cite: Guo 2017, Platt 1999, Zhang 2021 (differentiate), SQuAD 1.1/2.0,
   BERT. The annotator-ceiling measurement (81.5 EM) and the gold-convention
   base-rate gap (+12.2 EM single-gold convention) are from the original v1
   handoff §1 — reuse those numbers, do not recompute.

---

## 5. Gotchas for the next session

1. **Dev is burned.** Never fit on it; every further dev *read* must be labelled
   post-hoc or backed by the fresh split from §3.1.
2. **Do not resubmit `jobs/run_all.sh`** — it resubmits training and would
   overwrite `runs/baseline/model`. (Session-1 gotcha, still true.)
3. **stujump is password-only**; BatchMode/key-only ssh fails with "partial
   success". One-shot `ssh soc '<cmd>'` works from a fresh terminal; pasted
   multi-line `python3 -c` blocks get mangled by the cluster shell (newlines
   inside `-c` → IndentationError). Prefer `scp soc:'~/BERT/runs/baseline/*.json'`
   style single-line commands, or heredoc-free scripts.
4. **results/* is gitignored** — use `git add -f results/...` and
   `results/figures/*.png` for any new figure; plots also live under
   `runs/baseline/plots/` (ignored, not pushed).
5. **The fixture lies twice now.** `runs/_fixture/` is synthetic noise: it hid
   the margin sign bug (session 2) and was already flagged in the v1 handoff
   gotcha 4. Any new signal MUST be tested for orientation on real-shaped data
   (the `test_signals.py` contract tests exist for this — extend them, don't
   bypass them).
6. **`_signal` contract: higher = more confident.** Margin is returned as-is;
   entropy is negated; NaN → −inf (np.argsort puts NaN last → would rank as
   most confident). New signals must satisfy `test_higher_means_more_confident`.
7. **Local venv has no torch** — `predict.py`'s main() cannot run locally; the
   symtable test (`test_predict.py::test_no_undefined_globals`) is the guard
   that caught the NameError. Keep it.
8. **Local analysis env:** python 3.14.6, numpy 2.5.4, matplotlib 3.11.2,
   pytest 9.1.1 — `pip freeze` is recorded in the results docs. Cluster env =
   requirements.txt pins + pyarrow 20.0.0. Report both.
9. **Report writes artifacts before raising** (`4a555b6`): the §6 checks are
   evaluated as data into `assertions.failures`. If checks 2/3 fail on a future
   run, read the failure strings before assuming a bug — they failed
   legitimately for the base-rate reason in session 2.

---

## 6. Definition of done for the v2 session

- [ ] `calibrate report` prints v1 and v2 rows on dev; Z computed over the full
      legal grid; v2 outcome (drop or no-drop) recorded in both results docs.
- [ ] `calibrate fit --method platt` exists; half-split recorded; 29+ tests
      green including new platt + orientation tests.
- [ ] §3.4 table regenerated from the pipeline; Platt row relabelled.
- [ ] Decision recorded: fresh confirmation split (yes/no, with reasoning).
- [ ] `results/` updated; everything pushed; cluster untouched (or, if a fresh
      split was trained: new jobs logged here with IDs and artifacts).
- [ ] Paper skeleton exists (outline at minimum) with the citation list from §1.
