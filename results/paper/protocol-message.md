# Protocol message — draft for Kea to send (Telegram, Ziying + Alexander)

> Paste as-is. Numbers all from pipeline runs; links to repo docs below.

---

Hi both — protocol proposal for the confidence part of the report (deadline
12 Nov). Short version:

**1. Proposed confidence comparison table** (dev, all calibrators fitted on
the full validation partition — the split protocol I proposed earlier; speak
up if that's a problem):

| method | params | dev Brier | ECE | AUC |
|---|---|---|---|---|
| raw softmax | 0 | 0.1523 | 0.1457 | 0.8293 |
| + temperature | 1 | 0.1543 | 0.1522 | 0.8293 |
| v2 renormalised | 0 | 0.1456 | 0.1175 | 0.8187 |
| + Platt | 2 | 0.1391 | 0.1171 | 0.8293 |
| Kamath forest (cited) | ~200 trees | 0.1388 | 0.1267 | 0.8357 |

Story in one line: 2 parameters match a 200-tree forest; the cheapest good
fix needs zero fitted parameters; the 1-parameter temperature is an honest
negative (base-rate mismatch between val and dev, diagnosed and measured).

**2. Ziying** — does your span-head confidence output want to join this
table? Your thirds are fine as your internal thing (disclosed either way);
if it joins, tell me which row/label and I'll add it. Either way I'll write
the accuracy section to hand off to your reranker story: "the reranker
improves EM/F1 (+1.49 EM); the calibration layer attaches a confidence with
a guaranteed error budget (5% target holds on dev at 25.4% coverage)."

**3. Alexander** — can you confirm the Kamath numbers I'm citing (dev Brier
0.1388, ECE 0.1267, AUC 0.8357, val-fit/dev-report protocol) match your
summary.json? I want the cited row to trace to your run, not my re-read.

Full technical docs are on branch `experiment/temperature-confidence`:
`results/2026-10-27-confidence-v1.md` (§10 table) and
`results/2026-10-28-temperature-findings.md` (plain-language).

Merge plan (later, after the report draft exists): consolidation branch ←
`alex/baseline_kamath` first, then your span-head line, then my
calibration line — `postprocess.py`/`predict.py` resolved by hand. Not
through `main` blindly.

---

*Kea: after sending, record their answers at the top of
`HANDOFF-confidence-v3.md` §3.*
