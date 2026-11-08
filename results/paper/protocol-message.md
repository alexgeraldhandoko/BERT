# Protocol message — draft for Kea to send (Telegram, Ziying + Alexander)

> Paste as-is. Numbers all from pipeline runs; links to repo docs below.

---

Hi both — protocol update + proposal for the confidence part of the report
(deadline 12 Nov). The TA's split spec is now binding and I've refit
everything to comply:

**0. Binding split (TA).** 78,858 QA_train → BERT; 8,741 held-out = 5,845
head_train (forest/reranker training) + 2,896 calib (ALL temperature/Platt
fits); final comparison on the 10,570 dev IDs. The 2,896 calib IDs are
exactly the `calib` part of Ziying's `splits/heldout-v1.json` (commit
`f24d049` on her branch) — no re-splitting needed, use that file.

**1. Proposed confidence comparison table** (dev; all calibrators fitted on
the 2,896 calib IDs only):

| method | params | dev Brier | ECE | AUC |
|---|---|---|---|---|
| raw softmax | 0 | 0.1523 | 0.1457 | 0.8293 |
| + temperature | 1 | 0.1564 | 0.1586 | 0.8293 |
| v2 renormalised | 0 | 0.1456 | 0.1175 | 0.8187 |
| + Platt (v1 signal) | 2 | 0.1410 | 0.1239 | 0.8293 |
| Kamath forest (cited) | ~200 trees | 0.1388* | 0.1267 | 0.8357 |

*re-measure needed, see point 3.

Story in one line: the zero-parameter renormalisation comes within 0.005
Brier of the best learned calibrator (2 parameters); the 1-parameter
temperature is an honest negative (base-rate mismatch between held-out and
dev, diagnosed and measured).

**2. Ziying** — does your span-head confidence output want to join this
table? Also: your split file is now the canonical one (`splits/heldout-v1.json`) —
the course head_train (5,845) = your head_train + your test; your calib
(2,896) is the course calib. If your confidence output joins, tell me which
row/label and I'll add it. Either way I'll write the accuracy section to
hand off to your reranker story: "the reranker improves EM/F1 (+1.49 EM);
the calibration layer attaches a confidence with a guaranteed error budget
(5% target holds on dev at 23.8% coverage / 1.79% achieved)."

**3. Alexander** — two things. (a) The course forbids the forest from
training on the calib IDs, so please retrain the Kamath forest on the
5,845 head_train IDs (= your head_train + test from the same split file)
and send me the new summary — your 0.1388 was fitted under the old
protocol, so the cited row needs the new number. (b) Confirm you're using
the same baseline checkpoint (78,858 QA_train, seed 42) so the comparison
is apples-to-apples on dev.

Full technical docs are on branch `experiment/temperature-confidence`:
`results/2026-10-27-confidence-v1.md` (§11 = binding-protocol table, §10 =
superseded full-val table kept as provenance) and
`results/2026-10-28-temperature-findings.md` (plain-language).

Merge plan (later, after the report draft exists): consolidation branch ←
`alex/baseline_kamath` first, then your span-head line, then my
calibration line — `postprocess.py`/`predict.py` resolved by hand. Not
through `main` blindly.

---

*Kea: after sending, record their answers at the top of
`HANDOFF-confidence-v3.md` §3.*
