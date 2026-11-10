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

**2. Ziying** — saw your span-conditioned confidence head result (trained
head_train, temperature on calib — fully compliant, thank you). It's an
honest negative: raw beats it on AUROC/Brier/NLL. I'd like to include it in
the comparison table as the learned-head row (labelled as your result) — it
strengthens the paper's claim that calibration here is low-dimensional.
OK to include? Also: your split file is now the canonical one
(`splits/heldout-v1.json` → Alexander's shared file references it); the
course head_train (5,845) = your head_train + your test. Either way I'll
write the accuracy section to hand off to your reranker story: "the
reranker improves EM/F1 (+1.49 EM); the calibration layer attaches a
confidence with a guaranteed error budget (5% target holds on dev at 23.8%
coverage / 1.79% achieved)."

**3. Alexander** — saw your `68fab11` push: forest retrained on head_train
(5,845) is exactly what the protocol needed, and your shared
`splits/confidence-comparison-v1.json` matches the IDs I used byte-for-byte
(verified — my refits stand). But a bigger issue, raised before submission:
**the paper draft's "Platt-scaled Kamath" numbers are untraceable.** The
 ideation PDF (§4.1, Table 4) reports a 300-tree/depth-10 forest (Brier
0.1384) with T=1.0734 and Platt a=0.9037/b=+0.0786 fitted on calib (dev
Brier 0.1360) — none of that exists in any branch (your committed run is
200 trees, no scaling, 0.1405; I searched every ref). Please commit:
(a) the 300-tree config change, (b) the scaling-fitting code, (c) the
calibration JSON with a/b/T + half-split, (d) the dev summary. Easiest
path: export the forest's per-question calib+dev probabilities and I'll
run them through our tested Platt machinery (`calibrate fit --method
platt`) so the row gets the same gates and provenance as everything else.
Same for LIMECAL (Nicholas/Sandy): no code or artifacts in any branch.
Until then the draft's headline row is a chat-only number — the exact
failure mode we swore off in session 2. Also confirm: ECE binning in the
draft is 15 equal-WIDTH; ours is 15 equal-mass (documented reason:
confidences cluster near 0.8, equal-width goes sparse) — one convention
must win. And the draft's raw row (Brier 0.1535, EM 79.66) vs ours
(0.1523, EM 79.60) reflects the two different postprocess fixes — that
reconciliation is now the blocking issue for a consistent camera-ready.

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
