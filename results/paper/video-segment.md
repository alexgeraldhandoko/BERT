# Video segment script — Kea, ~3 minutes (confidence calibration axis)

> Skeleton: `results/2026-10-28-temperature-findings.md` (plain-language).
> Rubric: presentation 30% — the bug-and-gate story earns marks; keep the
> numbers big and few. Times are targets, not a cage.

---

**[0:00 — Hook, ~20s]**

"Our model answers 80% of questions correctly — but it only *claims* 65%
confidence. It hedges when it shouldn't. This segment is about measuring
that gap, fixing it, and finding out which fixes transfer and which don't."

**[0:20 — The signal, ~30s]**

"For every answer, BERT gives us a probability: the product of a start-score
and an end-score. One problem — that product spreads probability over tokens
that *can't* be answers. So our first result: renormalise over legal answer
spans only. Zero fitted parameters. Brier improves from 0.152 to 0.146, the
confidence gap shrinks by a third. That fix was predicted before we ran it —
and confirmed."

**[0:50 — The negative result, ~50s]**

"Then the surprise. The textbook fix — temperature scaling, one parameter —
*fails*. We fit it on validation, and it makes every score on dev *worse*.
Why? This is the interesting part. Validation has one gold answer per
question; dev has three. More accepted answers push the accuracy up — the
base rates differ by almost 10 points. A temperature only controls the
*spread* of the confidences. It cannot move their *centre*. The centre is
set by the base rate. We fitted on a 70%-accurate dataset; dev needs 80%.
The correction pushed the wrong way. We proved this by fitting on dev
itself — diagnostic only — where the same method works perfectly. The method
isn't broken. The *transfer* is."

**[1:40 — The fix that works, ~30s]**

"The fix needs two parameters, not one: Platt scaling — a slope and an
*offset*. The offset is exactly the thing that can move the centre. Fitted
on validation, it transfers: Brier 0.139, better than everything except...
a 200-tree random forest from the literature — which it matches to three
decimal places. Two numbers versus two hundred trees. Calibration here isn't
complex structure. It's a level shift."

**[2:10 — The engineering story, ~40s]**

"One honest war story. Our first v2 fit returned a temperature at the very
edge of the search grid — a phantom result. The cause: three validation
questions had *no* legal answer spans, so their probabilities were NaN — and
NaN silently poisoned the grid search. The half-split stability gate caught
it: two halves disagreed, which is exactly what the gate exists to flag.
After the fix, the disagreement vanished. The lesson we'll carry: *assert
your invariants as data, and let the gates catch what you can't.*"

**[2:50 — Close, ~10s]**

"So: ranking was never the problem — AUC 0.83, and a 5% error budget holds
at 25% coverage. What calibration buys you is an honest number next to each
answer. Over to Ziying on the reranker."
