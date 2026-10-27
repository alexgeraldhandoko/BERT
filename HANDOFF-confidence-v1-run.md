# Handoff — Confidence v1 RUN & REPORT session

**Status.** All v1 code is written, tested, and pushed. The training pipeline is
**running on the cluster right now**. This session's remaining work is purely:
wait → fetch results → fit T → report → write up. Estimated hands-on time once
training finishes: **~20 minutes**, all local except one scp.

Author state: written while job `1026139` was at 21% (epoch 1/3, ~8 it/s).
Started 2026-10-27 ~19:55 SGT; ETA ~3 h for training, then two chained predict
jobs (~15–30 min).

---

## 1. Where everything stands

### Code (all pushed to `origin/experiment/temperature-confidence`)
| commit | content |
|---|---|
| `78b2a19` | v1 core: `postprocess.best_spans` refactor, `confidence.py` (15 tests), `predict.py` scores cache, `calibrate.py` fit/report, `jobs/` sbatch pipeline |
| `4c63eae` | merged `origin/baseline-results` → **pyarrow==20.0.0** pin (teammate's recorded fix) |
| `de4ae5c` | evaluate step wired into both predict sbatch jobs (→ `val-metrics.json`, `metrics.json`) |
| `99bf018` | stretch goals, already done: `entropy_signal`, `renormalised_span_probability` (**v2**, sums to 1 over legal spans), `fit_two_temperature` (T_s/T_e), `calibrate target` subcommand (error-rate targeting across signals). 22 tests green. Validated on synthetic fixtures. |

### Cluster pipeline (submitted, chained with `--dependency=afterok`)
| job | name | what it produces |
|---|---|---|
| `1026139` | cs4248-train | `runs/baseline/model`, `training.json`, `validation-v1.1.json` |
| `1026140` | cs4248-predict-val | `val-predictions.json`, `val-confidence.json`, `val-scores.npz`, `val-metrics.json` |
| `1026141` | cs4248-predict-dev | `predictions.json`, `confidence.json`, `scores.npz`, `metrics.json` |

Config: `configs/baseline.json` (seed 42, no `--full-train` — the 8,741 val
rows stay out of training, per handoff decision 4).

**What the cluster deliberately does NOT produce: the temperature.** The
chained jobs stop at uncalibrated confidence (T=1). Fitting T on the val
cache and reporting on dev are LOCAL steps (C and D below) — pure numpy on
`scores.npz`, no GPU — kept local so that (a) dev reporting is a conscious,
one-shot command, never an accidental sweep, and (b) matplotlib (plots) stays
out of the pinned training env. If `calibration.json` is missing, that is
expected until step C runs.

### Environment facts learned the hard way
- **stujump is password-only, pass-through** ("PTY allocation failed" if you
  try a shell on it) and **blocks extra mux channels** — ControlMaster does
  NOT work. Fresh `ssh -J stujump xlogin` sessions need a password each time.
  One-shot ssh from local *does* work now (BatchMode key auth succeeded
  2026-10-27 evening; alias `ssh soc` exists in `~/.ssh/config`).
- Repo is **public** — plain `git clone https://github.com/alexgeraldhandoko/BERT.git`
  works from the cluster; no PAT needed; `git pull` non-interactive.
- Conda env `bert4248` (python 3.12) at `$HOME/miniconda3/envs/bert4248`;
  conda works **only on compute nodes** (xlogin crashes it).
- `gpufree` (guide helper) shows live GPU availability; h100-47 was chosen
  (6/40 free, matches teammate's reference hardware).
- Data verified on cluster: `~/BERT/data/raw/*.json`, sha256
  `352766…955` (train) / `95aa6a…2c9` (dev) — identical to local.

---

## 2. Session checklist (in order)

### Step A — confirm the pipeline finished
```bash
ssh soc 'squeue --me -h -o "%i %j %T %M"'
ssh soc 'tail -3 ~/BERT/logs/slurm/cs4248-predict-dev-1026141.out'
ssh soc 'ls -la ~/BERT/runs/baseline/'
```
Expect: no jobs in queue (or only finished), dev log ends "Dev prediction
finished", all 10 artifacts listed in §1 present. If a job failed, its log is
`~/BERT/logs/slurm/<name>-<id>.out` — diagnose before resubmitting
(`sbatch` the failed script again; dependencies auto-clear).

### Step B — fetch results (from the LOCAL repo root, 2 password prompts)
```bash
scp -o ProxyJump=keaharv@stujump.comp.nus.edu.sg \
  'keaharv@xlogin.comp.nus.edu.sg:~/BERT/runs/baseline/{training.json,validation-v1.1.json,val-predictions.json,val-confidence.json,val-scores.npz,val-metrics.json,predictions.json,confidence.json,scores.npz,metrics.json}' \
  runs/baseline/
```
(~70 MB; `scores.npz` ≈ 35 MB, `val-scores.npz` ≈ 30 MB.)

### Step C — fit temperature on the validation partition (local)
```bash
PYTHONPATH=src .venv/bin/python -m qa_baseline.calibrate fit \
  --scores runs/baseline/val-scores.npz \
  --data runs/baseline/validation-v1.1.json \
  --predictions runs/baseline/val-predictions.json \
  --output runs/baseline/calibration.json
```
Checks: y base rate should be ≈ 0.697 (val EM, teammate's reference 69.73);
half-split `T_A ≈ T_B` within ±0.05 (1 parameter on 8.7k rows). If T_A and
T_B disagree wildly, STOP and investigate before touching dev — dev is
one-shot (handoff gotcha 10).

### Step D — report on dev (local; 3 hard assertions run automatically)
```bash
PYTHONPATH=src .venv/bin/python -m qa_baseline.calibrate report \
  --scores runs/baseline/dev-scores.npz \
  --data data/raw/dev-v1.1.json \
  --predictions runs/baseline/predictions.json \
  --calibration runs/baseline/calibration.json \
  --metrics runs/baseline/metrics.json \
  --output runs/baseline/calibration-report.json \
  --plot-dir runs/baseline/plots
```
Prints the EM/F1/mean_c/base/Brier/BSS/ECE/NLL table (uncalibrated vs +T) and
writes 3 plots. Assertions (any failure = bug, do not ship):
1. EM/F1 match our own `metrics.json` (answers unchanged — T is monotone)
2. NLL decreased
3. Brier AND ECE decreased

Expected landscape (from handoff §5.1): v1 confidence is systematically
**deflated** (product not renormalised, sums ≪ 1 over legal spans) — expect
mean_c well below base, and T likely **< 1** (raising confidence). BSS vs
base(1−base) ≈ 0.19·(dev base ≈ 0.7) is the "is this worth anything" line.

### Step E — error-rate targeting (the most persuasive result; handoff §8)
```bash
PYTHONPATH=src .venv/bin/python -m qa_baseline.calibrate target \
  --val-scores runs/baseline/val-scores.npz \
  --val-data runs/baseline/validation-v1.1.json \
  --val-predictions runs/baseline/val-predictions.json \
  --scores runs/baseline/dev-scores.npz \
  --data data/raw/dev-v1.1.json \
  --predictions runs/baseline/predictions.json \
  --calibration runs/baseline/calibration.json \
  --target-error 0.05 \
  --output runs/baseline/target-report.json
```
Caveats learned from the fixture work: (a) cal-vs-uncal is ranking-identical
(T is monotone) — the meaningful comparison is **across signals** (prob vs
margin vs entropy); (b) a signal that can't reach the target is reported as
`UNREACHABLE` (a finding, not a crash); (c) dev gold convention differs from
val (~3 golds vs 1) — never compare ECE across the two files.

### Step F — write `results/2026-10-27-confidence-v1.md`
Required contents (handoff §8 definition of done): config, seed (42), commit
(`99bf018`), `pip freeze` (local `.venv/bin/pip freeze` + note cluster env is
requirements.txt pins + pyarrow==20.0.0), T, the §D table, the 3 plots,
half-split result, and the §E targeting table. Reference lines for the
narrative: human ceiling **81.5 EM** on dev (inter-annotator), teammate's
baseline **val EM 69.73 / dev EM 79.66**, base(1−base) Brier floor.

---

## 3. Gotchas specific to this session (beyond the original handoff §7)
1. **`ssh soc` may silently hang** if a stale ControlMaster socket exists —
   `ssh -O check soc` / remove `~/.ssh/soc-cm-*` then retry plain BatchMode.
2. **Never resubmit the whole `run_all.sh`** after partial failure — sbatch
   the single failed job; the dependency chain re-runs from there.
3. **`scores.npz` rows are the WINNING window only** — that's per design
   (handoff gotcha 5); confidence comes from that window alone.
4. **The fixture (`runs/_fixture/`) is synthetic noise** — prob ranking looks
   great there, margin/entropy look useless; do not extrapolate to real data.
5. **`target`'s threshold transfers between monotone transforms** (midpoint
   of the boundary gap), so the same τ is meaningful for T-scaled inference.
6. Two-temperature fit: `T_s`/`T_e` **share a base-rate budget** — sharpening
   one flattens the other. Don't interpret the cross-term as a bug.
7. v2's `renormalised_span_probability` exists and is tested but is NOT wired
   into any report yet — that is the natural v2 experiment (swap it for the
   v1 signal in `report` and compare ECE/Brier; handoff §9 predicts a Brier
   drop since v1 is systematically deflated).

## 4. Definition of done (mapping to the original handoff §8)
- [x] `best_answers` behaviour unchanged; 3 original tests green
- [x] `predict.py` writes predictions.json (format unchanged) + confidence.json + scores.npz
- [x] `confidence.py` + 22 unit tests
- [x] `calibrate.py fit` → calibration.json from the validation partition (code done; run in step C)
- [x] `calibrate.py report` → table + plots + assertions (code done; run in step D)
- [ ] half-split T check recorded (step C/D output → results file)
- [ ] `results/2026-10-27-confidence-v1.md` (step F)
- [x] stretch: margin/entropy signals; separate T_s/T_e; error-rate targeting
