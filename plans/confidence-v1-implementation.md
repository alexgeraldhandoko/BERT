# Plan — Confidence output v1 implementation

> Source spec: `HANDOFF-confidence-v1.md` (authority on WHAT). This file records HOW.
> Status: v1 code is **implemented, tested, and pushed** to
> `origin/experiment/temperature-confidence` (commits 78b2a19, 4c63eae).
> What remains is cluster execution + local reporting.

## Context
Emit a calibrated probability that the SQuAD model's answer is correct:
fine-tune normally → span probability as confidence → one temperature fitted
on the held-out validation partition → reported one-shot on dev. Brier is the
primary metric (proper scoring rule); ECE diagnoses. Hard constraint: the
`{qid: str}` predictions contract is untouched.

## Environment decisions (settled through grilling)
| # | Decision |
|---|---|
| 1 | Code locally, execute on SoC cluster via sbatch. No VPN; stujump is password-only pass-through and blocks mux channels → **"one paste" mode**: one interactive session, one paste to launch, one command to fetch results. |
| 2 | No existing checkpoint. Teammate's baseline (origin/baseline-results, H100 job 1015181, val EM 69.73/F1 82.64) lives in THEIR cluster home; verified **nothing >5 MB exists on any git ref** → **we retrain** (same seed 42/config/split → statistically equivalent). |
| 3 | Env: conda env `bert4248`, python 3.12, on a compute node (never xlogin). pyarrow==20.0.0 pinned (teammate's recorded fix; supersedes the 17.0.0 guess). |
| 4 | Data: `curl` from `https://rajpurkar.github.io/SQuAD-explorer/dataset/` on xlogin; sha256 verified (352766…955 train, 95aa6a…2c9 dev). User has files under ~/BERTdata — move into ~/BERT/data/raw. |
| 5 | Results back to local: **scp through stujump** (scores.npz ~35 MB; no tokens on cluster). |

## What is already done (local, pushed)
- [x] `postprocess.best_spans` refactor; `best_answers` wrapper; contract unchanged; 3 existing tests green
- [x] `confidence.py`: span_probability, span_margin, correctness (vendor scorer, gold-first, max over golds), ece (equal-mass), brier, nll, BSS, reliability_bins, risk_coverage, fit_temperature — 15 unit tests
- [x] `predict.py`: writes scores.npz (winning-window logit rows) + confidence.json (T=1); predictions.json format unchanged
- [x] `calibrate.py`: `fit` (T on val + by-article half-split check) / `report` (table + 3 hard assertions + reliability/risk-coverage/histogram plots)
- [x] End-to-end synthetic fixture validation: T recovered exactly, all 3 assertions passed, half-split delta 0.01
- [x] `jobs/`: train.sbatch, predict-val.sbatch, predict-dev.sbatch, run_all.sh (chains via --dependency=afterok)

## Remaining steps
- [ ] 1. Wire `evaluate.py` (existing module — teammate's `train_baseline.sh` already used it for val) into my two new predict sbatch jobs, mirroring their pattern: predict-val job also writes `val-metrics.json`, predict-dev job also writes `metrics.json`. This gives assertion 1 its baseline EM/F1 from OUR run. Update jobs/*.sbatch; commit + push.
- [ ] 2. User clones repo on cluster via **HTTPS + read-only PAT** (port 22 to GitHub blocked; PAT needs only `Contents: Read` — the cluster never pushes: code flows in via git, results flow out via scp, pushes happen from the local Mac over the existing SSH key). Then `git checkout experiment/temperature-confidence`.
- [ ] 3. User env fix: `pip install "pyarrow==20.0.0"`; verify `import pyarrow, datasets`.
- [ ] 4. User moves ~/BERTdata/*.json into ~/BERT/data/raw; verify sha256.
- [ ] 5. User runs one paste: `bash ~/BERT/jobs/run_all.sh` → train (H100, hours) → predict-val + predict-dev auto-chained.
- [ ] 6. User fetches results: scp -o ProxyJump=stujump xlogin:~/BERT/runs/baseline/{scores.npz,predictions.json,confidence.json,metrics.json,val-*} → local runs/baseline/.
- [ ] 7. Local: `calibrate fit` on the **8,741-question validation partition** (written by train.py to `runs/baseline/validation-v1.1.json`; single gold per question): y = correctness of val predictions vs gold (vendor scorer), T = argmin NLL over grid. The **half-split check** is a sanity check on T itself, not a new partition: cut the 8,741 val questions BY ARTICLE into halves A/B, fit T on each, expect T_A ≈ T_B (±0.05) — one parameter on 8.7k rows has nothing to overfit. dev (10,570, ~3 golds) is never used for fitting.
- [ ] 8. Local: `calibrate report` on dev → table + 3 plots + 3 hard assertions (install matplotlib locally already done).
- [ ] 9. Record `results/<date>-confidence-v1.md`: config, seed, commit, pip freeze, T, table, plots, half-split result.

## Files
- `src/qa_baseline/{postprocess,confidence,predict,calibrate}.py` (done; predict/calibrate may get minor fixes)
- `jobs/*.sbatch`, `jobs/run_all.sh` (step 1 touches the two predict scripts)
- `results/<date>-confidence-v1.md` (new, step 9)

## Reuse
- `vendor/evaluate-v2.0.py` compute_exact/compute_f1/normalize_answer (gold first, max over golds) via `confidence._load_vendor()`
- `src/qa_baseline/evaluate.py` CLI for metrics.json; `features.prediction_features` context mask (`offset is not None`)
- Guide conventions: gpu-long partition, h100-47:1, logs/slurm/<name>-<id>.out, conda activate with set +u

## Verification
- 18 unit tests green locally (`.venv/bin/python -m pytest tests/ -q` with PYTHONPATH=src)
- After step 5: train log shows `cuda: True`; training takes ~1.5–3 h on H100 (teammate's job 1015181 is the reference; 08:00:00 is a safety ceiling, not an expectation); val EM ≈ 69.7 ± noise vs teammate's record; dev EM ≈ 79.7 ± noise
- After step 8: assertion 1 (EM/F1 match our own metrics.json), assertion 2 (NLL decreased), assertion 3 (Brier & ECE decreased); half-split |T_A−T_B| small (≈±0.05 expected)
- Sanity: predictions.json is `{qid: str}`; `evaluate.py` accepts it (it validates the contract)
