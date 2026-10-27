# Plan — Confidence output v1 implementation

> Status: **grilling phase** — decisions being settled one by one with user.
> Source spec: `HANDOFF-confidence-v1.md` (read it; it is the authority on WHAT).
> This file records HOW: environment, sequencing, and the decisions made during grilling.

## Context
Implement confidence-v1 per the handoff: refactor `best_answers` to keep scores,
add `confidence.py` (pure numpy metrics) and `calibrate.py` (fit T on val,
report on dev). Nothing in `src/` has been touched yet; `runs/` is empty.

## Open decisions (grill log)
| # | Question | Status | Decision |
|---|---|---|---|
| 1 | Remote execution environment | RESOLVED | Code locally, execute on SoC cluster via **sbatch** (SLURM). 3 GPU jobs total: train (once), predict-on-val (once), predict-on-dev (once). All calibration/analysis is pure numpy on cached `scores.npz` → runs locally. |
| 2 | Does a trained checkpoint already exist on the cluster? | RESOLVED | No — user has never used the cluster. First-time setup per `paper_sources/SoC_Cluster_Guide.pdf` is required. |
| 3 | Code/data sync mechanism | RESOLVED (mostly) | Code: git push → clone on cluster (`origin = github.com:alexgeraldhandoko/BERT.git`). Data: `curl` SQuAD JSONs from rajpurkar/SQuAD-explorer directly on xlogin (guide sanctions login node for downloads; matches SOURCES.md). |
| 4 | sbatch specifics | PARTIALLY OPEN | Guide gives `job_template.sbatch` + `dummy_pipeline.sh`, logs to `logs/slurm/<name>-<id>.out`. Open: python 3.11 vs 3.12; GPU request flags; time limit for training. |
| 5 | How results come back to local | OPEN (mode known) | No VPN; stujump = password-only pass-through, blocks mux channels. Human touches per phase: 1 paste to launch chained pipeline, 1 paste/command to fetch results. Transfer mechanism TBD: scp pull vs cluster-side git commit. |
| 6 | Env python version | RESOLVED (default) | python=3.12 to match requirements.txt pins; user did not object. |

## Environment (facts)
- Local repo: only "Initial commit"; handoff + EDA files untracked.
- `runs/` empty locally — **no trained checkpoint exists in this repo**.
- Git remote: `git@github.com:alexgeraldhandoko/BERT.git`.
- Data sizes: train-v1.1.json 29 MB, dev-v1.1.json 4.6 MB (gitignored; curl from rajpurkar on cluster).
- requirements.txt: torch 2.6.0, transformers 4.52.4, datasets 3.6.0, numpy 1.26.4, accelerate 1.6.0; header says "Python 3.12 baseline" (guide example uses 3.11).
- User has completed guide steps 1–2: SoC account provisioned, StuJump SSH working.
- Guide path: `ssh -J <user>@stujump.comp.nus.edu.sg <user>@xlogin.comp.nus.edu.sg`;
  setup must happen on a compute node (`salloc -t 60` + `srun --pty bash`), never xlogin;
  no VSCode Remote-SSH; logs at `logs/slurm/<name>-<id>.out`.
- Open risks from guide: HF model download from compute nodes; storage quota for runs/ (checkpoint ~400 MB).
- **Probed live**: stujump = password-only, pass-through ("PTY allocation failed, account currently not available" on direct ssh), blocks extra mux channels (echo over ControlMaster hangs >60s). Direct xlogin times out off-campus. No NUS VPN.
- **=> Operating mode: "one paste".** User's interactive stujump→xlogin session is the only door. Pipeline is chained with `sbatch --dependency=afterok:` so ONE paste launches train → predict-val → predict-dev unattended. User then fetches results (scp or git pull) in a second touch.
- Cluster facts from probe: host xlogin2, slurm 26.05.2, conda NOT installed (guide step 5 pending, must be on compute node).

## Steps (draft, to be finalized after grilling)
- [ ] TBD

## Verification
- [ ] TBD
