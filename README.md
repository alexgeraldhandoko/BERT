# CS4248 BERT extractive QA baseline

This repository is the common starting point for experiments that try to improve SQuAD 1.1 exact match (EM) and token F1. The baseline fine-tunes `google-bert/bert-base-uncased` to predict answer start and end positions. Calibration, ECE, and other research ideas are optional branch work, not required by this scaffold.

The course provides `train-v1.1.json` for training and `dev-v1.1.json` for final testing. It requires the official `evaluate-v2.0.py` scorer. The scorer is copied without changes into [`vendor/evaluate-v2.0.py`](vendor/evaluate-v2.0.py); our [`src/qa_baseline/evaluate.py`](src/qa_baseline/evaluate.py) only checks that all question IDs have predictions and calls it. See [`SOURCES.md`](SOURCES.md) for the exact origin and placement of the linked material.

## Setup

Use Python 3.12. The local `data/raw/` folder already contains the course files, but Git ignores them. Teammates who clone the repo must put the two SQuAD JSON files from Canvas in `data/raw/`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests -v
```

The first training run also downloads the pretrained BERT checkpoint and tokenizer. The direct Python dependencies are version-pinned in `requirements.txt`; save `python -m pip freeze` with any result you report, since transitive packages and hardware can still differ.

## Train the shared baseline

```bash
PYTHONPATH=src python -m qa_baseline.train \
  --config configs/baseline.json \
  --data data/raw/train-v1.1.json \
  --output runs/baseline
```

The script makes a stable, approximately 90/10 *article-level* split of the training file using the configured seed. It fine-tunes on the larger partition, uses the other for validation loss, saves the checkpoint at `runs/baseline/model/`, and writes the held-out questions in official SQuAD format to `runs/baseline/validation-v1.1.json`. Questions and token windows from the same article stay in one partition.

## Compare experiments on held-out training questions

```bash
PYTHONPATH=src python -m qa_baseline.predict \
  --data runs/baseline/validation-v1.1.json \
  --model runs/baseline/model \
  --output runs/baseline/validation-predictions.json

PYTHONPATH=src python -m qa_baseline.evaluate \
  --data runs/baseline/validation-v1.1.json \
  --predictions runs/baseline/validation-predictions.json \
  --output runs/baseline/validation-metrics.json
```

The evaluator prints `exact`, `f1`, and `total`. The prediction JSON is a plain mapping from question ID to one answer string, which every experiment should preserve. Use the same held-out split and scorer on every branch. The official dev file is for the final common comparison; repeated design decisions based on dev results can overfit that test set.

After choosing an approach, you can retrain it on *all* of `train-v1.1.json` with `--full-train --output runs/final`. This mode skips the internal validation split. Keep this final run separate from the branch-comparison run.

## Final course dev evaluation

```bash
PYTHONPATH=src python -m qa_baseline.predict \
  --data data/raw/dev-v1.1.json \
  --model runs/baseline/model \
  --output runs/baseline/predictions.json

PYTHONPATH=src python -m qa_baseline.evaluate \
  --data data/raw/dev-v1.1.json \
  --predictions runs/baseline/predictions.json \
  --output runs/baseline/metrics.json
```

`runs/` and the raw datasets are not committed. Record final numbers, config, seed, commit, package versions, hardware, and commands in a short file under `results/`. Do not claim a baseline EM/F1 score until training and evaluation have actually run.

## Branches

1. First run and review the baseline, then commit and tag that working state as `baseline-v1`.
2. Each member branches from that tag, for example `git switch -c exp/alex/span-selection baseline-v1`.
3. Keep one main hypothesis per branch. Change any model, data, training, or postprocessing code needed for that hypothesis, but retain the shared prediction format and official scorer.
4. Submit a pull request with the hypothesis, commands, validation EM/F1 versus baseline, and any optional experiment-specific metric. Review before merging to `main`.

The initial config values are examples, not course-mandated settings. `max_length` includes the question and special tokens, so the available passage window is smaller. `doc_stride` is the overlap between passage windows. A gold answer must fit wholly in at least one window for this span model to learn or return it; the overlap does not solve arbitrarily long spans.

