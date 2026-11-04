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

## Kamath-style confidence baseline (this branch)

This experiment **does not fine-tune BERT again**. It runs the existing QA checkpoint, fits a separate random forest to predict whether BERT's final answer is exactly correct, and compares its confidence with raw start/end softmax confidence. It follows the seven-feature idea in [Kamath et al. (2020)](https://aclanthology.org/2020.acl-main.503/), adapted to this repository's cross-window span selector. The paper evaluated selective QA under domain shift; this branch evaluates SQuAD 1.1 confidence calibration.

Run these commands from the repository root on the SoC cluster after installing `requirements.txt` and placing the course SQuAD JSON files in `data/raw/`. Run BERT inference in a Slurm compute allocation, not on a login node. The 416 MB checkpoint stays at `/home/e/e1398508/baseline-model-v1`; it is never copied into this repo.

First, **confirm with the teammate's training record that this checkpoint was trained on the complementary 90% split**, using the same train file, seed 42, and article-level split. If the checkpoint was trained on all questions, these held-out questions are not valid forest-training data.

```bash
PYTHONPATH=src python -m qa_baseline.kamath prepare-validation \
  --data data/raw/train-v1.1.json \
  --output runs/kamath/validation-v1.1.json

PYTHONPATH=src python -m qa_baseline.kamath collect \
  --data runs/kamath/validation-v1.1.json \
  --model /home/e/e1398508/baseline-model-v1 \
  --role calibration \
  --output runs/kamath/calibration-records.json

PYTHONPATH=src python -m qa_baseline.kamath fit \
  --records runs/kamath/calibration-records.json \
  --output runs/kamath/forest.joblib

PYTHONPATH=src python -m qa_baseline.kamath collect \
  --data data/raw/dev-v1.1.json \
  --model /home/e/e1398508/baseline-model-v1 \
  --role evaluation \
  --output runs/kamath/dev-records.json

PYTHONPATH=src python -m qa_baseline.kamath evaluate \
  --records runs/kamath/dev-records.json \
  --data data/raw/dev-v1.1.json \
  --forest runs/kamath/forest.joblib \
  --output runs/kamath/evaluation \
  --summary results/kamath-summary.json
```

The seven forest columns, in fixed order, are passage word count, predicted answer word count, and the five largest candidate-span probabilities. Each span probability is the product of start and end softmax probabilities within its window. Duplicate character spans across overlapping windows count once; absent candidates are filled with zero. The **raw softmax** comparator is the start/end product for the span selected by the baseline's highest start-plus-end logit score across windows. These cross-window products are not a single normalized distribution.

The forest is fitted only on the held-out training questions. `evaluate` checks that dev records came from the same BERT checkpoint and a different SQuAD file, writes question-level confidence values and the unchanged BERT predictions under `runs/kamath/evaluation/`, and writes a small, Git-trackable metrics file to `results/kamath-summary.json`. ECE uses 10 equal-width bins; EM/F1 come from the official scorer. The train held-out split has one gold answer per question while dev often has several, so interpret cross-split calibration differences with that labeling difference in mind.

After the cluster run, copy `results/kamath-summary.json` back to the same path in the local checkout, along with any detailed `runs/kamath/evaluation/` files you need for analysis. Commit the source changes and summary; keep the BERT checkpoint, forest file, raw datasets, and large prediction files outside Git. Record `python -m pip freeze`, the Git commit, Slurm job/hardware, and checkpoint SHA-256 in the experiment notes.

## Branches

1. First run and review the baseline, then commit and tag that working state as `baseline-v1`.
2. Each member branches from that tag, for example `git switch -c exp/alex/span-selection baseline-v1`.
3. Keep one main hypothesis per branch. Change any model, data, training, or postprocessing code needed for that hypothesis, but retain the shared prediction format and official scorer.
4. Submit a pull request with the hypothesis, commands, validation EM/F1 versus baseline, and any optional experiment-specific metric. Review before merging to `main`.

The initial config values are examples, not course-mandated settings. `max_length` includes the question and special tokens, so the available passage window is smaller. `doc_stride` is the overlap between passage windows. A gold answer must fit wholly in at least one window for this span model to learn or return it; the overlap does not solve arbitrarily long spans.

