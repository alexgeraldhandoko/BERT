# Reproducing the QA experiments

## Code version

The experiment references below identify the code/settings used for reported results.

| Experiment                  | Branch or tag                                        |
| --------------------------- | ---------------------------------------------------- |
| Baseline                    | `baseline-v1`                                        |
| Frozen reranking            | tag `final-frozen-v1`                                |
| Span-head reranking         | `span-head`                                          |
| Edge-aware reranking        | `exp/start-end-readouts`, tag `final-edge-frozen-v1` |
| Span-conditioned confidence | `exp/span-conditioned-confidence`                    |

## Environment

Run commands from the repository root on the SoC cluster.

- Python: `3.12`
- Install packages: `pip install -r requirements.txt`
- Set the source path: `export PYTHONPATH=src`
- Create the Slurm log directory before submitting jobs: `mkdir -p logs/slurm`

The QA environment used NumPy 1.26.4 and PyArrow 20.0.0 (`pyarrow<21`).

## Required inputs

- Course data: `data/raw/train-v1.1.json` and `data/raw/dev-v1.1.json`
- Baseline checkpoint: `runs/baseline/model`
- Configurations and split IDs: `configs/` and `splits/`
- Candidate exports and tables are generated under `exports/`; they are not committed.

## Run the experiments

### 1. Train the baseline

```bash
sbatch job_baseline.sbatch
```

Expected checkpoint: runs/baseline/model.

### Prepare frozen-reranker inputs

Export the held-out candidates:

```bash
sbatch job_export_heldout.sbatch
```

Build the candidate table:

```bash
PYTHONPATH=src ~/miniconda3/envs/qa/bin/python -u \
  -m qa_baseline.candidate_table \
  --exports exports/v1/heldout \
  --split splits/heldout-v1.json \
  --output exports/v1/heldout/table.npz
```

Build text features:

```bash
PYTHONPATH=src ~/miniconda3/envs/qa/bin/python -u \
  -m qa_baseline.text_features \
  --exports exports/v1/heldout \
  --validation runs/baseline/validation-v1.1.json \
  --output exports/v1/heldout/text_features.npz
```

### 2. Run the frozen reranker comparison

Requires the held-out candidate export, table, and text features.

```bash
sbatch job_cv_all_features.sbatch
sbatch job_freeze.sbatch
```

Record the frozen settings before evaluating on dev. Then prepare dev:

```bash
sbatch job_export_dev.sbatch

PYTHONPATH=src ~/miniconda3/envs/qa/bin/python -u \
  -m qa_baseline.dev_table \
  --exports exports/v1/dev \
  --dev data/raw/dev-v1.1.json
```

Score dev:

```bash
sbatch job_dev_score.sbatch
```

### 3. Run the span-head reranker

```bash
sbatch job_span_head.sbatch
sbatch job_fit_final_head.sbatch
sbatch job_evaluate_dev_head.sbatch
```

See the Python module docstrings for direct commands and arguments.

### 4. Run the edge-aware reranker

First export and prepare the held-out edge features. Then run cross-validation, freeze the head, export and prepare dev, and score it.

```bash
sbatch job_export_edges_heldout.sbatch
sbatch job_prepare_edges.sbatch
sbatch job_cv_start_end_l2wide.sbatch
sbatch job_fit_final_edges.sbatch
sbatch job_export_dev_edges.sbatch
sbatch job_dev_table_edges.sbatch
sbatch job_dev_score_edges.sbatch
```

### 5. Run the span-conditioned confidence head

First create confidence_examples.npz using the example-builder command documented in its Python module.

```bash
sbatch job_fit_span_confidence.sbatch
sbatch job_score_span_confidence.sbatch
```

Expected outputs include the fitted head and the dev confidence summary under exports/v1/final_confidence/.

#### Results and scoring

See results/ for the reported metrics and settings. Use the official scorer command documented in the experiment notes. The .sbatch and .sh files contain the run procedures.

```bash
PYTHONPATH=src ~/miniconda3/envs/qa/bin/python -u \
  -m qa_baseline.evaluate \
  --data data/raw/dev-v1.1.json \
  --predictions exports/v1/final/dev/predictions-frozen-head-plus-mix.json \
  --output exports/v1/final/dev/metrics-frozen-head-plus-mix.json
```
