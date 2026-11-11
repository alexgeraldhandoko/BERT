#!/bin/bash
set -euo pipefail
export PYTHONPATH=src

python -u -m qa_baseline.candidate_table \
  --exports exports/v1/heldout_edges \
  --split splits/heldout-v1.json \
  --output exports/v1/heldout_edges/table.npz

python -u -m qa_baseline.text_features \
  --exports exports/v1/heldout_edges \
  --validation runs/baseline/validation-v1.1.json \
  --output exports/v1/heldout_edges/text_features.npz

python - <<'PY'
import json
import numpy as np
from pathlib import Path

folder = Path("exports/v1/heldout_edges")
n = sum(1 for _ in (folder / "candidates.jsonl").open(encoding="utf-8"))
table = np.load(folder / "table.npz")
text = np.load(folder / "text_features.npz")["X"]

print("candidate rows:", n)
print("table rows:", len(table["q_index"]))
print("text features:", text.shape)

assert len(table["q_index"]) == n
assert text.shape[0] == n
PY
