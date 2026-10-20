"""Validate predictions and run the unmodified course-provided official scorer."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .data import load_squad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/dev-v1.1.json"))
    parser.add_argument("--predictions", type=Path, default=Path("runs/baseline/predictions.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/baseline/metrics.json"))
    args = parser.parse_args()

    expected = {example["id"] for example in load_squad(args.data)}
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    if not isinstance(predictions, dict) or any(not isinstance(v, str) for v in predictions.values()):
        raise ValueError("Predictions must be a JSON object mapping question IDs to answer strings")
    missing = expected - predictions.keys()
    extra = predictions.keys() - expected
    if missing or extra:
        raise ValueError(f"Prediction IDs differ from data: {len(missing)} missing, {len(extra)} extra")

    scorer = Path(__file__).resolve().parents[2] / "vendor" / "evaluate-v2.0.py"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(scorer), str(args.data), str(args.predictions), "--out-file", str(args.output)],
        check=True,
    )
    metrics = json.loads(args.output.read_text(encoding="utf-8"))
    print(json.dumps({"exact": metrics["exact"], "f1": metrics["f1"], "total": metrics["total"]}, indent=2))


if __name__ == "__main__":
    main()

