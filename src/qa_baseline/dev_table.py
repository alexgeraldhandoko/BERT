"""Build the candidate table and text features for the dev export, with every dev question in the test part.

  PYTHONPATH=src python -m qa_baseline.dev_table --exports exports/v1/dev --dev data/raw/dev-v1.1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .candidate_table import build_table, summarize
from .text_features import build_text_features


def build_dev_inputs(exports_dir, dev_json, output_dir=None):
    exports_dir = Path(exports_dir)
    output_dir = Path(output_dir) if output_dir else exports_dir
    ids = [q["id"] for q in json.loads((exports_dir / "questions.json").read_text(encoding="utf-8"))]
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "dev_split.json"
    split_path.write_text(
        json.dumps({"parts": {"head_train": {"ids": []}, "calib": {"ids": []}, "test": {"ids": ids}}}) + "\n",
        encoding="utf-8",
    )
    table = build_table(exports_dir, split_path)
    X, names = build_text_features(exports_dir, dev_json)
    np.savez_compressed(output_dir / "table.npz", **table)
    np.savez_compressed(output_dir / "text_features.npz", X=X, names=np.array(names))
    return table, X


def export_rank_one(exports_dir):
    """The export's own rank-1 EM and F1 (percent) from questions.json, or None if the file lacks them."""
    questions = json.loads((Path(exports_dir) / "questions.json").read_text(encoding="utf-8"))
    if not questions or not all("top1_em" in q and "top1_f1" in q for q in questions):
        return None
    return (100 * float(np.mean([q["top1_em"] for q in questions])),
            100 * float(np.mean([q["top1_f1"] for q in questions])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="default: the exports folder")
    args = parser.parse_args()
    table, X = build_dev_inputs(args.exports, args.dev, args.output_dir)
    entry = summarize(table)["test"]
    print(f"Dev: {entry['questions']} questions ({entry['no_candidates']} without candidates), "
          f"{entry['candidates']} candidates, {X.shape[1]} text features")
    exported = export_rank_one(args.exports)
    print(f"Table rank-1 EM {entry['em@1']:.2f}  F1 {entry['f1@1']:.2f}")
    if exported is None:
        print("questions.json has no top1_em / top1_f1 fields, so the table could not be compared with the export.")
    else:
        print(f"Export's own rank-1 EM {exported[0]:.2f}  F1 {exported[1]:.2f}")
        if abs(entry["em@1"] - exported[0]) > 0.01 or abs(entry["f1@1"] - exported[1]) > 0.01:
            raise SystemExit("MISMATCH between the table and the export: do not continue")

if __name__ == "__main__":
    main()