"""Generate one SQuAD answer per question from a fine-tuned checkpoint.

Writes, next to the historical {qid: str} predictions.json:
- scores.npz: per-question logit-row cache (winning window only) so all
  downstream confidence/calibration experiments run without re-invoking BERT;
- confidence.json: {qid: {"c": float}} uncalibrated span probability (T=1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .confidence import span_probability
from .data import load_squad
from .features import prediction_features
from .postprocess import best_spans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--data", type=Path, default=Path("data/raw/dev-v1.1.json"))
    parser.add_argument("--model", type=Path, default=Path("runs/baseline/model"))
    parser.add_argument("--output", type=Path, default=Path("runs/baseline/predictions.json"))
    args = parser.parse_args()

    from datasets import Dataset
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments

    config = json.loads(args.config.read_text(encoding="utf-8"))
    examples = load_squad(args.data)
    max_length = config["max_length"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)

    raw = Dataset.from_list(examples)
    features = raw.map(
        lambda batch: prediction_features(
            batch, tokenizer, max_length, config["doc_stride"]
        ),
        batched=True,
        remove_columns=raw.column_names,
    )
    model_features = features.remove_columns(["example_id", "offset_mapping"])
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output.parent / "prediction_cache"),
            per_device_eval_batch_size=config["eval_batch_size"],
            report_to="none",
        ),
    )
    start_logits, end_logits = trainer.predict(model_features).predictions

    features_list = features.to_list()
    spans = best_spans(
        examples,
        features_list,
        start_logits,
        end_logits,
        config["n_best_size"],
        max_answer_tokens,
    )

    # Assemble the scores cache from the WINNING window of each question
    # (multi-window questions take confidence from that window only — see
    # handoff gotcha 5).
    windows_per_qid: dict[str, int] = {}
    for feature in features_list:
        windows_per_qid[feature["example_id"]] = (
            windows_per_qid.get(feature["example_id"], 0) + 1
        )

    qids = list(spans)
    n = len(qids)
    s_cache = np.zeros((n, max_length), dtype=np.float32)
    e_cache = np.zeros((n, max_length), dtype=np.float32)
    mask_cache = np.zeros((n, max_length), dtype=bool)
    best_i = np.full(n, -1, dtype=np.int32)
    best_j = np.full(n, -1, dtype=np.int32)
    n_windows = np.zeros(n, dtype=np.int8)
    confidence: dict[str, dict] = {}

    for row, qid in enumerate(qids):
        span = spans[qid]
        feature = features_list[span["feature"]] if span["feature"] >= 0 else None
        n_windows[row] = windows_per_qid.get(qid, 0)
        if feature is None:
            confidence[qid] = {"c": 0.0}
            continue
        s_row = np.asarray(start_logits[span["feature"]], dtype=np.float32)
        e_row = np.asarray(end_logits[span["feature"]], dtype=np.float32)
        mask = np.array(
            [offset is not None for offset in feature["offset_mapping"]], dtype=bool
        )
        s_cache[row] = s_row
        e_cache[row] = e_row
        mask_cache[row] = mask
        best_i[row] = span["start"]
        best_j[row] = span["end"]
        confidence[qid] = {
            "c": span_probability(s_row, e_row, mask, span["start"], span["end"], T=1.0),
            "feature": span["feature"],
            "score": span["score"],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({qid: span["text"] for qid, span in spans.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output.parent / "scores.npz",
        qid=np.array(qids, dtype="<U24"),
        s=s_cache,
        e=e_cache,
        mask=mask_cache,
        best_i=best_i,
        best_j=best_j,
        n_windows=n_windows,
    )
    (args.output.parent / "confidence.json").write_text(
        json.dumps(confidence, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(spans)} answers to {args.output} "
        f"(+ scores.npz, confidence.json in {args.output.parent})"
    )


if __name__ == "__main__":
    main()
