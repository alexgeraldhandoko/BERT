"""Generate one SQuAD answer per question from a fine-tuned checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_squad
from .features import prediction_features
from .postprocess import best_answers


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
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)

    raw = Dataset.from_list(examples)
    features = raw.map(
        lambda batch: prediction_features(
            batch, tokenizer, config["max_length"], config["doc_stride"]
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
    answers = best_answers(
        examples,
        features.to_list(),
        start_logits,
        end_logits,
        config["n_best_size"],
        config["max_answer_tokens"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(answers, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(answers)} answers to {args.output}")


if __name__ == "__main__":
    main()

