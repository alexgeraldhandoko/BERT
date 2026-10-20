"""Fine-tune the shared BERT QA baseline on the course SQuAD 1.1 training file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .data import load_squad, split_train, write_squad_subset
from .features import training_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--data", type=Path, default=Path("data/raw/train-v1.1.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/baseline"))
    parser.add_argument(
        "--full-train", action="store_true",
        help="Train on all SQuAD training questions after experiment selection",
    )
    args = parser.parse_args()

    # Imports stay here so the pure-Python checks can run before ML packages are installed.
    from datasets import Dataset
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["doc_stride"] >= config["max_length"]:
        raise ValueError("doc_stride must be smaller than max_length")
    examples = load_squad(args.data)
    if args.full_train:
        train_examples, validation_examples = examples, []
    else:
        train_examples, validation_examples = split_train(
            examples, config["validation_fraction"], config["seed"]
        )
    args.output.mkdir(parents=True, exist_ok=True)
    if validation_examples:
        write_squad_subset(
            args.data,
            {example["id"] for example in validation_examples},
            args.output / "validation-v1.1.json",
        )
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    if config["max_length"] > tokenizer.model_max_length:
        raise ValueError("max_length exceeds the checkpoint tokenizer's limit")
    model = AutoModelForQuestionAnswering.from_pretrained(config["model_name"])

    def encode(batch):
        return training_features(batch, tokenizer, config["max_length"], config["doc_stride"])

    train_raw = Dataset.from_list(train_examples)
    train_dataset = train_raw.map(encode, batched=True, remove_columns=train_raw.column_names)
    validation_dataset = None
    if validation_examples:
        validation_raw = Dataset.from_list(validation_examples)
        validation_dataset = validation_raw.map(
            encode, batched=True, remove_columns=validation_raw.column_names
        )

    training_args = TrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        num_train_epochs=config["epochs"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        eval_strategy="epoch" if validation_dataset is not None else "no",
        save_strategy="epoch",
        load_best_model_at_end=validation_dataset is not None,
        metric_for_best_model="eval_loss" if validation_dataset is not None else None,
        greater_is_better=False if validation_dataset is not None else None,
        save_total_limit=1,
        logging_steps=100,
        report_to="none",
        seed=config["seed"],
        data_seed=config["seed"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    model_dir = args.output / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    metadata = {
        "config": config,
        "full_train": args.full_train,
        "train_file_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "train_questions": len(train_examples),
        "validation_questions": len(validation_examples),
        "train_features": len(train_dataset),
        "validation_features": len(validation_dataset) if validation_dataset is not None else 0,
        "validation_loss": trainer.evaluate()["eval_loss"] if validation_dataset is not None else None,
    }
    (args.output / "training.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved model and training record in {args.output}")


if __name__ == "__main__":
    main()

