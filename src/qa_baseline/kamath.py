"""Post-hoc, Kamath-style random-forest confidence for a frozen QA model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from .data import load_squad, split_train, write_squad_subset
from .postprocess import candidate_spans


FEATURE_NAMES = [
    "passage_words", "predicted_answer_words",
    "top1_span_probability", "top2_span_probability", "top3_span_probability",
    "top4_span_probability", "top5_span_probability",
]
ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_answer(text: str) -> str:
    """Same normalization as vendor/evaluate-v2.0.py for SQuAD exact match."""
    import re
    import string

    text = "".join(char for char in text.lower() if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _softmax(row) -> list[float]:
    maximum = max(row)
    weights = [math.exp(float(value) - float(maximum)) for value in row]
    total = sum(weights)
    return [weight / total for weight in weights]


def build_records(examples, features, start_logits, end_logits, config) -> list[dict]:
    """One feature vector and correctness label per final question answer."""
    starts = [_softmax(row) for row in start_logits]
    ends = [_softmax(row) for row in end_logits]
    best = {example["id"]: None for example in examples}
    span_probs = {example["id"]: {} for example in examples}
    for span in candidate_spans(
        examples, features, start_logits, end_logits,
        config["n_best_size"], config["max_answer_tokens"],
    ):
        qid = span.question_id
        probability = starts[span.feature_index][span.start_index] * ends[span.feature_index][span.end_index]
        # Overlapping windows can contain the same character span. Count it once.
        key = (span.char_start, span.char_end)
        top = span_probs[qid]
        if key in top:
            top[key] = max(probability, top[key])
        elif len(top) < 5:
            top[key] = probability
        else:
            weakest = min(top, key=top.get)
            if probability > top[weakest]:
                del top[weakest]
                top[key] = probability
        if best[qid] is None or span.score > best[qid][0].score:
            best[qid] = (span, probability)

    records = []
    for example in examples:
        qid = example["id"]
        selected = best[qid]
        answer = selected[0].text if selected else ""
        probabilities = sorted(span_probs[qid].values(), reverse=True)[:5]
        probabilities += [0.0] * (5 - len(probabilities))
        gold = [a["text"] for a in example["answers"]]
        normalized_gold = [_normalize_answer(text) for text in gold if _normalize_answer(text)] or [""]
        records.append({
            "id": qid,
            "answer": answer,
            "correct": int(_normalize_answer(answer) in normalized_gold),
            "raw_confidence": selected[1] if selected else 0.0,
            "features": [len(example["context"].split()), len(answer.split()), *probabilities],
        })
    return records


def expected_calibration_error(labels, probabilities, bins: int = 10) -> float:
    if not labels or len(labels) != len(probabilities) or bins < 1:
        raise ValueError("Expected matching nonempty labels/probabilities and positive bins")
    counts = [0] * bins
    correct = [0.0] * bins
    confidence = [0.0] * bins
    for label, probability in zip(labels, probabilities):
        if not 0 <= probability <= 1 or label not in (0, 1):
            raise ValueError("Labels must be binary and probabilities in [0, 1]")
        index = min(int(probability * bins), bins - 1)
        counts[index] += 1
        correct[index] += label
        confidence[index] += probability
    return sum(abs(correct[i] - confidence[i]) for i in range(bins)) / len(labels)


def _read_records(path: Path, role: str) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("role") != role or document.get("feature_names") != FEATURE_NAMES:
        raise ValueError(f"Expected {role} records with the current feature ordering")
    records = document.get("records", [])
    if not records or len({row["id"] for row in records}) != len(records):
        raise ValueError("Expected nonempty records with unique question IDs")
    if any(len(row["features"]) != len(FEATURE_NAMES) for row in records):
        raise ValueError("Invalid feature vector length")
    return document


def prepare(args) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    examples = load_squad(args.data)
    _, held_out = split_train(examples, config["validation_fraction"], config["seed"])
    write_squad_subset(args.data, {item["id"] for item in held_out}, args.output)
    print(f"Wrote {len(held_out)} held-out questions to {args.output}")
    print("Check that this is the split excluded from the existing BERT checkpoint.")


def collect(args) -> None:
    from .predict import run_inference

    config = json.loads(args.config.read_text(encoding="utf-8"))
    examples = load_squad(args.data)
    features, start_logits, end_logits = run_inference(
        examples, config, args.model, args.output.parent / "prediction_cache"
    )
    model_weights = args.model / "model.safetensors"
    if not model_weights.is_file():
        raise FileNotFoundError(f"Expected {model_weights}")
    records = build_records(examples, features, start_logits, end_logits, config)
    document = {
        "role": args.role,
        "data_sha256": _sha256(args.data),
        "model_sha256": _sha256(model_weights),
        "config": config,
        "feature_names": FEATURE_NAMES,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} question records to {args.output}")


def fit(args) -> None:
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    document = _read_records(args.records, "calibration")
    rows = document["records"]
    labels = [row["correct"] for row in rows]
    if set(labels) != {0, 1}:
        raise ValueError("Forest fitting needs both correct and incorrect answers")
    forest = RandomForestClassifier(
        n_estimators=args.trees, min_samples_leaf=args.min_leaf,
        random_state=args.seed, n_jobs=args.jobs,
    )
    forest.fit([row["features"] for row in rows], labels)
    artifact = {
        "forest": forest,
        "feature_names": FEATURE_NAMES,
        "model_sha256": document["model_sha256"],
        "calibration_data_sha256": document["data_sha256"],
        "training_questions": len(rows),
        "seed": args.seed,
        "trees": args.trees,
        "min_leaf": args.min_leaf,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    print(f"Fitted forest on {len(rows)} questions; saved {args.output}")


def evaluate(args) -> None:
    import joblib
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    document = _read_records(args.records, "evaluation")
    artifact = joblib.load(args.forest)  # Only load a forest file produced by this workflow.
    if artifact["feature_names"] != FEATURE_NAMES:
        raise ValueError("Forest feature ordering does not match")
    if artifact["model_sha256"] != document["model_sha256"]:
        raise ValueError("Calibration and evaluation used different BERT checkpoints")
    if artifact["calibration_data_sha256"] == document["data_sha256"]:
        raise ValueError("Evaluation data is the same file used to train the forest")
    if _sha256(args.data) != document["data_sha256"]:
        raise ValueError("Evaluation SQuAD data does not match the extracted records")
    rows = document["records"]
    probabilities = artifact["forest"].predict_proba([row["features"] for row in rows])[:, 1].tolist()
    raw = [row["raw_confidence"] for row in rows]
    labels = [row["correct"] for row in rows]
    predictions = {row["id"]: row["answer"] for row in rows}
    args.output.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output / "predictions.json"
    prediction_path.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    qa_path = args.output / "qa-metrics.json"
    subprocess.run([
        sys.executable, str(ROOT / "vendor" / "evaluate-v2.0.py"),
        str(args.data), str(prediction_path), "--out-file", str(qa_path),
    ], check=True)
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    if qa["total"] != len(rows):
        raise ValueError("Official scorer did not evaluate every question")
    if not math.isclose(sum(labels) / len(labels), qa["exact"] / 100, abs_tol=1e-10):
        raise ValueError("Confidence correctness labels disagree with official exact match")
    scores = {
        "id": [row["id"] for row in rows],
        "correct": labels,
        "raw_softmax": raw,
        "kamath_forest": probabilities,
    }
    (args.output / "confidence.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")

    def metrics(values):
        return {
            "ece_10_bins": expected_calibration_error(labels, values),
            "brier": brier_score_loss(labels, values),
            "nll": log_loss(labels, values, labels=[0, 1]),
            "roc_auc": roc_auc_score(labels, values) if len(set(labels)) == 2 else None,
            "mean_confidence": sum(values) / len(values),
        }

    summary = {
        "questions": len(rows),
        "qa": qa,
        "accuracy_for_confidence": sum(labels) / len(labels),
        "raw_softmax": metrics(raw),
        "kamath_forest": metrics(probabilities),
        "forest": {key: artifact[key] for key in (
            "training_questions", "seed", "trees", "min_leaf",
            "model_sha256", "calibration_data_sha256",
        )},
        "evaluation_data_sha256": document["data_sha256"],
        "feature_names": FEATURE_NAMES,
        "feature_definition": "Whitespace word lengths; top five distinct character spans by per-window P(start)*P(end).",
        "ece_definition": "10 equal-width confidence bins; right endpoint included in final bin.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("prepare-validation")
    split.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    split.add_argument("--data", type=Path, default=Path("data/raw/train-v1.1.json"))
    split.add_argument("--output", type=Path, default=Path("runs/kamath/validation-v1.1.json"))
    split.set_defaults(func=prepare)

    extraction = commands.add_parser("collect")
    extraction.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    extraction.add_argument("--data", type=Path, required=True)
    extraction.add_argument("--model", type=Path, required=True)
    extraction.add_argument("--role", choices=["calibration", "evaluation"], required=True)
    extraction.add_argument("--output", type=Path, required=True)
    extraction.set_defaults(func=collect)

    training = commands.add_parser("fit")
    training.add_argument("--records", type=Path, required=True)
    training.add_argument("--output", type=Path, default=Path("runs/kamath/forest.joblib"))
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--trees", type=int, default=200)
    training.add_argument("--min-leaf", type=int, default=5)
    training.add_argument("--jobs", type=int, default=1)
    training.set_defaults(func=fit)

    scoring = commands.add_parser("evaluate")
    scoring.add_argument("--records", type=Path, required=True)
    scoring.add_argument("--data", type=Path, required=True)
    scoring.add_argument("--forest", type=Path, default=Path("runs/kamath/forest.joblib"))
    scoring.add_argument("--output", type=Path, default=Path("runs/kamath/evaluation"))
    scoring.add_argument("--summary", type=Path, default=Path("results/kamath-summary.json"))
    scoring.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
