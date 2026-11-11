"""Apply a frozen span-conditioned confidence head to exported dev rank-1 answers.

Run from the repository root:
  PYTHONPATH=src python qa_baseline.score_span_confidence.py \
      --exports exports/v1/dev_edges \
      --head exports/v1/final_confidence/span_confidence_head.npz \
      --output-dir exports/v1/final_confidence/dev
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def auc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives, negatives = int(y.sum()), int((1 - y).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def metrics(y: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> dict:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1 - 1e-12)
    bin_id = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = bin_id == b
        if mask.any():
            ece += float(mask.mean()) * abs(float(y[mask].mean() - p[mask].mean()))
    return {
        "n": int(len(y)),
        "accuracy_prevalence": float(y.mean()),
        "mean_confidence": float(p.mean()),
        "nll": float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p))),
        "brier": float(np.mean((p - y) ** 2)),
        "ece_15_bins": float(ece),
        "auroc": auc(y, p),
    }


def dev_examples(exports_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return X=[CLS; pooled span; has_candidate], EM labels, qids and endpoint products."""
    questions = json.loads((exports_dir / "questions.json").read_text(encoding="utf-8"))
    qids = [str(q["id"]) for q in questions]
    q_index = {qid: i for i, qid in enumerate(qids)}
    pooled = np.load(exports_dir / "span_pooled.npy", mmap_mode="r")
    h_cls = np.load(exports_dir / "windows.npz")["h_cls"]

    # Same layout as the training examples: 768 CLS + 768 pooled + presence flag.
    X = np.zeros((len(qids), 1537), dtype=np.float32)
    y = np.zeros(len(qids), dtype=np.int8)
    has_candidate = np.zeros(len(qids), dtype=np.int8)
    endpoint_product = np.zeros(len(qids), dtype=np.float64)
    seen = set()
    candidate_rows = 0

    with (exports_dir / "candidates.jsonl").open(encoding="utf-8") as stream:
        for row_index, line in enumerate(stream):
            row = json.loads(line)
            candidate_rows += 1
            if row["rank"] != 1:
                continue
            qid = str(row["qid"])
            if qid not in q_index:
                raise ValueError(f"Candidate question {qid} is absent from questions.json")
            if qid in seen:
                raise ValueError(f"Multiple rank-1 rows for question {qid}")
            i = q_index[qid]
            window = int(row["window"])
            if not 0 <= window < len(h_cls) or not 0 <= row_index < pooled.shape[0]:
                raise ValueError(f"Invalid vector row/window for question {qid}")
            X[i, :768] = h_cls[window]
            X[i, 768:1536] = pooled[row_index]
            X[i, 1536] = 1.0
            y[i] = int(row["em"])
            has_candidate[i] = 1
            endpoint_product[i] = float(np.exp(np.clip(row["logp_all"], -745.0, 0.0)))
            if y[i] != int(questions[i]["top1_em"]):
                raise ValueError(f"Candidate/question label mismatch for {qid}")
            seen.add(qid)

    for i, question in enumerate(questions):
        if not has_candidate[i]:
            # This is the export's score for the empty fallback prediction.
            y[i] = int(question["top1_em"])

    if pooled.shape[0] != candidate_rows:
        raise ValueError(f"{candidate_rows} candidate rows but {pooled.shape[0]} pooled vectors")
    if len(seen) + int((has_candidate == 0).sum()) != len(qids):
        raise ValueError("Unexpected missing rank-1 rows")
    top1 = {
        "questions": len(questions),
        "candidates": candidate_rows,
        "rank1_candidates": len(seen),
        "rank1_em": 100 * float(np.mean([q["top1_em"] for q in questions])),
        "rank1_f1": 100 * float(np.mean([q["top1_f1"] for q in questions])),
    }
    return X, y, np.asarray(qids), endpoint_product, top1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    X, y, qids, endpoint_product, top1 = dev_examples(args.exports)
    model = np.load(args.head, allow_pickle=False)
    w = model["w"].astype(np.float32)
    mean = model["mean"].astype(np.float32)
    std = model["std"].astype(np.float32)
    bias = float(model["bias"])
    temperature = float(model["temperature"])
    if w.shape != (1537,) or mean.shape != w.shape or std.shape != w.shape:
        raise ValueError(f"Unexpected model feature shapes: w={w.shape}, mean={mean.shape}, std={std.shape}")
    if not np.isfinite(w).all() or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Model contains non-finite weights or normalization values")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"Invalid temperature {temperature}")

    X = np.clip((X - mean) / std, -10.0, 10.0).astype(np.float32)
    logits = X @ w + bias
    raw_head = sigmoid(logits)
    calibrated_head = sigmoid(logits / temperature)

    report = {
        "dev_rank1": top1,
        "confidence": {
            "raw_endpoint_probability_product": metrics(y, endpoint_product),
            "span_head_before_temperature": metrics(y, raw_head),
            "span_head_after_temperature": metrics(y, calibrated_head),
        },
        "head_temperature": temperature,
        "note": (
            "The temperature and head were frozen before this dev scoring. "
            "The endpoint probability product is from exported per-window logp_all; "
            "it is not temperature-scaled here."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dev_confidence_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    predictions = {
        str(qid): {
            "exact_match": int(label),
            "raw_endpoint_probability_product": float(base),
            "span_head_before_temperature": float(raw),
            "span_head_after_temperature": float(calibrated),
        }
        for qid, label, base, raw, calibrated in zip(
            qids, y, endpoint_product, raw_head, calibrated_head
        )
    }
    (args.output_dir / "dev_confidence_predictions.json").write_text(
        json.dumps(predictions, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Dev questions: {top1['questions']}; rank-1 EM {top1['rank1_em']:.2f} F1 {top1['rank1_f1']:.2f}")
    print(f"Frozen head temperature: {temperature:.6g}")
    for name, values in report["confidence"].items():
        print(f"{name}: " + json.dumps(values, sort_keys=True))
    print("Saved:", args.output_dir / "dev_confidence_summary.json")
    print("Saved:", args.output_dir / "dev_confidence_predictions.json")


if __name__ == "__main__":
    main()
