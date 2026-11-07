"""Fit and temperature-calibrate a post-hoc span-conditioned confidence head.

Input is the one-example-per-question file written by the rank-1 feature
builder: confidence_examples.npz. The head is trained only on `head_train`;
`calib` is used only to fit the scalar temperature.

Run from the repository root:
  PYTHONPATH=src python fit_span_confidence.py \
      --examples exports/v1/heldout_edges/confidence_examples.npz \
      --output-dir exports/v1/final_confidence
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


def standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def transform(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.clip((x - mean) / std, -10.0, 10.0).astype(np.float32)


def binary_nll(logits: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, logits) - y * logits))


## Train a regularized logistic confidence head on head_train and fits a temperature on calib.
def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    l2: float,
    steps: int,
    lr: float,
) -> tuple[np.ndarray, float, float]:
    """Full-batch Adam for binary cross-entropy plus L2 on weights (not bias)."""
    if not 0 <= y.min() <= y.max() <= 1 or len(np.unique(y)) < 2:
        raise ValueError("Training labels must contain both 0 and 1")
    if l2 < 0 or steps < 1 or lr <= 0:
        raise ValueError("Require l2 >= 0, steps >= 1, and lr > 0")

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    n, d = x.shape
    w = np.zeros(d, dtype=np.float32)
    prevalence = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    b = float(np.log(prevalence / (1.0 - prevalence)))
    mw, vw = np.zeros(d, dtype=np.float64), np.zeros(d, dtype=np.float64)
    mb = vb = 0.0

    for step in range(1, steps + 1):
        logits = x @ w + b
        residual = (sigmoid(logits) - y).astype(np.float32)
        grad_w = (x.T @ residual) / n + l2 * w
        grad_b = float(residual.mean())

        mw = 0.9 * mw + 0.1 * grad_w
        vw = 0.999 * vw + 0.001 * grad_w.astype(np.float64) ** 2
        mb = 0.9 * mb + 0.1 * grad_b
        vb = 0.999 * vb + 0.001 * grad_b**2
        w -= lr * (mw / (1 - 0.9**step)) / (np.sqrt(vw / (1 - 0.999**step)) + 1e-8)
        b -= lr * (mb / (1 - 0.9**step)) / (np.sqrt(vb / (1 - 0.999**step)) + 1e-8)

    return w.astype(np.float32), float(b), binary_nll(x @ w + b, y)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Minimize calibration-set NLL over log(T), using golden-section search."""
    y = np.asarray(y, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    lo, hi = -5.0, 5.0  # T in approximately [0.0067, 148.4]
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = hi - ratio * (hi - lo), lo + ratio * (hi - lo)

    def objective(log_t: float) -> float:
        return binary_nll(logits / np.exp(log_t), y)

    f1, f2 = objective(x1), objective(x2)
    for _ in range(120):
        if f1 <= f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - ratio * (hi - lo)
            f1 = objective(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + ratio * (hi - lo)
            f2 = objective(x2)
    return float(np.exp((lo + hi) / 2.0))


def auc(y: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC with average ranks for tied scores; returns NaN if one class is absent."""
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


def confidence_metrics(y: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> dict:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1 - 1e-12)
    n = len(y)
    bin_id = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = bin_id == b
        if mask.any():
            ece += float(mask.mean()) * abs(float(y[mask].mean() - p[mask].mean()))
    return {
        "n": n,
        "accuracy_prevalence": float(y.mean()),
        "mean_confidence": float(p.mean()),
        "nll": binary_nll(np.log(p) - np.log1p(-p), y),
        "brier": float(np.mean((p - y) ** 2)),
        "ece_15_bins": float(ece),
        "auroc": auc(y, p),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=0.01,
                        help="fixed L2 penalty; choose in advance or using head_train only")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    with np.load(args.examples, allow_pickle=False) as data:
        x_raw = data["X"].astype(np.float32)
        y = data["y"].astype(np.int8)
        part = data["part"].astype(str)
        qids = data["qid"].astype(str)

    train = part == "head_train"
    calib = part == "calib"
    if not train.any() or not calib.any() or np.any(train & calib):
        raise ValueError("Expected disjoint, non-empty head_train and calib parts")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Labels must be binary")

    # Fit feature normalization and head weights using head_train only.
    mean, std = standardizer(x_raw[train])
    x_train = transform(x_raw[train], mean, std)
    x_calib = transform(x_raw[calib], mean, std)
    w, bias, train_nll = fit_logistic(x_train, y[train], args.l2, args.steps, args.lr)

    # Fit one scalar temperature on calibration logits; do not update the head here.
    calib_logits = x_calib @ w + bias
    temperature = fit_temperature(calib_logits, y[calib])
    before = confidence_metrics(y[calib], sigmoid(calib_logits))
    after = confidence_metrics(y[calib], sigmoid(calib_logits / temperature))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "span_confidence_head.npz"
    np.savez_compressed(
        checkpoint,
        w=w,
        bias=np.float32(bias),
        mean=mean,
        std=std,
        temperature=np.float32(temperature),
        l2=np.float32(args.l2),
        steps=np.int32(args.steps),
        lr=np.float32(args.lr),
    )
    summary = {
        "method": "post-hoc logistic confidence head on [CLS; mean-pooled rank-1 span; has_candidate]",
        "training_questions": int(train.sum()),
        "training_exact_matches": int(y[train].sum()),
        "calibration_questions": int(calib.sum()),
        "calibration_exact_matches": int(y[calib].sum()),
        "l2": args.l2,
        "steps": args.steps,
        "learning_rate": args.lr,
        "temperature": temperature,
        "training_nll": train_nll,
        "calibration_before_temperature": before,
        "calibration_after_temperature": after,
        "calibration_metrics_note": "Temperature was fit on this same calibration set; these are not an independent final estimate.",
        "checkpoint": str(checkpoint),
    }
    (args.output_dir / "span_confidence_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )

    print(f"Train: {int(train.sum())} questions, {int(y[train].sum())} exact matches")
    print(f"Calibration: {int(calib.sum())} questions, {int(y[calib].sum())} exact matches")
    print(f"Features: {x_raw.shape[1]}  fixed L2: {args.l2:g}  steps: {args.steps}")
    print(f"Fitted temperature: {temperature:.6g}")
    print("Calibration metrics before T:", json.dumps(before, sort_keys=True))
    print("Calibration metrics after T: ", json.dumps(after, sort_keys=True))
    print("Saved:", checkpoint)
    print("Saved:", args.output_dir / "span_confidence_summary.json")


if __name__ == "__main__":
    main()
