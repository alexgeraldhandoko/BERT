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


## Trains regularized logidtic confidence head on head_train and fit temparature on calib.

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