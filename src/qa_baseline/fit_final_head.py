from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .candidate_table import load_table
from .span_head import (
    features,
    group_layout,
    listwise_loss_and_grad,
    standardize,
    standardizer,
)


def train_fixed(X, correct, groups, l2, steps, lr):
    """Train for exactly `steps` full-data updates; do not select a step on these rows."""
    w = np.zeros(X.shape[1], dtype=np.float32)
    m = np.zeros(X.shape[1])
    v = np.zeros(X.shape[1])

    for step in range(1, steps + 1):
        _, grad_scores = listwise_loss_and_grad(X @ w, correct, *groups)
        grad = X.T @ grad_scores.astype(np.float32) + l2 * w

        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        m_hat = m / (1 - 0.9**step)
        v_hat = v / (1 - 0.999**step)
        w = (w - lr * m_hat / (np.sqrt(v_hat) + 1e-8)).astype(np.float32)

    return w


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=0.03)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.55)
    args = parser.parse_args()

    table = load_table(args.table)
    rows = np.arange(len(table["em"]), dtype=np.int64)

    pooled = np.load(args.exports / "span_pooled.npy", mmap_mode="r")
    h_cls = np.load(args.exports / "windows.npz")["h_cls"]

    X_raw = features(table, rows, "scalars+span", pooled, h_cls)
    mean, std = standardizer(X_raw)
    X = standardize(X_raw, mean, std)
    del X_raw

    groups = group_layout(table["q_index"][rows])
    correct = table["em"][rows].astype(np.float64)
    w = train_fixed(X, correct, groups, args.l2, args.steps, args.lr)
    final_loss, _ = listwise_loss_and_grad(X @ w, correct, *groups)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        w=w,
        mean=mean,
        std=std,
        config=np.array("scalars+span"),
        l2=np.float32(args.l2),
        step=np.int32(args.steps),
        lr=np.float32(args.lr),
        alpha=np.float32(args.alpha),
    )

    print(f"candidate rows: {len(rows)}")
    print(f"questions represented by candidates: {len(np.unique(table['q_index'][rows]))}")
    print(f"config=scalars+span l2={args.l2} steps={args.steps} alpha={args.alpha}")
    print(f"final training loss: {final_loss:.6f}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()