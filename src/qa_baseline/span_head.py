"""Listwise linear head that rescored the top-K candidates of each question (numpy only).

Trains on the head_train part, picks the L2 strength and stopping step on the calib part, and
reports how often the head's top pick is an exact match. The test part is never touched here.

  PYTHONPATH=src python -m qa_baseline.span_head \
      --exports exports/v1/heldout --table exports/v1/heldout/table.npz --output-dir exports/v1/heldout/heads
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .candidate_table import PARTS, load_table

CONFIGS = ("scalars", "scalars+span")

def group_layout(q_index):
    """Start offsets and sizes of each question's rows (rows of one question must be contiguous)."""
    q = np.asarray(q_index)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    if len(np.unique(q)) != len(starts):
        raise ValueError("Rows of one question are not contiguous")
    return starts, np.diff(np.r_[starts, len(q)])


def listwise_loss_and_grad(scores, correct, starts, sizes):
    """-log(probability mass on correct candidates), averaged over questions that have one.

    Several correct rows per question are fine: any of them counts. Returns (loss, d loss / d scores)."""
    scores = np.asarray(scores, dtype=np.float64)
    shift = np.repeat(np.maximum.reduceat(scores, starts), sizes)
    e = np.exp(scores - shift)
    total = np.add.reduceat(e, starts)
    good = np.add.reduceat(e * correct, starts)
    has = good > 0
    prob = e / np.repeat(total, sizes)
    target = e * correct / np.repeat(np.where(has, good, 1.0), sizes)
    n = max(int(has.sum()), 1)
    loss = float((np.log(total[has]) - np.log(good[has])).sum() / n)
    grad = (prob - target) * np.repeat(has, sizes) / n
    return loss, grad


def top1_metrics(scores, em, f1, starts, sizes, n_questions):
    """EM and F1 (percent) of each question's best-scoring candidate; ties pick the earlier row.

    n_questions counts every question in the part, including those with no candidates (scored 0)."""
    chosen = np.array([s + int(np.argmax(scores[s : s + n])) for s, n in zip(starts, sizes)])
    return 100 * float(em[chosen].sum()) / n_questions, 100 * float(f1[chosen].sum()) / n_questions


def standardizer(X):
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(X, mean, std):
    return np.clip((X - mean) / std, -10, 10).astype(np.float32)


def train_listwise(X, correct, groups, X_val, correct_val, groups_val, l2, steps, lr, eval_every=25):
    """Adam on the listwise loss. Keeps the weights with the lowest validation loss."""
    d = X.shape[1]
    w = np.zeros(d, dtype=np.float32)  # all-zero scores tie, so the start point is the baseline's rank-1 pick
    m, v = np.zeros(d), np.zeros(d)
    best = (np.inf, w.copy(), 0)
    for step in range(1, steps + 1):
        _, g = listwise_loss_and_grad(X @ w, correct, *groups)
        grad = X.T @ g.astype(np.float32) + l2 * w
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        w = (w - lr * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)).astype(np.float32)
        if step % eval_every == 0 or step == steps:
            val_loss, _ = listwise_loss_and_grad(X_val @ w, correct_val, *groups_val)
            if val_loss < best[0]:
                best = (val_loss, w.copy(), step)
    return best[1], best[0], best[2]


def features(table, rows, config, pooled, h_cls, edges=None):
    """Scalars plus optional text, span, CLS, and boundary-vector features."""
    parts = config.split("+")
    unknown = set(parts) - {"scalars", "text", "span", "cls", "edges"}
    if parts[0] != "scalars" or unknown:
        raise ValueError(f"Unknown feature config {config!r}")

    blocks = [table["scalars"][rows]]

    if "text" in parts:
        blocks.append(np.asarray(table["text"][rows], dtype=np.float32))
    if "span" in parts:
        blocks.append(np.asarray(pooled[rows], dtype=np.float32))
    if "cls" in parts:
        blocks.append(np.asarray(h_cls[table["window"][rows]], dtype=np.float32))
    if "edges" in parts:
        if edges is None:
            raise ValueError("This feature config requires span_edges.npy")
        edge_rows = np.asarray(edges[rows], dtype=np.float32)
        if edge_rows.ndim != 3 or edge_rows.shape[1] != 2:
            raise ValueError(f"Expected boundary vectors shaped (rows, 2, hidden), got {edge_rows.shape}")
        blocks.append(edge_rows.reshape(len(rows), edge_rows.shape[1] * edge_rows.shape[2]))

    return np.concatenate(blocks, axis=1)


def part_layout(table, number):
    rows = np.where(table["part"] == number)[0]
    return rows, group_layout(table["q_index"][rows]), int((table["question_part"] == number).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--l2", default="0.001,0.01,0.1")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.02)
    args = parser.parse_args()

    table = load_table(args.table)
    pooled = np.load(args.exports / "span_pooled.npy", mmap_mode="r")
    h_cls = np.load(args.exports / "windows.npz")["h_cls"]
    train_rows, train_groups, _ = part_layout(table, PARTS.index("head_train"))
    calib_rows, calib_groups, calib_questions = part_layout(table, PARTS.index("calib"))
    correct_train = table["em"][train_rows].astype(np.float64)
    correct_calib = table["em"][calib_rows].astype(np.float64)

    base_em, base_f1 = top1_metrics(
        -table["rank"][calib_rows].astype(np.float64), table["em"][calib_rows], table["f1"][calib_rows],
        *calib_groups, calib_questions,
    )
    print(f"baseline (rank 1) on calib: EM {base_em:.2f}  F1 {base_f1:.2f}")

    for config in args.configs.split(","):
        X_train = features(table, train_rows, config, pooled, h_cls)
        mean, std = standardizer(X_train)
        X_train = standardize(X_train, mean, std)
        X_calib = standardize(features(table, calib_rows, config, pooled, h_cls), mean, std)
        best = None
        for l2 in (float(value) for value in args.l2.split(",")):
            w, loss, step = train_listwise(
                X_train, correct_train, train_groups, X_calib, correct_calib, calib_groups, l2, args.steps, args.lr
            )
            if best is None or loss < best[1]:
                best = (w, loss, step, l2)
        w, loss, step, l2 = best
        em, f1 = top1_metrics(X_calib @ w, table["em"][calib_rows], table["f1"][calib_rows], *calib_groups, calib_questions)
        print(
            f"{config:18s} l2={l2:g} step={step:3d} calib loss {loss:.4f}  "
            f"head-alone EM {em:.2f} ({em - base_em:+.2f})  F1 {f1:.2f} ({f1 - base_f1:+.2f})"
        )
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            np.savez(args.output_dir / f"head_{config}.npz", w=w, mean=mean, std=std, l2=l2, step=step)


if __name__ == "__main__":
    main()