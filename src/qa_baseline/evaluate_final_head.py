"""Apply a frozen scalars+span head to an exported candidate set.

    Run from the repository root:
    PYTHONPATH=src python -m qa_baseline.evaluate_final_head \
        --exports exports/v1/dev \
        --head exports/v1/heldout/final/head_final.npz \
        --output exports/v1/final/dev/predictions-span-head.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCALAR_NAMES = (
    "score", "start_logit", "end_logit", "logp_all", "logp_ctx", "rank",
    "span_tokens", "span_chars", "score_gap", "logp_ctx_gap",
    "ent_ctx_start", "ent_ctx_end", "cls_start", "cls_end",
)

WINDOW_FIELDS = ("ent_ctx_start", "ent_ctx_end", "cls_start", "cls_end")


def _column(rows, name, dtype=np.float64):
    return np.asarray([row[name] for row in rows], dtype=dtype)


def predict_export(exports_dir, head_path):
    """Return (predictions, summary) for one candidate export."""
    exports_dir = Path(exports_dir)
    questions = json.loads(
        (exports_dir / "questions.json").read_text(encoding="utf-8")
    )
    question_ids = [str(question["id"]) for question in questions]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Duplicate question IDs in questions.json")
    question_index = {qid: i for i, qid in enumerate(question_ids)}

    with (exports_dir / "candidates.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]

    span_pooled = np.load(
        exports_dir / "span_pooled.npy", mmap_mode="r", allow_pickle=False
    )
    if span_pooled.shape[0] != len(rows):
        raise ValueError(
            f"{len(rows)} candidates but {span_pooled.shape[0]} span vectors"
        )

    with np.load(exports_dir / "windows.npz", allow_pickle=False) as archive:
        windows = {name: archive[name] for name in WINDOW_FIELDS}

    row_qids = [str(row["qid"]) for row in rows]
    unknown = sorted(set(row_qids) - set(question_index))
    if unknown:
        raise ValueError(f"Candidate qid is missing from questions.json: {unknown[0]}")
    q_index = np.asarray([question_index[qid] for qid in row_qids], dtype=np.int32)

    n_questions = len(questions)
    rank = _column(rows, "rank", dtype=np.int32)
    score = _column(rows, "score")
    logp_ctx = _column(rows, "logp_ctx")
    window = _column(rows, "window", dtype=np.int32)

    if len(rows):
        if np.any(window < 0) or np.any(window >= len(windows["cls_start"])):
            raise ValueError("Candidate refers to an invalid window")

    rank1 = rank == 1
    row_counts = np.bincount(q_index, minlength=n_questions)
    rank1_counts = np.bincount(q_index[rank1], minlength=n_questions)
    has_candidates = row_counts > 0
    if np.any(rank1_counts[has_candidates] != 1):
        raise ValueError("Each question with candidates must have exactly one rank-1 row")

    best_score = np.full(n_questions, np.nan, dtype=np.float64)
    best_logp = np.full(n_questions, np.nan, dtype=np.float64)
    best_score[q_index[rank1]] = score[rank1]
    best_logp[q_index[rank1]] = logp_ctx[rank1]

    scalars = np.stack(
        [
            score,
            _column(rows, "start_logit"),
            _column(rows, "end_logit"),
            _column(rows, "logp_all"),
            logp_ctx,
            rank,
            _column(rows, "tok_end") - _column(rows, "tok_start") + 1,
            _column(rows, "char_end") - _column(rows, "char_start"),
            best_score[q_index] - score,
            best_logp[q_index] - logp_ctx,
            windows["ent_ctx_start"][window],
            windows["ent_ctx_end"][window],
            windows["cls_start"][window],
            windows["cls_end"][window],
        ],
        axis=1,
    ).astype(np.float32)

    span_features = np.asarray(span_pooled, dtype=np.float32)
    X = np.concatenate([scalars, span_features], axis=1)

    with np.load(head_path, allow_pickle=False) as archive:
        config = str(archive["config"].item())
        w = np.asarray(archive["w"], dtype=np.float32)
        mean = np.asarray(archive["mean"], dtype=np.float32)
        std = np.asarray(archive["std"], dtype=np.float32)
        alpha = float(archive["alpha"].item())

    if config != "scalars+span":
        raise ValueError(f"Expected scalars+span checkpoint, found {config!r}")
    if X.shape[1:] != w.shape or mean.shape != w.shape or std.shape != w.shape:
        raise ValueError(
            f"Feature/checkpoint shape mismatch: X={X.shape}, "
            f"w={w.shape}, mean={mean.shape}, std={std.shape}"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Invalid alpha in checkpoint: {alpha}")
    if not (
        np.isfinite(w).all()
        and np.isfinite(mean).all()
        and np.isfinite(std).all()
        and np.all(std > 0)
    ):
        raise ValueError("Checkpoint contains non-finite weights or invalid standardization")

    X = np.clip((X - mean) / std, -10, 10).astype(np.float32)
    head_score = X @ w
    mixed_score = alpha * score + (1.0 - alpha) * head_score

    # Strict > preserves the first candidate in export order when scores tie.
    best_row = np.full(n_questions, -1, dtype=np.int64)
    best_value = np.full(n_questions, -np.inf, dtype=np.float64)
    for row_index, q in enumerate(q_index):
        if mixed_score[row_index] > best_value[q]:
            best_value[q] = mixed_score[row_index]
            best_row[q] = row_index

    predictions = {qid: "" for qid in question_ids}
    for q, row_index in enumerate(best_row):
        if row_index >= 0:
            predictions[question_ids[q]] = str(rows[row_index]["text"])

    # Check that the export's own rank-1 predictions match its rank-1 rows.
    baseline_path = exports_dir / "predictions.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    rank1_text = {
        str(row["qid"]): str(row["text"])
        for row, is_rank1 in zip(rows, rank1)
        if is_rank1
    }
    mismatches = [
        qid for qid in question_ids
        if baseline.get(qid, "") != rank1_text.get(qid, "")
    ]
    if mismatches:
        raise ValueError(
            f"Export predictions disagree with rank-1 candidates "
            f"for {len(mismatches)} questions; first is {mismatches[0]}"
        )

    summary = {
        "questions": n_questions,
        "candidate_rows": len(rows),
        "questions_without_candidates": int((~has_candidates).sum()),
        "alpha": alpha,
        "rank1_export_mismatches": 0,
    }
    return predictions, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions, summary = predict_export(args.exports, args.head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(predictions, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"Questions {summary['questions']}, candidates {summary['candidate_rows']}, "
        f"without candidates {summary['questions_without_candidates']}"
    )
    print(
        f"Config scalars+span, alpha {summary['alpha']:.2f}, "
        f"rank-1 export mismatches {summary['rank1_export_mismatches']}"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()