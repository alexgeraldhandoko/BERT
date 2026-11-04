"""Choose the best valid answer span across all windows of each question."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class SpanCandidate:
    question_id: str
    score: float
    text: str
    feature_index: int
    start_index: int
    end_index: int
    char_start: int
    char_end: int


def candidate_spans(
    examples: list[dict], features: list[dict], start_logits, end_logits,
    n_best_size: int, max_answer_tokens: int,
) -> Iterator[SpanCandidate]:
    """Yield exactly the valid spans considered by the baseline answer selector."""
    if len(features) != len(start_logits) or len(features) != len(end_logits):
        raise ValueError("Feature and logit counts do not match")
    if n_best_size < 1 or max_answer_tokens < 1:
        raise ValueError("Candidate and answer lengths must be positive")

    contexts = {example["id"]: example["context"] for example in examples}
    for feature_index, (feature, start_row, end_row) in enumerate(
        zip(features, start_logits, end_logits)
    ):
        qid = feature["example_id"]
        offsets = feature["offset_mapping"]
        if qid not in contexts or len(offsets) != len(start_row) or len(offsets) != len(end_row):
            raise ValueError(f"Invalid feature for question {qid}")
        top_starts = sorted(range(len(start_row)), key=lambda i: start_row[i], reverse=True)[:n_best_size]
        top_ends = sorted(range(len(end_row)), key=lambda i: end_row[i], reverse=True)[:n_best_size]
        for start in top_starts:
            for end in top_ends:
                if (
                    offsets[start] is None
                    or offsets[end] is None
                    or end < start
                    or end - start + 1 > max_answer_tokens
                ):
                    continue
                char_start, char_end = offsets[start][0], offsets[end][1]
                if not 0 <= char_start < char_end <= len(contexts[qid]):
                    continue
                yield SpanCandidate(
                    qid, float(start_row[start]) + float(end_row[end]),
                    contexts[qid][char_start:char_end], feature_index, start, end,
                    char_start, char_end,
                )


def best_answers(
    examples: list[dict],
    features: list[dict],
    start_logits,
    end_logits,
    n_best_size: int,
    max_answer_tokens: int,
) -> dict[str, str]:
    best = {example["id"]: (float("-inf"), "") for example in examples}
    for candidate in candidate_spans(
        examples, features, start_logits, end_logits, n_best_size, max_answer_tokens
    ):
        if candidate.score > best[candidate.question_id][0]:
            best[candidate.question_id] = (candidate.score, candidate.text)
    return {qid: value[1] for qid, value in best.items()}

