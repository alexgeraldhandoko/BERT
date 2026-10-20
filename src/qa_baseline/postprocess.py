"""Choose the best valid answer span across all windows of each question."""

from __future__ import annotations


def best_answers(
    examples: list[dict],
    features: list[dict],
    start_logits,
    end_logits,
    n_best_size: int,
    max_answer_tokens: int,
) -> dict[str, str]:
    if len(features) != len(start_logits) or len(features) != len(end_logits):
        raise ValueError("Feature and logit counts do not match")
    if n_best_size < 1 or max_answer_tokens < 1:
        raise ValueError("Candidate and answer lengths must be positive")

    contexts = {example["id"]: example["context"] for example in examples}
    best = {example["id"]: (float("-inf"), "") for example in examples}
    for feature, start_row, end_row in zip(features, start_logits, end_logits):
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
                score = float(start_row[start]) + float(end_row[end])
                if score > best[qid][0]:
                    best[qid] = (score, contexts[qid][char_start:char_end])
    return {qid: value[1] for qid, value in best.items()}

