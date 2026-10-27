"""Choose the best valid answer span across all windows of each question.

`best_spans` performs the search and keeps the full winning record (score,
winning feature, token offsets); `best_answers` preserves the historical
{qid: str} contract on top of it.
"""

from __future__ import annotations


def best_spans(
    examples: list[dict],
    features: list[dict],
    start_logits,
    end_logits,
    n_best_size: int,
    max_answer_tokens: int,
) -> dict[str, dict]:
    """Return {qid: {"text", "feature", "start", "end", "score"}} for the single
    best legal span per question, chosen across all windows.

    "feature" is the index into `features` of the window that produced the
    winner; "start"/"end" are token positions within that window's logit rows.
    Questions with no legal span map to score -inf, empty text, feature -1.
    """
    if len(features) != len(start_logits) or len(features) != len(end_logits):
        raise ValueError("Feature and logit counts do not match")
    if n_best_size < 1 or max_answer_tokens < 1:
        raise ValueError("Candidate and answer lengths must be positive")

    contexts = {example["id"]: example["context"] for example in examples}
    best = {
        example["id"]: {"text": "", "feature": -1, "start": -1, "end": -1, "score": float("-inf")}
        for example in examples
    }
    for feature_index, (feature, start_row, end_row) in enumerate(zip(features, start_logits, end_logits)):
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
                if score > best[qid]["score"]:
                    best[qid] = {
                        "text": contexts[qid][char_start:char_end],
                        "feature": feature_index,
                        "start": start,
                        "end": end,
                        "score": score,
                    }
    return best


def best_answers(
    examples: list[dict],
    features: list[dict],
    start_logits,
    end_logits,
    n_best_size: int,
    max_answer_tokens: int,
) -> dict[str, str]:
    spans = best_spans(
        examples, features, start_logits, end_logits, n_best_size, max_answer_tokens
    )
    return {qid: span["text"] for qid, span in spans.items()}
