"""Break down the baseline's fixable errors by how the wrong rank-1 span differs from the correct candidate.

A fixable error is a question whose rank-1 answer is not an exact match but whose top-K candidates
contain one. Spans are compared after the official normalization (lowercase, no punctuation or articles).

  PYTHONPATH=src python -m qa_baseline.boundary_report --exports exports/v1/heldout
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .export_candidates import load_official_scorer


def find(big, small):
    """Start index of the token list `small` inside `big`, or -1."""
    n = len(small)
    if n == 0:
        return -1
    for i in range(len(big) - n + 1):
        if big[i : i + n] == small:
            return i
    return -1


def classify(wrong, right):
    """How the wrong span (token list) differs from the correct one.

    Returns (category, extra leading words, extra trailing words); for 'too short' the words the wrong span misses."""
    if not wrong or not right:
        return "empty answer", [], []
    i = find(wrong, right)
    if i >= 0:
        lead, trail = wrong[:i], wrong[i + len(right):]
        if lead and trail:
            return "too long: extra words on both sides", lead, trail
        return ("too long: extra leading words", lead, trail) if lead else ("too long: extra trailing words", lead, trail)
    j = find(right, wrong)
    if j >= 0:
        return "too short: missing words", right[:j], right[j + len(wrong):]
    if set(wrong) & set(right):
        return "shifted: partial overlap", [], []
    return "different answer: no overlap", [], []


def fixable_errors(rows, top_k=5):
    """rows: candidate records in file order. Yields (qid, wrong text, right text, right rank)."""
    by_question = {}
    for row in rows:
        if row["rank"] <= top_k:
            by_question.setdefault(row["qid"], []).append(row)
    for qid, candidates in by_question.items():
        first = candidates[0]
        if first["rank"] != 1 or first["em"] == 1:
            continue
        right = next((c for c in candidates if c["em"] == 1), None)
        if right is not None:
            yield qid, first["text"], right["text"], right["rank"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, default=Path("vendor/evaluate-v2.0.py"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()

    scorer = load_official_scorer(args.scorer)
    with (args.exports / "candidates.jsonl").open(encoding="utf-8") as stream:
        rows = [row for row in map(json.loads, stream) if row["rank"] <= args.top_k]
    cases = list(fixable_errors(rows, args.top_k))
    categories = Counter()
    extra_leading, extra_trailing = Counter(), Counter()
    missing_leading, missing_trailing = Counter(), Counter()
    shown = []

    for qid, wrong, right, rank in cases:
        wrong_tokens = scorer.normalize_answer(wrong).split()
        right_tokens = scorer.normalize_answer(right).split()
        category, lead, trail = classify(wrong_tokens, right_tokens)

        categories[category] += 1

        if category.startswith("too long:"):
            extra_leading.update(lead)
            extra_trailing.update(trail)
        elif category == "too short: missing words":
            missing_leading.update(lead)
            missing_trailing.update(trail)

        shown.append((category, wrong, right, rank))

    print(f"Fixable errors (rank 1 wrong, a correct answer within the top {args.top_k}): {len(cases)}")
    for category, count in categories.most_common():
        print(f"  {category:40s} {count:5d}  {100 * count / len(cases):5.1f}%")

    print("\nExtra words in wrong spans that are too long:")
    print("  leading :", extra_leading.most_common(12))
    print("  trailing:", extra_trailing.most_common(12))

    print("\nWords missing from wrong spans that are too short:")
    print("  leading :", missing_leading.most_common(12))
    print("  trailing:", missing_trailing.most_common(12))

    print("\nExamples (wrong rank-1 | correct candidate and its rank):")
    for category, wrong, right, rank in random.Random(0).sample(
        shown, min(args.examples, len(shown))
    ):
        print(f"  [{category}] {wrong!r} | {right!r} (rank {rank})")

if __name__ == "__main__":
    main()