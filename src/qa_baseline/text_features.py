"""Boundary-aware text features for every exported candidate.

The error report showed that most fixable errors are boundary errors: spans with a stray function word at an
edge ("towers and", "of 1972"), or spans that stop just before one ("early Medieval" for "early Medieval periods").
These features describe the words at and around each candidate's edges, how the candidate differs from the
baseline's rank-1 span, and whether the span's shape fits the question word.

  PYTHONPATH=src python -m qa_baseline.text_features --exports exports/v1/heldout \
      --validation runs/baseline/validation-v1.1.json --output exports/v1/heldout/text_features.npz

Row i of the output is row i of candidates.jsonl.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

WORDS = ("of", "and", "to", "in", "or", "as", "for", "by", "with", "over", "than", "about", "on", "between",
         "is", "was", "century")
WORD_SET = set(WORDS)
WH = ("who", "when", "where", "how_many", "what", "which", "why", "how")
WH_PATTERNS = (
    ("how_many", r"\bhow (?:many|much)\b"), ("who", r"\bwho(?:m|se)?\b"), ("when", r"\bwhen\b"),
    ("where", r"\bwhere\b"), ("why", r"\bwhy\b"), ("how", r"\bhow\b"), ("which", r"\bwhich\b"),
    ("what", r"\bwhat\b"),
)
RELATIONS = ("nested_in_rank1", "contains_rank1", "overlap_rank1", "lead_delta", "trail_delta", "n_nested",
             "drops_lead_W", "drops_trail_W", "adds_lead_W", "adds_trail_W")
SHAPE = ("n_words", "has_digit", "starts_capital", "frac_capitalized")
EDGE_NAMES = tuple(
    f"{position}_{word}"
    for position in ("starts", "ends", "before", "after")
    for word in WORDS
)

QUESTION_FIT_NAMES = tuple(
    f"{wh}_x_{shape}"
    for wh in WH
    for shape in ("digit", "capital")
)

NAMES = EDGE_NAMES + RELATIONS + SHAPE + QUESTION_FIT_NAMES

BINARY_NAMES = (
    EDGE_NAMES
    + (
        "nested_in_rank1", "contains_rank1", "overlap_rank1",
        "drops_lead_W", "drops_trail_W", "adds_lead_W", "adds_trail_W",
    )
    + ("has_digit", "starts_capital")
    + QUESTION_FIT_NAMES
)

INDEX = {name: i for i, name in enumerate(NAMES)}


def words_of(text):
    return re.findall(r"\w+", text.lower())


def question_word(question):
    """The earliest wh-word of the question, or 'other'."""
    q = question.lower()
    # earliest match wins; at the same position the more specific pattern (listed first) wins
    found = [(m.start(), order, name) for order, (name, pattern) in enumerate(WH_PATTERNS)
             for m in [re.search(pattern, q)] if m]
    return min(found)[2] if found else "other"


def signed_log(count):
    return float(np.sign(count) * np.log1p(abs(count)))


def question_features(group, context, question):
    """Feature matrix (len(group), len(NAMES)) for the candidates of one question; group[0] must be rank 1."""
    if group[0]["rank"] != 1:
        raise ValueError("The first candidate of a question must be rank 1")
    c1, e1 = group[0]["char_start"], group[0]["char_end"]
    wh = question_word(question)
    spans = [(row["char_start"], row["char_end"]) for row in group]
    out = np.zeros((len(group), len(NAMES)), dtype=np.float32)
    for i, row in enumerate(group):
        cs, ce, text = row["char_start"], row["char_end"], row["text"]
        words = words_of(text)
        before, after = words_of(context[max(0, cs - 40):cs]), words_of(context[ce:ce + 40])
        edges = {"starts": words[:1], "ends": words[-1:], "before": before[-1:], "after": after[:1]}
        for position, found in edges.items():
            for word in found:
                if word in WORD_SET:
                    out[i, INDEX[f"{position}_{word}"]] = 1
        same = (cs, ce) == (c1, e1)
        out[i, INDEX["nested_in_rank1"]] = cs >= c1 and ce <= e1 and not same
        out[i, INDEX["contains_rank1"]] = cs <= c1 and ce >= e1 and not same
        out[i, INDEX["overlap_rank1"]] = max(cs, c1) < min(ce, e1)
        lead = words_of(context[cs:c1]) if cs < c1 else words_of(context[c1:cs])
        trail = words_of(context[e1:ce]) if ce > e1 else words_of(context[ce:e1])
        out[i, INDEX["lead_delta"]] = signed_log(len(lead) if cs < c1 else -len(lead))
        out[i, INDEX["trail_delta"]] = signed_log(len(trail) if ce > e1 else -len(trail))
        out[i, INDEX["drops_lead_W"]] = cs > c1 and any(w in WORD_SET for w in lead)
        out[i, INDEX["adds_lead_W"]] = cs < c1 and any(w in WORD_SET for w in lead)
        out[i, INDEX["drops_trail_W"]] = ce < e1 and any(w in WORD_SET for w in trail)
        out[i, INDEX["adds_trail_W"]] = ce > e1 and any(w in WORD_SET for w in trail)
        out[i, INDEX["n_nested"]] = sum(
            1 for j, (s, e) in enumerate(spans)
            if j != i and ((cs <= s and e <= ce) or (s <= cs and ce <= e))
        )
        surface_words = re.findall(r"\w+", text)
        capitals = [token[0].isupper() for token in surface_words]
        has_digit = any(ch.isdigit() for ch in text)
        starts_capital = capitals[0] if capitals else False
        out[i, INDEX["n_words"]] = len(words)
        out[i, INDEX["has_digit"]] = has_digit
        out[i, INDEX["starts_capital"]] = starts_capital
        out[i, INDEX["frac_capitalized"]] = (
            sum(capitals) / len(capitals) if capitals else 0.0
        )
        if wh in WH:
            out[i, INDEX[f"{wh}_x_digit"]] = has_digit
            out[i, INDEX[f"{wh}_x_capital"]] = starts_capital
    return out


def load_passages(validation_path):
    document = json.loads(Path(validation_path).read_text(encoding="utf-8"))
    return {
        qa["id"]: (paragraph["context"], qa["question"])
        for article in document["data"]
        for paragraph in article["paragraphs"]
        for qa in paragraph["qas"]
    }


def build_text_features(exports_dir, validation_path):
    """Feature matrix aligned with candidates.jsonl, and the feature names."""
    passages = load_passages(validation_path)
    with (Path(exports_dir) / "candidates.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    X = np.zeros((len(rows), len(NAMES)), dtype=np.float32)
    start = 0
    while start < len(rows):
        qid, end = rows[start]["qid"], start
        while end < len(rows) and rows[end]["qid"] == qid:
            end += 1
        context, question = passages[qid]
        X[start:end] = question_features(rows[start:end], context, question)
        start = end
    return X, NAMES


def load_text_features(path, expected_rows, expected_names=NAMES):
    expected_names = tuple(expected_names)

    with np.load(path, allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float32)
        names = tuple(str(name) for name in data["names"].tolist())

    if X.ndim != 2:
        raise ValueError(f"Text features must be 2-D; got shape {X.shape}")
    if X.shape != (expected_rows, len(expected_names)):
        raise ValueError(
            f"Expected text features with shape "
            f"({expected_rows}, {len(expected_names)}); got {X.shape}"
        )
    if names != expected_names:
        raise ValueError("Saved text feature names or order do not match this code")
    if not np.isfinite(X).all():
        raise ValueError("Text feature matrix contains non-finite values")

    return X


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    X, names = build_text_features(args.exports, args.validation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, X=X, names=np.array(names))
    print(f"Wrote {args.output}: {X.shape[0]} candidates x {X.shape[1]} features")
    binary_indices = [INDEX[name] for name in BINARY_NAMES]
    shares = X[:, binary_indices].mean(axis=0)
    top = np.argsort(-shares)[:6]

    print(
        "Most common binary features:",
        ", ".join(
            f"{BINARY_NAMES[i]} {100 * shares[i]:.1f}%"
            for i in top
        ),
    )


if __name__ == "__main__":
    main()