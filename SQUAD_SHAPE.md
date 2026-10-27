# SQuAD 1.1 — first-pass shape reference

Source: `data/raw/{train,dev}-v1.1.json` (gitignored). Regenerate with
`python3 eda_squad.py` and `python3 eda_squad2.py`.

## Container schema

```
{ "version": "1.1",
  "data": [                        # articles      train 442 / dev 48
    { "title": str,               # Wikipedia page title, article-level split key
      "paragraphs": [              # train 18,896 / dev 2,067
        { "context": str,          # the evidence passage (1 paragraph)
          "qas": [                 # train 87,599 / dev 10,570
            { "id": str,           # 24-hex, stable; prediction key
              "question": str,
              "answers": [ {"text": str, "answer_start": int} ]   # char offset
            } ] } ] } ] }
```

Every level is a list; only `answer_start` + `text` are scored.

## Size and counts

| | train | dev |
|---|---|---|
| articles (titles) | 442 | 48 (disjoint from train) |
| paragraphs / contexts | 18,896 (18,891 unique) | 2,067 (all unique) |
| questions | 87,599 | 10,570 |
| questions/paragraph | 4.64 | 5.11 |
| paragraphs/article | min 5 / med 36 / max 149 | min 21 / med 43 / max 98 |
| **answers per question** | **exactly 1 (100%)** | 3 (80%), 2 (1%), 4–6 (18%) |

The asymmetry is the single most important quirk: **train gives one gold span,
dev gives ~3 annotators' spans.** 53% of dev questions have >1 *distinct* answer
text (5,625/10,570), so dev EM rewards hitting any plausible annotator span.

## Length distributions (whitespace tokens)

| field | p25 | med | p75 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|---|
| context | 87 | 107 | 139 | 180 | 279 | 653 | 117 |
| question | 8 | 10 | 12 | 15 | 21 | 40 | 10 |
| answer | 1 | 2 | 3 | 7 | 18 | 43 | 3.2 |

Context chars: med 678, max 3,706. `answer_start` char offset: med 262, max 3,126.

Relative answer position inside context is nearly uniform:
p10 = 0.05, med = 0.40, p90 = 0.87. No position prior worth exploiting.

## Answer surface form (train / dev)

- multi-word span: 65.4% / 67.6%
- single word: 26.9% / 24.9%
- pure number: 7.7% / 7.5%
- contains a digit: 21.9% / 19.7%
- starts uppercase: 41.0% / 38.7%
- ends in punctuation: 3.8% / 3.3%
- leading/trailing whitespace in gold text: **0** (clean, char-exact spans)
- gold span is an exact substring of context: **100%** (by construction)

## Question mix (wh-word among first 3 tokens)

what 48.7%, other 14.7%, how 9.9%, who 9.4%, when 6.3%, which 5.7%,
where 3.9%, why 1.4%. "other" is mostly preposition-fronted ("In what year…").

## Windowing (approx WordPiece, `max_length=384`, `doc_stride=128`)

- 2.0% of train / 3.0% of dev questions need >1 window
- mean 1.02 / 1.05 windows; p95 = 1; max 6
- So ~98% of examples are single-window — the `doc_stride` machinery barely fires.
  Only 3% of dev benefits from cross-window aggregation at 384.
- Longest context is 653 whitespace tokens (~820 WordPiece) → still ≤6 windows.

## Facts that constrain ideation

1. **No unanswerable questions** (SQuAD 1.1). Every question has a span, so a
   model that always predicts something is never punished for answering. This
   is what makes `null_odds`/no-answer heads pure loss in v1.1.
2. **CLS-labelled windows.** `features.py` labels overflow windows missing the
   gold span as `(CLS, CLS)`. With only 2% multi-window questions, that's a tiny
   no-answer signal — effectively a pure span task.
3. **Span length is capped** at `max_answer_tokens=128`; p99 answer is 18 tokens,
   so the cap is slack. Gold answers must fit wholly inside one window.
4. **Answer spans are clean**: no whitespace padding, always exact substrings,
   so char→token offset mapping has no workaround-worthy edge cases. Only
   3.3–3.8% of golds end in punctuation (period-stripping is marginal).
5. **Article-level splitting** (`split_train` hashes titles) is essential —
   paragraphs from one Wikipedia page share facts, and train/dev titles are
   fully disjoint (only 8 question strings coincide by chance out of 97k).
6. **Dev is small** (10,570 q): EM/F1 noise is roughly ±0.7–1.0 points, so
   compare variants on the internal held-out split first.
7. **Headroom is in surface form**, not retrieval: contexts are short and
   single-paragraph, so the hard part is span boundary selection (2-token
   median answers) and handling of long/degenerate spans — not evidence finding.
