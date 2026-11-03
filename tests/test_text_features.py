import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from qa_baseline.span_head import features
from qa_baseline.text_features import INDEX, NAMES, build_text_features, question_features, question_word

CONTEXT = "The company was founded in 1912 and moved to Paris."


def span(text, rank):
    start = CONTEXT.find(text)
    return {"qid": "q", "rank": rank, "char_start": start, "char_end": start + len(text), "text": text}


class TextFeatureChecks(unittest.TestCase):
    def test_question_words(self):
        self.assertEqual(question_word("When was the company founded?"), "when")
        self.assertEqual(question_word("How many people live in Paris?"), "how_many")
        self.assertEqual(question_word("In what year did it move?"), "what")
        self.assertEqual(question_word("Whose book was it?"), "who")
        self.assertEqual(question_word("How did it end?"), "how")
        self.assertEqual(question_word("Name the founder."), "other")

    def test_a_trimmed_candidate_is_described_relative_to_rank_one(self):
        group = [span("in 1912 and", 1), span("1912", 2)]
        out = question_features(group, CONTEXT, "When was the company founded?")
        rank1, trimmed = out[0], out[1]
        f = lambda row, name: float(row[INDEX[name]])
        self.assertEqual((f(rank1, "starts_in"), f(rank1, "ends_and")), (1, 1))
        self.assertEqual((f(rank1, "lead_delta"), f(rank1, "trail_delta"), f(rank1, "nested_in_rank1")), (0, 0, 0))
        self.assertEqual(f(trimmed, "nested_in_rank1"), 1)
        self.assertEqual(f(trimmed, "contains_rank1"), 0)
        self.assertAlmostEqual(f(trimmed, "lead_delta"), -math.log1p(1))   # one leading word dropped
        self.assertAlmostEqual(f(trimmed, "trail_delta"), -math.log1p(1))  # one trailing word dropped
        self.assertEqual((f(trimmed, "drops_lead_W"), f(trimmed, "drops_trail_W")), (1, 1))
        self.assertEqual((f(trimmed, "adds_lead_W"), f(trimmed, "adds_trail_W")), (0, 0))
        self.assertEqual((f(trimmed, "before_in"), f(trimmed, "after_and")), (1, 1))   # neighbours in the passage
        self.assertEqual((f(trimmed, "starts_in"), f(trimmed, "ends_and")), (0, 0))
        self.assertEqual((f(trimmed, "has_digit"), f(trimmed, "when_x_digit"), f(trimmed, "who_x_digit")), (1, 1, 0))
        self.assertEqual((f(rank1, "n_nested"), f(trimmed, "n_nested")), (1, 1))

    def test_an_extended_candidate_adds_words(self):
        group = [span("1912", 1), span("in 1912 and", 2)]
        out = question_features(group, CONTEXT, "When was it founded?")
        f = lambda name: float(out[1][INDEX[name]])
        self.assertEqual((f("contains_rank1"), f("adds_lead_W"), f("adds_trail_W")), (1, 1, 1))
        self.assertAlmostEqual(f("lead_delta"), math.log1p(1))

    def test_first_candidate_must_be_rank_one(self):
        with self.assertRaises(ValueError):
            question_features([span("1912", 2)], CONTEXT, "When?")

    def test_build_from_an_export_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            rows = [dict(span("in 1912 and", 1), qid="a"), dict(span("1912", 2), qid="a"),
                    dict(span("Paris", 1), qid="b")]
            (folder / "candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            doc = {"data": [{"title": "T", "paragraphs": [{"context": CONTEXT, "qas": [
                {"id": "a", "question": "When was it founded?"}, {"id": "b", "question": "Where did it move?"}]}]}]}
            (folder / "validation.json").write_text(json.dumps(doc), encoding="utf-8")
            X, names = build_text_features(folder, folder / "validation.json")
        self.assertEqual(X.shape, (3, len(NAMES)))
        self.assertEqual(tuple(names), NAMES)
        self.assertEqual(float(X[2][INDEX["where_x_capital"]]), 1.0)

    def test_head_features_concatenate_in_a_fixed_order(self):
        table = {"scalars": np.zeros((4, 3), np.float32), "text": np.ones((4, 2), np.float32),
                 "window": np.zeros(4, dtype=int)}
        pooled, h_cls = np.full((4, 5), 2.0, np.float32), np.full((1, 7), 3.0, np.float32)
        rows = np.arange(4)
        self.assertEqual(features(table, rows, "scalars", pooled, h_cls).shape, (4, 3))
        both = features(table, rows, "scalars+text+span", pooled, h_cls)
        self.assertEqual(both.shape, (4, 10))
        self.assertTrue((both[:, 3:5] == 1).all() and (both[:, 5:] == 2).all())
        self.assertEqual(features(table, rows, "scalars+span+cls", pooled, h_cls).shape, (4, 15))
        with self.assertRaises(ValueError):
            features(table, rows, "scalars+bogus", pooled, h_cls)


if __name__ == "__main__":
    unittest.main()