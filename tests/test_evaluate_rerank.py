import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from qa_baseline.candidate_table import PARTS, SCALAR_NAMES
from qa_baseline.evaluate_rerank import (
    article_bootstrap,
    article_net,
    baseline_scores,
    load_titles,
    part_outcomes,
    penalty_scores,
    select_lambda,
)


def make_table():
    """Type A question: rank 1 is long and wrong, rank 2 short and right (a penalty fixes it).
    Type B question: rank 1 is short and right."""
    layout = {"head_train": "AAB", "calib": "ABA", "test": "AB"}  # test also gets a question with no candidates
    rows = []
    q = 0
    for part_number, name in enumerate(PARTS):
        for kind in layout[name]:
            candidates = (
                [(5.0, 6, 0, 0.5), (4.8, 2, 1, 1.0)] if kind == "A" else [(5.0, 2, 1, 1.0), (4.9, 6, 0, 0.5)]
            )
            for rank, (score, tokens, em, f1) in enumerate(candidates, start=1):
                rows.append((q, part_number, rank, score, tokens, em, f1))
            q += 1
    n = len(rows)
    scalars = np.zeros((n, len(SCALAR_NAMES)), dtype=np.float32)
    names = list(SCALAR_NAMES)
    scalars[:, names.index("score")] = [r[3] for r in rows]
    scalars[:, names.index("span_tokens")] = [r[4] for r in rows]
    question_part = [PARTS.index(name) for name in PARTS for _ in layout[name]] + [PARTS.index("test")]
    return {
        "qid": np.array([f"q{r[0]}" for r in rows]),
        "q_index": np.array([r[0] for r in rows], dtype=np.int32),
        "part": np.array([r[1] for r in rows], dtype=np.int8),
        "rank": np.array([r[2] for r in rows], dtype=np.int16),
        "em": np.array([r[5] for r in rows], dtype=np.int8),
        "f1": np.array([r[6] for r in rows], dtype=np.float32),
        "scalars": scalars,
        "scalar_names": np.array(SCALAR_NAMES),
        "num_questions": np.array(len(question_part)),
        "question_part": np.array(question_part, dtype=np.int8),
    }


class EvaluateRerankChecks(unittest.TestCase):
    def setUp(self):
        self.table = make_table()

    def test_baseline_takes_rank_one_and_counts_missing_questions_as_wrong(self):
        number = PARTS.index("test")
        rows = np.where(self.table["part"] == number)[0]
        questions, em, f1, picked = part_outcomes(self.table, number, baseline_scores(self.table, rows))
        self.assertEqual(len(questions), 3)           # A, B and the question with no candidates
        self.assertEqual(em.tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(len(picked), 2)

    def test_length_penalty_flips_only_the_long_wrong_answers(self):
        number = PARTS.index("calib")
        rows = np.where(self.table["part"] == number)[0]
        _, em, _, _ = part_outcomes(self.table, number, penalty_scores(self.table, rows, 0.2))
        self.assertEqual(em.tolist(), [1.0, 1.0, 1.0])

    def test_lambda_selection_prefers_the_smallest_lambda_that_helps(self):
        lam, curve = select_lambda(self.table)
        self.assertEqual(curve[0][0], 0.0)
        self.assertAlmostEqual(curve[0][1], 100 * 2 / 6)   # baseline: only the 2 type-B questions of 6 are right
        self.assertTrue(0.05 <= lam <= 0.06)
        self.assertAlmostEqual(dict((c[0], c[1]) for c in curve)[lam], 100.0)

    def test_bootstrap_detects_a_consistent_gain_and_a_wash(self):
        articles = np.repeat(np.arange(8), 5)
        mean, low, high, share = article_bootstrap(np.ones(40), articles)
        self.assertEqual((mean, low, high, share), (1.0, 1.0, 1.0, 1.0))
        mixed = np.where(np.arange(8).repeat(5) % 2 == 0, 1.0, -1.0)
        mean, low, high, share = article_bootstrap(mixed, articles)
        self.assertAlmostEqual(mean, 0.0)
        self.assertLess(low, 0.0)
        self.assertGreater(high, 0.0)

    def test_article_net_counts_articles_up_down_and_unchanged(self):
        diff = np.array([1, 0, -1, 0, 0, 0])
        articles = np.array(["a", "a", "b", "b", "c", "c"])
        self.assertEqual(article_net(diff, articles), (1, 1, 1))

    def test_titles_come_from_the_validation_file(self):
        document = {"data": [{"title": "T1", "paragraphs": [{"qas": [{"id": "x"}, {"id": "y"}]}]},
                             {"title": "T2", "paragraphs": [{"qas": [{"id": "z"}]}]}]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "v.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(load_titles(path), {"x": "T1", "y": "T1", "z": "T2"})


if __name__ == "__main__":
    unittest.main()