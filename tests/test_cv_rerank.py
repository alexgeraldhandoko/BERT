import unittest

import numpy as np

from qa_baseline.candidate_table import SCALAR_NAMES
from qa_baseline.cv_rerank import assign_folds, mixed_scores, run_cv
from qa_baseline.span_head import group_layout


def synthetic(n_articles=20, per_article=6, k=6, dim=8, seed=0):
    """Candidate k of each question is correct for one random k; the hidden signal sits in the span vector."""
    rng = np.random.default_rng(seed)
    titles, rows, pooled = [], [], []
    for a in range(n_articles):
        for _ in range(per_article):
            q = len(titles)
            titles.append(f"article{a}")
            hit = rng.integers(0, k)
            for rank in range(k):
                correct = int(rank == hit)
                vec = rng.normal(0, 1, dim)
                vec[0] += 2.5 * correct
                pooled.append(vec)
                rows.append((q, rank + 1, 10 - rank + rng.normal(0, 0.2), int(rng.integers(1, 6)), correct))
    n = len(rows)
    scalars = np.zeros((n, len(SCALAR_NAMES)), dtype=np.float32)
    names = list(SCALAR_NAMES)
    scalars[:, names.index("score")] = [r[2] for r in rows]
    scalars[:, names.index("span_tokens")] = [r[3] for r in rows]
    scalars[:, names.index("rank")] = [r[1] for r in rows]
    table = {
        "q_index": np.array([r[0] for r in rows], dtype=np.int32),
        "rank": np.array([r[1] for r in rows], dtype=np.int16),
        "em": np.array([r[4] for r in rows], dtype=np.int8),
        "f1": np.array([float(r[4]) for r in rows], dtype=np.float32),
        "window": np.array([r[0] for r in rows], dtype=np.int32),
        "scalars": scalars,
        "scalar_names": np.array(SCALAR_NAMES),
        "num_questions": np.array(len(titles)),
    }
    return table, np.array(pooled, dtype=np.float32), rng.normal(size=(len(titles), dim)).astype(np.float32), titles


class CvChecks(unittest.TestCase):
    def test_folds_keep_articles_together_and_balance_questions(self):
        titles = [f"a{i}" for i in range(23) for _ in range(5 + i % 4)]
        folds = assign_folds(titles, 5)
        by_article = {}
        for title, fold in zip(titles, folds):
            by_article.setdefault(title, set()).add(int(fold))
        self.assertTrue(all(len(f) == 1 for f in by_article.values()))
        counts = np.bincount(folds, minlength=5)
        self.assertEqual(counts.sum(), len(titles))
        self.assertLessEqual(counts.max() - counts.min(), 8)
        np.testing.assert_array_equal(folds, assign_folds(titles, 5))

    def test_every_question_is_scored_once_and_a_hidden_signal_is_found(self):
        table, pooled, h_cls, titles = synthetic()
        result = run_cv(table, pooled, h_cls, titles, configs=("scalars+span",),
                        l2_grids={"scalars+span": (0.1,)}, steps=150, lr=0.05)
        self.assertTrue(result["scored_once"])
        self.assertEqual(len(result["settings"]), 5)
        for train, calib, score in result["fold_sizes"]:
            self.assertEqual(train + calib + score, len(titles))
        report = result["report"]
        self.assertLess(report["baseline"]["em"], 40)          # the baseline ignores the signal
        self.assertGreater(report["scalars+span"]["em"], 70)   # the span head finds it
        self.assertTrue(report["scalars+span"]["adopt"])
        self.assertEqual(report["scalars+span"]["folds_improved_em"], 5)
        self.assertFalse(report["length penalty"]["adopt"])    # no length signal in this data


    def test_mixed_scores_endpoints(self):
        base = np.array([3.0, 1.0])
        head = np.array([1.0, 3.0])

        np.testing.assert_allclose(mixed_scores(base, head, 1.0), base)
        np.testing.assert_allclose(mixed_scores(base, head, 0.0), head)
        np.testing.assert_allclose(
            mixed_scores(base, head, 0.25),
            0.25 * base + 0.75 * head,
        )

if __name__ == "__main__":
    unittest.main()