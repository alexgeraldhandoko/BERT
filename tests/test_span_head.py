import unittest

import numpy as np

from qa_baseline.span_head import (
    group_layout,
    listwise_loss_and_grad,
    standardize,
    standardizer,
    top1_metrics,
    train_listwise,
)


class ListwiseChecks(unittest.TestCase):
    def test_group_layout_and_noncontiguous_rows(self):
        starts, sizes = group_layout([4, 4, 4, 7, 9, 9])
        self.assertEqual(starts.tolist(), [0, 3, 4])
        self.assertEqual(sizes.tolist(), [3, 1, 2])
        with self.assertRaises(ValueError):
            group_layout([1, 2, 1])

    def test_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        starts, sizes = group_layout([0] * 3 + [1] * 4 + [2] * 2 + [3] * 5)
        scores = rng.normal(size=14)
        correct = np.zeros(14)
        correct[[1, 4, 6, 9, 12]] = 1  # group 2 (rows 7-8) has no correct candidate on purpose

        def loss_of(s):
            return listwise_loss_and_grad(s, correct, starts, sizes)[0]

        _, grad = listwise_loss_and_grad(scores, correct, starts, sizes)
        numeric = np.array([(loss_of(scores + e) - loss_of(scores - e)) / 2e-6
                            for e in np.eye(14) * 1e-6])
        np.testing.assert_allclose(grad, numeric, atol=1e-7)
        self.assertEqual(grad[7:9].tolist(), [0.0, 0.0])

    def test_loss_is_zero_when_all_mass_is_on_correct_rows(self):
        starts, sizes = group_layout([0, 0, 0])
        loss, _ = listwise_loss_and_grad(np.array([50.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), starts, sizes)
        self.assertAlmostEqual(loss, 0.0, places=8)

    def test_top1_ties_pick_the_first_row_and_missing_questions_count_as_wrong(self):
        starts, sizes = group_layout([0, 0, 0, 1, 1])
        em = np.array([1, 0, 0, 0, 1])
        f1 = np.array([1.0, 0.0, 0.5, 0.2, 1.0])
        self.assertEqual(top1_metrics(np.zeros(5), em, f1, starts, sizes, 2), (50.0, 60.0))
        self.assertEqual(top1_metrics(np.zeros(5), em, f1, starts, sizes, 4), (25.0, 30.0))

    def test_standardizer_handles_constant_columns(self):
        X = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]], dtype=np.float32)
        mean, std = standardizer(X)
        Z = standardize(X, mean, std)
        np.testing.assert_allclose(Z[:, 0].mean(), 0, atol=1e-6)
        self.assertTrue((Z[:, 1] == 0).all())

    def test_training_recovers_a_hidden_rule_the_baseline_ignores(self):
        rng = np.random.default_rng(1)

        def make(n_questions):
            rows, correct, q = [], [], []
            for i in range(n_questions):
                hit = rng.integers(0, 6)
                signal = rng.normal(0, 0.3, 6)
                signal[hit] += 2.0
                for rank in range(6):
                    rows.append([-rank, signal[rank]])  # feature 0 follows rank, feature 1 is the hidden signal
                    correct.append(float(rank == hit))
                    q.append(i)
            return np.array(rows, dtype=np.float32), np.array(correct), q

        X, y, q = make(150)
        Xv, yv, qv = make(60)
        groups, groups_v = group_layout(q), group_layout(qv)
        mean, std = standardizer(X)
        X, Xv = standardize(X, mean, std), standardize(Xv, mean, std)
        w, loss, step = train_listwise(X, y, groups, Xv, yv, groups_v, l2=0.01, steps=200, lr=0.05)
        em_head, _ = top1_metrics(Xv @ w, yv, yv, *groups_v, 60)
        em_base, _ = top1_metrics(np.tile(np.arange(6)[::-1], 60).astype(float), yv, yv, *groups_v, 60)
        self.assertGreater(em_head, 85)
        self.assertLess(em_base, 40)
        self.assertGreater(step, 0)


if __name__ == "__main__":
    unittest.main()