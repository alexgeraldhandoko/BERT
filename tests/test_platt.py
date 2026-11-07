"""Platt scaling: Newton fit recovers known parameters; dispatch works.

The known-parameters case: sample c from a known Platt map applied to a
uniform logit grid, label stochastically from the mapped probability, refit,
and require the recovered (a, b) to be close to the generating values.
"""

from __future__ import annotations

import unittest

import numpy as np

from qa_baseline.calibrate import _apply_calibration, _calibration_label
from qa_baseline.confidence import _platt_apply, fit_platt


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class TestPlattFit(unittest.TestCase):
    def test_recovers_known_parameters(self) -> None:
        rng = np.random.default_rng(42)
        n = 20000
        z = rng.uniform(-4, 4, n)
        a_true, b_true = 1.7, -0.6
        p = _sigmoid(a_true * z + b_true)
        y = (rng.uniform(0, 1, n) < p).astype(float)
        c = _sigmoid(z)  # invert the logit: c = sigmoid(z), so logit(c) = z
        a_fit, b_fit = fit_platt(c, y)
        self.assertAlmostEqual(a_fit, a_true, delta=0.15)
        self.assertAlmostEqual(b_fit, b_true, delta=0.15)

    def test_identity_initialisation_is_fixed_point(self) -> None:
        # Perfectly calibrated c (y drawn exactly from c) should stay near a=1, b=0.
        rng = np.random.default_rng(0)
        c = rng.uniform(0.05, 0.95, 20000)
        y = (rng.uniform(0, 1, c.size) < c).astype(float)
        a, b = fit_platt(c, y)
        self.assertAlmostEqual(a, 1.0, delta=0.15)
        self.assertAlmostEqual(b, 0.0, delta=0.15)

    def test_monotone_and_bounded(self) -> None:
        c = np.linspace(0.01, 0.99, 200)
        out = _platt_apply(c, 2.0, 0.5)
        self.assertTrue(np.all(out >= 0) and np.all(out <= 1))
        self.assertTrue(np.all(np.diff(out) > 0))  # a > 0 keeps the order

    def test_rejects_bad_input(self) -> None:
        with self.assertRaises(ValueError):
            fit_platt(np.array([0.5, 0.6]), np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            fit_platt(np.array([[0.5], [0.6]]), np.array([1.0, 0.0]))


class TestCalibrationDispatch(unittest.TestCase):
    def _scores(self) -> dict:
        # two rows, winning span at (0, 0); row 0 sharp, row 1 flat
        return {
            "s": np.array([[5.0, -5.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
            "e": np.array([[5.0, -5.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
            "mask": np.ones((2, 3), dtype=bool),
            "best_i": np.zeros(2, dtype=np.int32),
            "best_j": np.zeros(2, dtype=np.int32),
        }

    def test_platt_dispatch(self) -> None:
        scores = self._scores()
        c_uncal = np.array([0.9, 0.3])
        cal = {"method": "platt", "a": 1.5, "b": 0.2}
        out = _apply_calibration(cal, scores, c_uncal)
        expected = _platt_apply(c_uncal, 1.5, 0.2)
        np.testing.assert_allclose(out, expected)

    def test_temperature_dispatch_unchanged(self) -> None:
        # regression: the temperature path must produce identical numbers
        # through the dispatcher as the direct call it replaced.
        from qa_baseline.calibrate import _confidence

        scores = self._scores()
        cal = {"method": "temperature scaling", "temperature": 1.02}
        np.testing.assert_allclose(
            _apply_calibration(cal, scores, _confidence(scores, T=1.0)),
            _confidence(scores, T=1.02),
        )

    def test_label(self) -> None:
        self.assertEqual(_calibration_label({"method": "platt", "a": 0.767}), "+Platt (a=0.77)")
        self.assertEqual(
            _calibration_label({"method": "temperature scaling", "temperature": 1.02}),
            "+T (T=1.02)",
        )


class TestV2Signal(unittest.TestCase):
    def test_chunked_matches_per_row_and_normalises(self) -> None:
        from qa_baseline.confidence import (
            renormalised_probabilities,
            renormalised_span_probability,
        )

        rng = np.random.default_rng(7)
        n, length = 5, 12
        s = rng.normal(0, 3, (n, length)).astype(np.float32)
        e = rng.normal(0, 3, (n, length)).astype(np.float32)
        mask = np.ones((n, length), dtype=bool)
        bi = rng.integers(0, length, n)
        bj = np.minimum(bi + rng.integers(0, 4, n), length - 1)
        for T in (1.0, 0.7, 2.5):
            fast = renormalised_probabilities(s, e, mask, bi, bj, [T])[:, 0]
            slow = np.array([
                renormalised_span_probability(s[k], e[k], mask[k], bi[k], bj[k], T=T)
                for k in range(n)
            ])
            np.testing.assert_allclose(fast, slow, rtol=1e-10)
        # distribution property: v2 sums to 1 over the legal grid
        k = 0
        grid = np.array([
            renormalised_span_probability(s[k], e[k], mask[k], i, j)
            for i in range(length) for j in range(i, length)
        ])
        self.assertAlmostEqual(grid.sum(), 1.0, places=10)

    def test_v2_orientation(self) -> None:
        # extend the higher = more confident contract to the v2 signal
        from qa_baseline.calibrate import _signal

        cache = {
            "s": np.array([[5.0, -5.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
            "e": np.array([[5.0, -5.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
            "mask": np.ones((2, 3), dtype=bool),
            "best_i": np.zeros(2, dtype=np.int32),
            "best_j": np.zeros(2, dtype=np.int32),
        }
        values = _signal(cache, "v2")
        self.assertGreater(values[0], values[1])


if __name__ == "__main__":
    unittest.main()
