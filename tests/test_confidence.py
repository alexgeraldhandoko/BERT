"""Unit tests for confidence signals and calibration metrics."""

from __future__ import annotations

import unittest

import numpy as np

from qa_baseline.confidence import (
    brier,
    brier_skill_score,
    correctness,
    ece,
    entropy_signal,
    fit_temperature,
    fit_two_temperature,
    nll,
    reliability_bins,
    renormalised_span_probability,
    risk_coverage,
    softmax_rows,
    span_margin,
    span_probability,
)


class SoftmaxAndSpanProbability(unittest.TestCase):
    def test_softmax_rows_is_stable_at_large_logits(self):
        x = np.array([[1000.0, 1000.0, 1000.0], [-20.0, 20.0, 0.0]])
        out = softmax_rows(x)
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out.sum(axis=-1), [1.0, 1.0])
        self.assertAlmostEqual(out[1][1], 1.0, places=6)

    def test_hand_built_two_token_example(self):
        # Window of 4 positions: only positions 1,2 are context tokens.
        # Start logits favour 1; end logits favour 2.
        s = np.array([0.0, 3.0, 1.0, 0.0])
        e = np.array([0.0, 1.0, 3.0, 0.0])
        mask = np.array([False, True, True, False])
        # T=1: softmax over the full row (4 positions).
        ps = np.exp(3) / (np.exp(0) + np.exp(3) + np.exp(1) + np.exp(0))
        pe = np.exp(3) / (np.exp(0) + np.exp(1) + np.exp(3) + np.exp(0))
        self.assertAlmostEqual(span_probability(s, e, mask, 1, 2, T=1.0), ps * pe)
        # Temperature flattens towards uniform: 1/4 * 1/4 as T -> inf is
        # approximated by a large T pushing probabilities closer to (1/4)^2.
        c_large = span_probability(s, e, mask, 1, 2, T=100.0)
        self.assertGreater(c_large, (0.25 * 0.25) - 0.01)
        self.assertLess(c_large, ps * pe)

    def test_temperature_monotone_between_one_and_large(self):
        rng = np.random.default_rng(0)
        s = rng.normal(0, 5, size=8)
        e = rng.normal(0, 5, size=8)
        mask = np.ones(8, dtype=bool)
        c1 = span_probability(s, e, mask, int(s.argmax()), int(e.argmax()), T=1.0)
        c2 = span_probability(s, e, mask, int(s.argmax()), int(e.argmax()), T=2.0)
        self.assertGreater(c1, c2)  # raising T flattens the distribution

    def test_span_margin_positive_and_masked(self):
        s = np.array([0.0, 5.0, 1.0, 0.0])
        e = np.array([0.0, 1.0, 5.0, 0.0])
        mask = np.array([False, True, True, False])
        margin = span_margin(s, e, mask, 1, 2)
        self.assertGreater(margin, 0.0)
        # All-masked window yields no legal span.
        empty = span_margin(s, e, np.zeros(4, dtype=bool), -1, -1)
        self.assertTrue(np.isnan(empty))


class Correctness(unittest.TestCase):
    def test_exact_match_max_over_golds(self):
        example = {"answers": [{"text": "New York"}, {"text": "NYC"}]}
        self.assertEqual(correctness(example, "New York"), 1)
        self.assertEqual(correctness(example, "new york"), 1)  # vendor normalises
        self.assertEqual(correctness(example, "Paris"), 0)

    def test_fallback_when_all_golds_normalise_to_empty(self):
        example = {"answers": [{"text": "  "}, {"text": ""}]}
        self.assertEqual(correctness(example, ""), 1)
        self.assertEqual(correctness(example, "x"), 0)


class Metrics(unittest.TestCase):
    def test_ece_zero_on_perfectly_calibrated_synthetic_data(self):
        # Bin k has confidence k/n_bins and accuracy exactly k/n_bins.
        n_bins = 15
        n_per = 500
        rng = np.random.default_rng(7)
        cs, ys = [], []
        for k in range(n_bins):
            p = (k + 0.5) / n_bins
            cs.append(np.full(n_per, p))
            ys.append(rng.random(n_per) < p)  # sampling noise only
        # With sampling noise ECE is small but nonzero; assert tight bound.
        value = ece(np.concatenate(cs), np.concatenate(ys).astype(float), n_bins)
        self.assertLess(value, 0.05)

    def test_ece_of_constant_predictor_is_abs_p_minus_base(self):
        c = np.full(100, 0.9)
        y = np.concatenate([np.ones(74), np.zeros(26)])  # base rate 0.74
        self.assertAlmostEqual(ece(c, y, n_bins=1), abs(0.9 - 0.74), places=9)

    def test_brier_known_values(self):
        c = np.array([1.0, 0.0, 0.5])
        y = np.array([1.0, 0.0, 1.0])
        self.assertAlmostEqual(brier(c, y), (0 + 0 + 0.25) / 3)
        base = 2 / 3
        self.assertAlmostEqual(brier_skill_score(c, y), 1 - (1 / 12) / (base * (1 - base)))

    def test_nll_decreases_with_sharper_correct_confidence(self):
        y = np.ones(100)
        self.assertLess(nll(np.full(100, 0.9), y), nll(np.full(100, 0.6), y))

    def test_reliability_bins_are_equal_mass(self):
        rng = np.random.default_rng(1)
        c = rng.random(1000)
        y = (rng.random(1000) < c).astype(float)
        bins = reliability_bins(c, y, n_bins=10)
        self.assertEqual(len(bins), 10)
        self.assertEqual({b["n"] for b in bins}, {100})
        self.assertEqual(sum(b["n"] for b in bins), 1000)

    def test_risk_coverage_em_on_top_fraction(self):
        c = np.array([0.9, 0.8, 0.3, 0.1])
        y = np.array([1.0, 1.0, 0.0, 0.0])
        rows = risk_coverage(c, y, coverages=(0.5, 1.0))
        self.assertAlmostEqual(rows[0]["em"], 1.0)  # top 2 are all correct
        self.assertAlmostEqual(rows[1]["em"], 0.5)  # full set base rate
        self.assertEqual(rows[0]["n"], 2)


class V2Signals(unittest.TestCase):
    def test_renormalised_probability_sums_to_one_over_legal_spans(self):
        # THE v2 property: the renormalised scores form a distribution over
        # legal spans, unlike the v1 raw product.
        rng = np.random.default_rng(5)
        L = 6
        s, e = rng.normal(0, 2, L), rng.normal(0, 2, L)
        mask = np.array([False, True, True, True, True, False])
        total = 0.0
        for a in range(L):
            for b in range(a, L):
                if mask[a] and mask[b]:
                    total += renormalised_span_probability(s, e, mask, a, b)
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_renormalised_temperature_flattens(self):
        rng = np.random.default_rng(6)
        L = 6
        s, e = rng.normal(0, 3, L), rng.normal(0, 3, L)
        mask = np.ones(L, dtype=bool)
        i, j = 2, 3
        p1 = renormalised_span_probability(s, e, mask, i, j, T=1.0)
        p2 = renormalised_span_probability(s, e, mask, i, j, T=10.0)
        self.assertGreater(p2, p1)  # higher T flattens towards uniform 1/n_legal

    def test_entropy_lower_for_sharper_distribution(self):
        L = 8
        mask = np.ones(L, dtype=bool)
        sharp_s = np.array([0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        flat_s = np.zeros(L)
        e = np.zeros(L)
        self.assertLess(
            entropy_signal(sharp_s, e, mask), entropy_signal(flat_s, e, mask)
        )


class FitTwoTemperature(unittest.TestCase):
    def test_two_temperatures_fix_overconfident_component_and_nll_improves(self):
        # When only ONE component is overconfident (truth = ps^0.5 * pe), the
        # fit sharpens it (T < 1) and the OTHER component flattens (T > 1) to
        # keep the predicted base rate matched — the two temperatures share a
        # base-rate budget. The mirrored case must produce the mirrored
        # cross-pattern; NLL must decrease in both.
        rng = np.random.default_rng(21)
        n, L = 600, 6
        s = rng.normal(0, 4.0, size=(n, L))
        e = rng.normal(0, 4.0, size=(n, L))
        i, j = s.argmax(axis=1), e.argmax(axis=1)
        ps = softmax_rows(s)[np.arange(n), i]
        pe = softmax_rows(e)[np.arange(n), j]

        def nll_at(ts, te):
            p = np.clip(
                softmax_rows(s / ts)[np.arange(n), i] * softmax_rows(e / te)[np.arange(n), j],
                1e-12, 1 - 1e-12,
            )
            return nll(p, y)

        # Case 1: start overconfident.
        y = (rng.random(n) < np.clip(ps ** 0.5 * pe / (ps ** 0.5 * pe).mean() * 0.5, 1e-3, 0.99)).astype(float)
        ts, te = fit_two_temperature(s, e, np.ones((n, L), bool), i, j, y)
        self.assertLess(ts, 1.0)
        self.assertGreater(te, 1.0)
        self.assertLess(nll_at(ts, te), nll_at(1.0, 1.0))

        # Case 2: end overconfident — mirrored pattern.
        y = (rng.random(n) < np.clip(ps * pe ** 0.5 / (ps * pe ** 0.5).mean() * 0.5, 1e-3, 0.99)).astype(float)
        ts, te = fit_two_temperature(s, e, np.ones((n, L), bool), i, j, y)
        self.assertLess(te, 1.0)
        self.assertGreater(ts, 1.0)
        self.assertLess(nll_at(ts, te), nll_at(1.0, 1.0))


class FitTemperature(unittest.TestCase):
    def _synthetic(self, n=400, seed=3, conf_scale=1.0):
        rng = np.random.default_rng(seed)
        L = 6
        s = rng.normal(0, conf_scale, size=(n, L))
        e = rng.normal(0, conf_scale, size=(n, L))
        i = s.argmax(axis=1)
        j = e.argmax(axis=1)
        c = np.array(
            [span_probability(s[k], e[k], np.ones(L, bool), i[k], j[k]) for k in range(n)]
        )
        # Label by sampling from confidence: a well-specified but miscalibrated setup.
        y = (rng.random(n) < c).astype(float)
        return s, e, np.ones((n, L), bool), i, j, y, c

    def test_fitted_temperature_reduces_nll(self):
        s, e, mask, i, j, y, c = self._synthetic()
        base = nll(c, y)
        T = fit_temperature(s, e, mask, i, j, y)
        ps = softmax_rows(s / T)[np.arange(len(y)), i]
        pe = softmax_rows(e / T)[np.arange(len(y)), j]
        self.assertLess(nll(ps * pe, y), base)

    def test_fitted_temperature_is_reasonable_float(self):
        s, e, mask, i, j, y, _ = self._synthetic()
        T = fit_temperature(s, e, mask, i, j, y)
        self.assertTrue(0.25 <= T <= 5.0)

    def test_overconfident_data_gets_temperature_above_one(self):
        # y sampled from a SHARPER distribution than the logits imply (c**2 < c)
        # means the model is overconfident: true success rate is lower than the
        # stated confidence, so fitting must RAISE T to flatten it.
        s, e, mask, i, j, _, c = self._synthetic(conf_scale=1.0)
        rng = np.random.default_rng(11)
        y = (rng.random(len(c)) < c**2).astype(float)
        T = fit_temperature(s, e, mask, i, j, y)
        self.assertGreater(T, 1.0)


if __name__ == "__main__":
    unittest.main()
