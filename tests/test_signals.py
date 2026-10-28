"""Orientation contract for the ablation ranking signals.

Every signal returned by `calibrate._signal` must satisfy **higher = more
confident**, because `_threshold_for_error` ranks by a descending sort. Margin
violated this (it was negated even though a larger top1-top2 gap *is* more
confidence), which made `calibrate target` report margin as UNREACHABLE on real
dev/val data while its own AUC was 0.76.
"""

from __future__ import annotations

import unittest

import numpy as np

from qa_baseline.calibrate import _signal, _threshold_for_error


def _cache(*rows: tuple[list[float], list[float], list[bool]]) -> dict:
    """Build a scores-cache dict from (start_logits, end_logits, mask) rows."""
    return {
        "qid": np.array([f"q{i}" for i in range(len(rows))], dtype="<U24"),
        "s": np.array([r[0] for r in rows], dtype=np.float32),
        "e": np.array([r[1] for r in rows], dtype=np.float32),
        "mask": np.array([r[2] for r in rows], dtype=bool),
        # every row's winning span is (0, 0) so the sharp row is the correct one
        "best_i": np.zeros(len(rows), dtype=np.int32),
        "best_j": np.zeros(len(rows), dtype=np.int32),
        "n_windows": np.ones(len(rows), dtype=np.int8),
    }


SHARP = ([5.0, -5.0, 0.0], [5.0, -5.0, 0.0], [True, True, True])
FLAT = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [True, True, True])


class TestSignalOrientation(unittest.TestCase):
    def test_higher_means_more_confident(self) -> None:
        cache = _cache(SHARP, FLAT)
        for name in ("prob", "margin", "entropy"):
            with self.subTest(signal=name):
                values = _signal(cache, name)
                self.assertGreater(
                    values[0], values[1],
                    f"{name} ranks the sharp row below the flat row -> inverted orientation",
                )

    def test_margin_without_legal_span_is_least_confident(self) -> None:
        # No context tokens -> span_margin is NaN -> must sort last, not first
        # (np.argsort places NaN last, so an unmapped NaN would be "most confident").
        cache = _cache(SHARP, ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [False, False, False]))
        values = _signal(cache, "margin")
        self.assertEqual(values[1], -np.inf)
        self.assertGreater(values[0], values[1])

    def test_margin_targeting_reaches_target(self) -> None:
        # Behavioural regression for the sign bug: with a correctly oriented
        # margin, a 5% error target is reachable and keeps exactly the correct
        # rows. Negated margin starts the ranking with the wrong rows, so no
        # prefix meets the target and _threshold_for_error raises.
        cache = _cache(SHARP, SHARP, FLAT, FLAT)
        y = np.array([1, 1, 0, 0])
        tau = _threshold_for_error(_signal(cache, "margin"), y, 0.05)
        kept = _signal(cache, "margin") >= tau
        np.testing.assert_array_equal(kept, [True, True, False, False])
        self.assertEqual(float(1.0 - y[kept].mean()), 0.0)


if __name__ == "__main__":
    unittest.main()
