"""Checks for the scoring and statistics helpers in export_candidates."""


import math
import unittest
from pathlib import Path


from qa_baseline.export_candidates import (
    STAT_FIELDS,
    load_official_scorer,
    score_answer,
    window_statistics,
)


SCORER = Path(__file__).resolve().parents[1] / "vendor" / "evaluate-v2.0.py"


try:
    import torch
except ImportError:
    torch = None




class ScoreAnswerChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = load_official_scorer(SCORER)


    def test_exact_match_ignores_case_articles_and_punctuation(self):
        self.assertEqual(score_answer(self.scorer, ["The Gravity."], "gravity"), (1, 1.0))


    def test_partial_answer_gets_partial_f1(self):
        exact, f1 = score_answer(self.scorer, ["gravity"], "falls under gravity")
        self.assertEqual(exact, 0)
        self.assertAlmostEqual(f1, 0.5)


    def test_best_gold_answer_is_used(self):
        self.assertEqual(score_answer(self.scorer, ["wrong", "gravity"], "gravity"), (1, 1.0))


    def test_empty_prediction_scores_zero(self):
        self.assertEqual(score_answer(self.scorer, ["gravity"], ""), (0, 0))




@unittest.skipIf(torch is None, "torch is not installed")
class WindowStatisticsChecks(unittest.TestCase):
    # 8 positions: 0 = [CLS], 1-2 = question, 3-6 = context, 7 = padding
    IN_CONTEXT = [False, False, False, True, True, True, True, False]


    def stats(self, values):
        logits = torch.tensor(values, dtype=torch.float32)
        return window_statistics(logits, logits, torch.tensor(self.IN_CONTEXT))


    def test_uniform_logits(self):
        stats = self.stats([0.0] * 8)
        self.assertAlmostEqual(stats["lse_all_start"], math.log(8), places=5)
        self.assertAlmostEqual(stats["lse_ctx_start"], math.log(4), places=5)
        self.assertAlmostEqual(stats["ent_ctx_start"], math.log(4), places=5)
        self.assertAlmostEqual(stats["ctx_mass_start"], 0.5, places=5)
        self.assertAlmostEqual(stats["cls_start"], 0.0, places=5)


    def test_uneven_logits_match_hand_computation(self):
        values = [2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, -1.0]
        stats = self.stats(values)
        everything = sum(math.exp(v) for v in values)
        context = sum(math.exp(v) for v, keep in zip(values, self.IN_CONTEXT) if keep)
        self.assertAlmostEqual(stats["lse_all_end"], math.log(everything), places=5)
        self.assertAlmostEqual(stats["lse_ctx_end"], math.log(context), places=5)
        self.assertAlmostEqual(stats["ctx_mass_end"], context / everything, places=5)
        self.assertAlmostEqual(stats["cls_end"], 2.0, places=5)


    def test_keys_match_stat_fields(self):
        self.assertEqual(sorted(self.stats([0.0] * 8)), sorted(STAT_FIELDS))




if __name__ == "__main__":
    unittest.main()