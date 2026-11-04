import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from qa_baseline.candidate_table import load_table, summarize
from qa_baseline.cv_rerank import ALPHA_GRID, LAMBDA_GRID
from qa_baseline.dev_score import answers_for, decide, head_scores_chunked, score_dev
from qa_baseline.dev_table import build_dev_inputs, export_rank_one
from qa_baseline.final_head import freeze, load_final, save_final
from qa_baseline.text_features import NAMES

from test_candidate_table import write_export  # run with: unittest discover -s tests
from test_cv_rerank import synthetic


def as_dev(table):
    """Give a synthetic table the extra keys a dev table has: every question in the test part."""
    table = dict(table)
    table["part"] = np.full(len(table["q_index"]), 2, dtype=np.int8)
    table["question_part"] = np.full(int(table["num_questions"]), 2, dtype=np.int8)
    table["qid"] = np.array([f"q{i}" for i in table["q_index"]])
    return table


class FinalHeadChecks(unittest.TestCase):
    def setUp(self):
        self.table, self.pooled, _, self.titles = synthetic(seed=0)
        self.frozen = freeze(self.table, self.pooled, self.titles, l2_grid=(0.1,), steps=150, lr=0.05)

    def test_trains_on_four_folds_and_calibrates_on_the_fifth(self):
        self.assertEqual(self.frozen["calib_questions"], 24)
        self.assertEqual(self.frozen["train_questions"], 96)
        self.assertEqual(self.frozen["w"].shape[0], 14 + 4 + 8)      # scalars + text + span
        self.assertIn(self.frozen["alpha"], ALPHA_GRID)
        self.assertIn(self.frozen["lambda"], LAMBDA_GRID)
        self.assertEqual(self.frozen["config"], "scalars+text+span")

    def test_freezing_is_deterministic(self):
        again = freeze(self.table, self.pooled, self.titles, l2_grid=(0.1,), steps=150, lr=0.05)
        np.testing.assert_array_equal(again["w"], self.frozen["w"])
        self.assertEqual((again["alpha"], again["lambda"], again["step"]),
                         (self.frozen["alpha"], self.frozen["lambda"], self.frozen["step"]))

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            save_final(folder, self.frozen)
            loaded = load_final(folder)
            settings = json.loads((Path(folder) / "final_settings.json").read_text(encoding="utf-8"))
        np.testing.assert_array_equal(loaded["w"], self.frozen["w"])
        np.testing.assert_array_equal(loaded["mean"], self.frozen["mean"])
        self.assertEqual((loaded["config"], loaded["alpha"], loaded["lambda"], loaded["step"]),
                         (self.frozen["config"], self.frozen["alpha"], self.frozen["lambda"], self.frozen["step"]))
        self.assertEqual(settings["n_features"], 26)


class DevScoreChecks(unittest.TestCase):
    def setUp(self):
        held, pooled, _, titles = synthetic(seed=0)
        self.final = freeze(held, pooled, titles, l2_grid=(0.1,), steps=150, lr=0.05)
        dev, self.dev_pooled, _, self.dev_titles = synthetic(seed=1)
        self.dev = as_dev(dev)

    def test_chunked_scoring_equals_one_pass(self):
        a = head_scores_chunked(self.dev, self.dev_pooled, self.final, chunk=7)
        b = head_scores_chunked(self.dev, self.dev_pooled, self.final, chunk=10**6)
        # float32 matrix products round differently for different chunk shapes (about 1e-7 absolute)
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)

    def test_frozen_head_is_applied_to_unseen_questions_and_the_rule_is_evaluated(self):
        report, outcomes, paired = score_dev(self.dev, self.dev_pooled, self.final, np.array(self.dev_titles))
        self.assertLess(report["baseline"]["em"], 40)
        self.assertGreater(report["frozen head + mix"]["em"], 60)
        verdict = decide(report, paired)
        self.assertTrue(verdict["adopt_over_baseline"])
        self.assertTrue(verdict["beats_penalty"])          # no length signal in this data, so the penalty is weak
        self.assertEqual(len(outcomes["baseline"][0]), len(self.dev_titles))

    def test_answers_for_fills_missing_questions_with_empty_strings(self):
        table = {"qid": np.array(["a", "a", "b"])}
        answers = answers_for(table, [0, 2], ["x", "y", "z"], ["a", "b", "c"])
        self.assertEqual(answers, {"a": "x", "b": "z", "c": ""})


class DevTableChecks(unittest.TestCase):
    def test_every_dev_question_lands_in_the_test_part(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            write_export(folder)                                   # candidates for q0..q3; q4 has none
            doc = {"data": [{"title": "T", "paragraphs": [{"context": "c" * 200, "qas": [
                {"id": f"q{i}", "question": "When?"} for i in range(5)]}]}]}
            dev_json = folder / "dev.json"
            dev_json.write_text(json.dumps(doc), encoding="utf-8")
            out = folder / "out"
            table, X = build_dev_inputs(folder, dev_json, out)
            reloaded = load_table(out / "table.npz")
            split = json.loads((out / "dev_split.json").read_text(encoding="utf-8"))["parts"]
            self.assertTrue((out / "text_features.npz").exists())
        self.assertEqual(int(table["num_questions"]), 5)
        self.assertTrue((table["question_part"] == 2).all() and (table["part"] == 2).all())
        self.assertEqual(X.shape, (len(table["q_index"]), len(NAMES)))
        np.testing.assert_array_equal(reloaded["em"], table["em"])
        self.assertEqual((split["head_train"]["ids"], split["calib"]["ids"]), ([], []))
        self.assertEqual(split["test"]["ids"], [f"q{i}" for i in range(5)])
        self.assertEqual(list(summarize(table)), ["test"])         # empty parts are skipped, no NaN warnings

    def test_table_can_be_checked_against_the_exports_own_rank_one_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            write_export(folder)
            self.assertIsNone(export_rank_one(folder))             # this fixture has no top1 fields
            rank_one_em = [1, 0, 0, 1, 0]                          # q0..q4; q4 has no candidates and counts as wrong
            (folder / "questions.json").write_text(json.dumps(
                [{"id": f"q{i}", "top1_em": e, "top1_f1": float(e)} for i, e in enumerate(rank_one_em)]), encoding="utf-8")
            doc = {"data": [{"title": "T", "paragraphs": [{"context": "c" * 200, "qas": [
                {"id": f"q{i}", "question": "When?"} for i in range(5)]}]}]}
            (folder / "dev.json").write_text(json.dumps(doc), encoding="utf-8")
            table, _ = build_dev_inputs(folder, folder / "dev.json", folder / "out")
            exported = export_rank_one(folder)
        self.assertEqual(exported, (40.0, 40.0))
        self.assertAlmostEqual(summarize(table)["test"]["em@1"], exported[0])


if __name__ == "__main__":
    unittest.main()