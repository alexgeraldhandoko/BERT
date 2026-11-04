"""Checks for question-level features, forest fitting, and official QA evaluation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from qa_baseline.kamath import FEATURE_NAMES, build_records, evaluate, expected_calibration_error, fit
from qa_baseline.postprocess import best_answers


class KamathChecks(unittest.TestCase):
    def test_cross_window_features_follow_final_baseline_answer(self):
        examples = [{
            "id": "q1", "context": "London Paris", "answers": [{"text": "Paris"}]
        }]
        features = [
            {"example_id": "q1", "offset_mapping": [None, [0, 6], None]},
            {"example_id": "q1", "offset_mapping": [None, None, [7, 12]]},
        ]
        starts = [[0, 11, 0], [0, 0, 9]]
        ends = [[0, 11, 0], [0, 0, 9]]
        config = {"n_best_size": 3, "max_answer_tokens": 3}
        records = build_records(examples, features, starts, ends, config)
        self.assertEqual(best_answers(examples, features, starts, ends, 3, 3), {"q1": "London"})
        self.assertEqual(records[0]["answer"], "London")
        self.assertEqual(records[0]["correct"], 0)
        self.assertEqual(records[0]["features"][:2], [2, 1])
        self.assertEqual(len(records[0]["features"]), len(FEATURE_NAMES))
        self.assertGreater(records[0]["raw_confidence"], 0.9)

    def test_ece_uses_five_equal_width_bins(self):
        self.assertAlmostEqual(expected_calibration_error([0, 1], [0.1, 0.9], 5), 0.1)

    def test_fit_and_evaluate_keep_predictions_and_report_metrics(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is not installed")
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            document = {
                "version": "1.1",
                "data": [{"title": "toy", "paragraphs": [{
                    "context": "London Paris",
                    "qas": [
                        {"id": "q1", "question": "Where?", "answers": [{"answer_start": 7, "text": "Paris"}]},
                        {"id": "q2", "question": "Where else?", "answers": [{"answer_start": 0, "text": "London"}]},
                    ],
                }]}],
            }
            data_path = base / "data.json"
            data_path.write_text(json.dumps(document), encoding="utf-8")
            from qa_baseline.kamath import _sha256
            common = {"feature_names": FEATURE_NAMES, "model_sha256": "same-model"}
            train = {**common, "role": "calibration", "data_sha256": "different-data", "records": [
                {"id": "train1", "correct": 1, "features": [2, 1, .9, 0, 0, 0, 0]},
                {"id": "train2", "correct": 0, "features": [2, 1, .1, 0, 0, 0, 0]},
            ]}
            test = {**common, "role": "evaluation", "data_sha256": _sha256(data_path), "records": [
                {"id": "q1", "answer": "Paris", "correct": 1, "raw_confidence": .8,
                 "features": [2, 1, .8, 0, 0, 0, 0]},
                {"id": "q2", "answer": "Paris", "correct": 0, "raw_confidence": .2,
                 "features": [2, 1, .2, 0, 0, 0, 0]},
            ]}
            train_path, test_path = base / "train.json", base / "test.json"
            train_path.write_text(json.dumps(train), encoding="utf-8")
            test_path.write_text(json.dumps(test), encoding="utf-8")
            forest_path = base / "forest.joblib"
            fit(SimpleNamespace(records=train_path, output=forest_path, trees=10, min_leaf=1, seed=42, jobs=1))
            output = base / "output"
            summary_path = base / "summary.json"
            evaluate(SimpleNamespace(records=test_path, data=data_path, forest=forest_path,
                                     output=output, summary=summary_path))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["qa"]["exact"], 50.0)
            self.assertEqual(summary["questions"], 2)
            self.assertEqual(json.loads((output / "predictions.json").read_text()),
                             {"q1": "Paris", "q2": "Paris"})


if __name__ == "__main__":
    unittest.main()
