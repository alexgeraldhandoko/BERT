"""Fast checks for the fixed split, span extraction, and course scorer."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from qa_baseline.data import load_squad, split_train, write_squad_subset
from qa_baseline.postprocess import best_answers


ROOT = Path(__file__).resolve().parents[1]


class BaselineChecks(unittest.TestCase):
    def test_split_is_stable_when_input_order_changes(self):
        examples = [{"id": str(i), "article_title": str(i // 2)} for i in range(100)]
        train_a, val_a = split_train(examples, 0.1, 42)
        train_b, val_b = split_train(list(reversed(examples)), 0.1, 42)
        self.assertEqual({x["id"] for x in train_a}, {x["id"] for x in train_b})
        self.assertEqual({x["id"] for x in val_a}, {x["id"] for x in val_b})
        self.assertFalse({x["id"] for x in train_a} & {x["id"] for x in val_a})
        self.assertFalse(
            {x["article_title"] for x in train_a}
            & {x["article_title"] for x in val_a}
        )

    def test_best_span_uses_original_context_offsets_across_windows(self):
        examples = [{"id": "q1", "context": "New York City"}]
        features = [
            {"example_id": "q1", "offset_mapping": [None, [0, 3], None, None]},
            {"example_id": "q1", "offset_mapping": [None, [0, 3], [4, 8], [9, 13]]},
        ]
        starts = [[0, 1, 0, 0], [0, 8, 0, 0]]
        ends = [[0, 0, 1, 0], [0, 0, 8, 0]]
        answers = best_answers(examples, features, starts, ends, 3, 3)
        self.assertEqual(answers, {"q1": "New York"})

    def test_course_scorer_accepts_v1_data_and_reports_em_f1(self):
        document = {
            "version": "1.1",
            "data": [{"title": "example", "paragraphs": [{
                "context": "New York City",
                "qas": [{"id": "q1", "question": "Where?", "answers": [
                    {"answer_start": 0, "text": "New York"}
                ]}],
            }]}],
        }
        with tempfile.TemporaryDirectory() as temp:
            data_path = Path(temp) / "data.json"
            predictions_path = Path(temp) / "predictions.json"
            data_path.write_text(json.dumps(document), encoding="utf-8")
            predictions_path.write_text(json.dumps({"q1": "New York"}), encoding="utf-8")
            self.assertEqual(load_squad(data_path)[0]["id"], "q1")
            subset_path = Path(temp) / "subset.json"
            write_squad_subset(data_path, {"q1"}, subset_path)
            self.assertEqual(load_squad(subset_path)[0]["id"], "q1")
            result = subprocess.run(
                [sys.executable, str(ROOT / "vendor" / "evaluate-v2.0.py"),
                 str(data_path), str(predictions_path)],
                capture_output=True, text=True, check=True,
            )
            metrics = json.loads(result.stdout)
            self.assertEqual(metrics["exact"], 100.0)
            self.assertEqual(metrics["f1"], 100.0)
            self.assertEqual(metrics["total"], 1)


if __name__ == "__main__":
    unittest.main()

