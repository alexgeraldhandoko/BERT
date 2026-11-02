import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from qa_baseline.evaluate_final_head import predict_export


def make_export(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)

    questions = [{"id": "q1"}, {"id": "q2"}]
    (folder / "questions.json").write_text(json.dumps(questions), encoding="utf-8")

    rows = [
        {
            "qid": "q1", "rank": 1, "window": 0, "score": 5.0,
            "start_logit": 2.5, "end_logit": 2.5, "logp_all": -1.0,
            "logp_ctx": -0.5, "tok_start": 1, "tok_end": 1,
            "char_start": 0, "char_end": 4, "text": "base",
        },
        {
            "qid": "q1", "rank": 2, "window": 0, "score": 4.0,
            "start_logit": 2.0, "end_logit": 2.0, "logp_all": -1.2,
            "logp_ctx": -0.7, "tok_start": 2, "tok_end": 2,
            "char_start": 5, "char_end": 9, "text": "head",
        },
    ]
    with (folder / "candidates.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")

    np.save(folder / "span_pooled.npy", np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float16))
    np.savez(
        folder / "windows.npz",
        ent_ctx_start=np.array([0.1], dtype=np.float32),
        ent_ctx_end=np.array([0.2], dtype=np.float32),
        cls_start=np.array([-1.0], dtype=np.float32),
        cls_end=np.array([-1.1], dtype=np.float32),
    )
    (folder / "predictions.json").write_text(
        json.dumps({"q1": "base", "q2": ""}), encoding="utf-8"
    )


def save_head(path: Path, alpha: float, choose_span_candidate: bool):
    w = np.zeros(16, dtype=np.float32)  # 14 scalar features + 2 span dims
    if choose_span_candidate:
        w[14] = 1.0

    np.savez(
        path,
        w=w,
        mean=np.zeros(16, dtype=np.float32),
        std=np.ones(16, dtype=np.float32),
        config=np.array("scalars+span"),
        l2=np.array(0.03),
        step=np.array(250),
        lr=np.array(0.02),
        alpha=np.array(alpha),
    )


class FinalHeadPredictionChecks(unittest.TestCase):
    def test_alpha_one_reproduces_rank_one_and_empty_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "exports"
            make_export(folder)
            head = Path(tmp) / "head.npz"
            save_head(head, alpha=1.0, choose_span_candidate=False)

            predictions, summary = predict_export(folder, head)

            self.assertEqual(predictions, {"q1": "base", "q2": ""})
            self.assertEqual(summary["questions_without_candidates"], 1)

    def test_alpha_zero_can_select_using_span_vector(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "exports"
            make_export(folder)
            head = Path(tmp) / "head.npz"
            save_head(head, alpha=0.0, choose_span_candidate=True)

            predictions, _ = predict_export(folder, head)

            self.assertEqual(predictions, {"q1": "head", "q2": ""})


if __name__ == "__main__":
    unittest.main()