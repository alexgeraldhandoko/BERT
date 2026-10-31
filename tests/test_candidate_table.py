import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from qa_baseline.candidate_table import PARTS, SCALAR_NAMES, build_table, summarize, vectors_for

DIM = 5
# (qid, rank, window, em, score)
CANDIDATES = [
    ("q0", 1, 0, 1, 9.0), ("q0", 2, 0, 0, 7.0), ("q0", 3, 0, 0, 5.0),
    ("q1", 1, 1, 0, 8.0), ("q1", 2, 1, 1, 6.5), ("q1", 3, 1, 0, 4.0),
    ("q2", 1, 2, 0, 3.0), ("q2", 2, 2, 0, 2.0),
    ("q3", 1, 3, 1, 6.0), ("q3", 2, 4, 0, 1.0),
]
SPLIT = {"head_train": ["q0", "q1"], "calib": ["q2"], "test": ["q3", "q4"]}  # q4 has no candidates


def write_export(folder, candidates=CANDIDATES, split=SPLIT, pooled_rows=None):
    folder = Path(folder)
    qids = ["q0", "q1", "q2", "q3", "q4"]
    (folder / "questions.json").write_text(json.dumps([{"id": q} for q in qids]), encoding="utf-8")
    with (folder / "candidates.jsonl").open("w", encoding="utf-8") as stream:
        for qid, rank, window, em, score in candidates:
            stream.write(json.dumps({
                "qid": qid, "rank": rank, "window": window, "tok_start": 10, "tok_end": 10 + rank,
                "char_start": 100, "char_end": 100 + 5 * rank, "text": "x", "score": score,
                "start_logit": score / 2, "end_logit": score / 2, "logp_all": -rank, "logp_ctx": -rank / 2,
                "em": em, "f1": float(em),
            }) + "\n")
    n = len(candidates) if pooled_rows is None else pooled_rows
    np.save(folder / "span_pooled.npy", np.repeat(np.arange(n)[:, None], DIM, axis=1).astype(np.float16))
    windows = 5
    stats = {name: np.arange(windows, dtype=np.float32) * scale for name, scale in
             (("ent_ctx_start", 1), ("ent_ctx_end", 2), ("cls_start", 3), ("cls_end", 4))}
    h_cls = (1000 + np.repeat(np.arange(windows)[:, None], DIM, axis=1)).astype(np.float16)
    np.savez(folder / "windows.npz", h_cls=h_cls, **stats)
    split_path = folder / "split.json"
    split_path.write_text(json.dumps({"parts": {name: {"ids": split[name]} for name in PARTS}}), encoding="utf-8")
    return split_path


class CandidateTableChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.split_path = write_export(self.folder)
        self.table = build_table(self.folder, self.split_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_parts_and_labels_follow_the_files(self):
        self.assertEqual(self.table["part"].tolist(), [0, 0, 0, 0, 0, 0, 1, 1, 2, 2])
        self.assertEqual(self.table["em"].tolist(), [1, 0, 0, 0, 1, 0, 0, 0, 1, 0])
        self.assertEqual(self.table["q_index"].tolist(), [0, 0, 0, 1, 1, 1, 2, 2, 3, 3])

    def test_gap_features_are_relative_to_rank_one(self):
        names = list(self.table["scalar_names"])
        gap = self.table["scalars"][:, names.index("score_gap")]
        self.assertEqual(gap.tolist(), [0, 2, 4, 0, 1.5, 4, 0, 1, 0, 5])
        self.assertEqual(self.table["scalars"].shape, (10, len(SCALAR_NAMES)))

    def test_window_statistics_are_looked_up_by_candidate_window(self):
        names = list(self.table["scalar_names"])
        cls_end = self.table["scalars"][:, names.index("cls_end")]
        self.assertEqual(cls_end.tolist(), [0, 0, 0, 4, 4, 4, 8, 8, 12, 16])

    def test_summary_matches_hand_counts(self):
        result = summarize(self.table, ks=(1, 3))
        self.assertAlmostEqual(result["head_train"]["em@1"], 50.0)   # q0 right, q1 wrong
        self.assertAlmostEqual(result["head_train"]["em@3"], 100.0)  # q1's answer is rank 2
        self.assertAlmostEqual(result["calib"]["em@3"], 0.0)
        self.assertAlmostEqual(result["test"]["em@1"], 50.0)         # q3 right, q4 has no candidates
        self.assertEqual(result["test"]["questions"], 2)
        self.assertEqual(result["test"]["no_candidates"], 1)
        self.assertEqual(result["head_train"]["no_candidates"], 0)
        self.assertEqual(result["head_train"]["candidates"], 6)

    def test_vectors_follow_row_and_window(self):
        pooled = np.load(self.folder / "span_pooled.npy")
        h_cls = np.load(self.folder / "windows.npz")["h_cls"]
        rows = np.array([4, 9])
        vectors = vectors_for(self.table, rows, pooled, h_cls)
        self.assertEqual(vectors.shape, (2, 2 * DIM))
        self.assertTrue((vectors[0, :DIM] == 4).all() and (vectors[0, DIM:] == 1001).all())
        self.assertTrue((vectors[1, :DIM] == 9).all() and (vectors[1, DIM:] == 1004).all())

    def test_row_count_mismatch_is_rejected(self):
        write_export(self.folder, pooled_rows=7)
        with self.assertRaises(ValueError):
            build_table(self.folder, self.split_path)

    def test_question_missing_from_split_is_rejected(self):
        bad = {"head_train": ["q0", "q1"], "calib": ["q2"], "test": ["q4"]}
        path = write_export(self.folder, split=bad)
        with self.assertRaises(ValueError):
            build_table(self.folder, path)

    def test_question_in_two_parts_is_rejected(self):
        bad = {"head_train": ["q0", "q1"], "calib": ["q2", "q0"], "test": ["q3", "q4"]}
        path = write_export(self.folder, split=bad)
        with self.assertRaises(ValueError):
            build_table(self.folder, path)


if __name__ == "__main__":
    unittest.main()