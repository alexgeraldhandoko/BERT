import unittest

from qa_baseline.boundary_report import classify, find, fixable_errors


class BoundaryReportChecks(unittest.TestCase):
    def test_find(self):
        self.assertEqual(find(["in", "1912"], ["1912"]), 1)
        self.assertEqual(find(["a", "b"], ["c"]), -1)
        self.assertEqual(find(["a"], []), -1)

    def test_categories(self):
        self.assertEqual(classify(["in", "1912"], ["1912"]), ("too long: extra leading words", ["in"], []))
        self.assertEqual(classify(["1912", "and", "later"], ["1912"])[0], "too long: extra trailing words")
        self.assertEqual(classify(["in", "1912", "and", "later"], ["1912"])[0], "too long: extra words on both sides")
        self.assertEqual(classify(["denver"], ["denver", "broncos"]), ("too short: missing words", [], ["broncos"]))
        self.assertEqual(classify(["red", "car"], ["blue", "car"])[0], "shifted: partial overlap")
        self.assertEqual(classify(["paris"], ["rome"])[0], "different answer: no overlap")
        self.assertEqual(classify([], ["rome"])[0], "empty answer")

    def test_fixable_errors_need_a_wrong_rank_one_and_a_correct_candidate_in_range(self):
        def row(qid, rank, em, text):
            return {"qid": qid, "rank": rank, "em": em, "text": text}
        rows = [
            row("a", 1, 0, "in 1912"), row("a", 2, 1, "1912"),          # fixable at rank 2
            row("b", 1, 1, "paris"), row("b", 2, 0, "lyon"),            # rank 1 already right
            row("c", 1, 0, "x"), row("c", 2, 0, "y"), row("c", 6, 1, "z"),  # correct only beyond the top 5
            row("d", 1, 0, "foo"), row("d", 2, 0, "bar"),               # no correct candidate at all
        ]
        self.assertEqual(list(fixable_errors(rows, 5)), [("a", "in 1912", "1912", 2)])


if __name__ == "__main__":
    unittest.main()