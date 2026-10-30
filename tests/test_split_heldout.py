import random
import unittest

from qa_baseline.split_heldout import PARTS, split_articles


def make_document(sizes):
    return {
        "version": "1.1",
        "data": [
            {
                "title": title,
                "paragraphs": [{"context": "c", "qas": [{"id": f"{title}-{i}"} for i in range(count)]}],
            }
            for title, count in sizes
        ],
    }


class SplitHeldoutChecks(unittest.TestCase):
    def setUp(self):
        rng = random.Random(1)
        self.sizes = [(f"article{i}", rng.randint(100, 250)) for i in range(50)]
        self.document = make_document(self.sizes)

    def test_every_question_lands_in_exactly_one_part(self):
        parts = split_articles(self.document)
        everything = [qid for name in PARTS for qid in parts[name]["ids"]]
        expected = [f"{title}-{i}" for title, count in self.sizes for i in range(count)]
        self.assertEqual(sorted(everything), sorted(expected))
        self.assertEqual(len(everything), len(set(everything)))

    def test_articles_are_never_shared_between_parts(self):
        parts = split_articles(self.document)
        seen = set()
        for name in PARTS:
            titles = set(parts[name]["articles"])
            self.assertFalse(titles & seen)
            seen |= titles
        self.assertEqual(len(seen), len(self.sizes))

    def test_parts_are_roughly_equal(self):
        parts = split_articles(self.document)
        total = sum(count for _, count in self.sizes)
        biggest_article = max(count for _, count in self.sizes)
        for name in PARTS:
            self.assertLessEqual(abs(len(parts[name]["ids"]) - total / 3), biggest_article)

    def test_split_is_stable_when_article_order_changes(self):
        shuffled = make_document(random.Random(7).sample(self.sizes, len(self.sizes)))
        a, b = split_articles(self.document), split_articles(shuffled)
        for name in PARTS:
            self.assertEqual(set(a[name]["ids"]), set(b[name]["ids"]))

    def test_seed_changes_the_split(self):
        a, b = split_articles(self.document, seed=1), split_articles(self.document, seed=2)
        self.assertNotEqual(set(a["test"]["ids"]), set(b["test"]["ids"]))


if __name__ == "__main__":
    unittest.main()