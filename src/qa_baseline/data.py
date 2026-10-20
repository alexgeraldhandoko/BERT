"""Load the course SQuAD files and make a stable train/validation split."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


def load_squad(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        document = json.load(stream)
    if str(document.get("version")) != "1.1":
        raise ValueError(f"Expected SQuAD 1.1 in {path}")

    examples = []
    seen_ids = set()
    for article in document["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                qid = qa["id"]
                if qid in seen_ids:
                    raise ValueError(f"Duplicate question ID: {qid}")
                seen_ids.add(qid)
                answers = qa["answers"]
                if not answers:
                    raise ValueError(f"SQuAD 1.1 question has no answer: {qid}")
                examples.append(
                    {
                        "id": qid,
                        "article_title": article["title"],
                        "question": qa["question"],
                        "context": context,
                        "answers": answers,
                    }
                )
    return examples


def split_train(
    examples: list[dict], validation_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Hash article titles so related contexts stay in one partition."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    article_sizes = Counter(example["article_title"] for example in examples)
    if len(article_sizes) < 2:
        raise ValueError("Need at least two articles for an article-level split")
    ordered_articles = sorted(
        article_sizes,
        key=lambda title: hashlib.sha256(f"{seed}:{title}".encode()).digest(),
    )
    target = len(examples) * validation_fraction
    running = 0
    best_count, best_difference = 1, float("inf")
    for count, title in enumerate(ordered_articles[:-1], start=1):
        running += article_sizes[title]
        difference = abs(running - target)
        if difference < best_difference:
            best_count, best_difference = count, difference
    validation_articles = set(ordered_articles[:best_count])
    train, validation = [], []
    for example in examples:
        (validation if example["article_title"] in validation_articles else train).append(example)
    if not train or not validation:
        raise ValueError("Split produced an empty partition")
    return train, validation


def write_squad_subset(source: str | Path, ids: set[str], destination: str | Path) -> None:
    """Preserve official SQuAD JSON shape for scoring an internal split."""
    with Path(source).open(encoding="utf-8") as stream:
        document = json.load(stream)
    articles = []
    for article in document["data"]:
        paragraphs = []
        for paragraph in article["paragraphs"]:
            qas = [qa for qa in paragraph["qas"] if qa["id"] in ids]
            if qas:
                paragraphs.append({**paragraph, "qas": qas})
        if paragraphs:
            articles.append({**article, "paragraphs": paragraphs})
    subset = {**document, "data": articles}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(subset, ensure_ascii=False) + "\n", encoding="utf-8")

