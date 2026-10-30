"""Split the held-out validation questions into three parts by article.
 
Questions on one article share passages, so the parts must not share articles.
Parts: 
    1. head_train (fit the span head)
    2. calib (tune alpha and temperatures)
    3. test (report results)
 
  PYTHONPATH=src python -m qa_baseline.split_heldout \
      --data runs/baseline/validation-v1.1.json --output splits/heldout-v1.json
"""


from __future__ import annotations
 
import argparse
import hashlib
import json
from pathlib import Path
 
PARTS = ("head_train", "calib", "test")
 
def split_articles(doc, frac=(1/3, 1/3, 1/3), seed=42):
    """
    Split the articles in a document into three parts.
 
    Args:
        doc: A SQuAD-style document with 'data' field containing articles.
        frac: Fractions of articles to assign to each part.
        seed: Random seed for reproducibility.
        
    Returns:
        A dictionary mapping part names to dictionaries containing 'articles' and 'ids'.
    """
    sizes = {}
    # Count the number of questions in each article
    for article in doc["data"]:
        count = sum(len(paragraph["qas"]) for paragraph in article["paragraphs"])
        sizes[article["title"]] = sizes.get(article["title"], 0) + count
    
    total = sum(sizes.values())
    targets = [total * f for f in frac]
    ordered_articles = sorted(sizes, key=lambda x: hashlib.sha256(f"{seed}{x}".encode()).digest())
    counts = [0] * len(PARTS)
    part_of = {}
    
    # Assign articles to parts that is furthest below its target
    for title in ordered_articles:
        index = max(range(len(PARTS)), key=lambda i: targets[i] - counts[i])
        part_of[title] = index
        counts[index] += sizes[title]
    result = {name: {"articles": [], "ids": []} for name in PARTS}
    
    for article in doc["data"]:
        part = result[PARTS[part_of[article["title"]]]]
        if article["title"] not in part["articles"]:
            part["articles"].append(article["title"])
        for paragraph in article["paragraphs"]:
            part["ids"].extend(qa["id"] for qa in paragraph["qas"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    document = json.loads(args.data.read_text(encoding="utf-8"))
    parts = split_articles(document, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"seed": args.seed, "source": str(args.data), "parts": parts}) + "\n", encoding="utf-8"
    )
    for name in PARTS:
        print(f"{name:10s} {len(parts[name]['ids']):5d} questions, {len(parts[name]['articles']):3d} articles")
 
 
if __name__ == "__main__":
    main()
 