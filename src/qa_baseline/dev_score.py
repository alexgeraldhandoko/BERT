"""Score the frozen settings on dev, once, and write predictions for the official scorer.

Declared before running: the primary candidate is the frozen head mixed with the baseline; the reference is the
length penalty. The primary is adopted over the baseline only if its EM gain has a 95% article-bootstrap interval
above 0 and its F1 gain has a point estimate of at least 0. If the primary cannot beat the penalty (the interval of
the paired EM difference includes 0), prefer the penalty.

  PYTHONPATH=src python -m qa_baseline.dev_score --exports exports/v1/dev --table exports/v1/dev/table.npz \
      --text-features exports/v1/dev/text_features.npz --dev data/raw/dev-v1.1.json \
      --final exports/v1/final --output-dir exports/v1/final/dev
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .candidate_table import PARTS, load_table
from .cv_rerank import mixed_scores
from .evaluate_rerank import article_bootstrap, evaluate_methods, load_titles
from .final_head import load_final
from .span_head import features, group_layout, standardize

PRIMARY = "frozen head + mix"


def head_scores_chunked(table, pooled, final, chunk=20000):
    """The frozen head's score for every candidate row, computed in chunks to bound memory."""
    n = len(table["q_index"])
    out = np.zeros(n)
    for start in range(0, n, chunk):
        rows = np.arange(start, min(start + chunk, n))
        X = standardize(features(table, rows, final["config"], pooled, None), final["mean"], final["std"])
        out[start : start + len(rows)] = X @ final["w"]
    return out


def build_scores(table, pooled, final):
    names = list(table["scalar_names"])
    base = table["scalars"][:, names.index("score")].astype(np.float64)
    tokens = table["scalars"][:, names.index("span_tokens")].astype(np.float64)
    starts, sizes = group_layout(table["q_index"])
    head = head_scores_chunked(table, pooled, final)
    return {
        "baseline": -table["rank"].astype(np.float64),
        "length penalty": base - final["lambda"] * tokens,
        PRIMARY: mixed_scores(base, head, final["alpha"], starts, sizes),
    }


def score_dev(table, pooled, final, articles):
    """Report vs baseline for each method, per-method outcomes, and the paired primary-minus-penalty difference."""
    number = PARTS.index("test")
    report, outcomes = evaluate_methods(table, number, build_scores(table, pooled, final), articles)
    paired = {}
    for key, column in (("em", 1), ("f1", 2)):
        mean, low, high, _ = article_bootstrap(outcomes[PRIMARY][column] - outcomes["length penalty"][column], articles)
        paired[f"d_{key}"] = (100 * mean, 100 * low, 100 * high)
    return report, outcomes, paired


def decide(report, paired):
    primary = report[PRIMARY]
    return {"adopt_over_baseline": bool(primary["d_em"][1] > 0 and primary["d_f1"][0] >= 0),
            "beats_penalty": bool(paired["d_em"][1] > 0)}


def answers_for(table, picked_rows, texts, question_ids):
    answers = {qid: "" for qid in question_ids}
    for row in picked_rows:
        answers[str(table["qid"][row])] = texts[row]
    return answers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from .text_features import load_text_features
    table = load_table(args.table)
    table["text"] = load_text_features(args.text_features, len(table["q_index"]))
    pooled = np.load(args.exports / "span_pooled.npy", mmap_mode="r")
    final = load_final(args.final)
    ids = [q["id"] for q in json.loads((args.exports / "questions.json").read_text(encoding="utf-8"))]
    titles = load_titles(args.dev)
    number = PARTS.index("test")
    questions = np.where(table["question_part"] == number)[0]
    articles = np.array([titles[ids[q]] for q in questions])

    report, outcomes, paired = score_dev(table, pooled, final, articles)
    verdict = decide(report, paired)
    print(f"Dev: {len(questions)} questions, {len(set(articles))} articles. "
          f"Frozen: {final['config']}, l2={final['l2']:g}, step={final['step']}, alpha={final['alpha']:g}, lambda={final['lambda']:g}")
    for name, e in report.items():
        line = f"{name:18s} EM {e['em']:.2f}  F1 {e['f1']:.2f}"
        if name != "baseline":
            d, f = e["d_em"], e["d_f1"]
            line += (f"   dEM {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]  dF1 {f[0]:+.2f} [{f[1]:+.2f}, {f[2]:+.2f}]"
                     f"   changed {e['changed']} helped {e['helped']} hurt {e['hurt']}  articles up/down/same {e['articles_up_down_same']}")
        print(line)
    d, f = paired["d_em"], paired["d_f1"]
    print(f"\n{PRIMARY} minus length penalty: dEM {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]  dF1 {f[0]:+.2f} [{f[1]:+.2f}, {f[2]:+.2f}]")
    print(f"Declared rule: adopt over baseline = {verdict['adopt_over_baseline']}; beats the penalty = {verdict['beats_penalty']}"
          + ("" if verdict["beats_penalty"] else "  -> prefer the length penalty"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.exports / "candidates.jsonl").open(encoding="utf-8") as stream:
        texts = [json.loads(line)["text"] for line in stream]
    for name, (_, _, _, picked) in outcomes.items():
        answers = answers_for(table, picked, texts, [ids[q] for q in questions])
        slug = name.replace(" ", "-").replace("+", "plus")
        (args.output_dir / f"predictions-{slug}.json").write_text(json.dumps(answers, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "dev_summary.json").write_text(
        json.dumps({"report": report, "paired": paired, "verdict": verdict, "frozen": {
            k: final[k] for k in ("config", "l2", "step", "alpha", "lambda")}}, indent=2, default=float) + "\n",
        encoding="utf-8")
    print("\nCheck each prediction file with the official scorer, for example:")
    print(f"  PYTHONPATH=src python -m qa_baseline.evaluate --data {args.dev} "
          f"--predictions {args.output_dir}/predictions-baseline.json --output {args.output_dir}/metrics-baseline.json")


if __name__ == "__main__":
    main()