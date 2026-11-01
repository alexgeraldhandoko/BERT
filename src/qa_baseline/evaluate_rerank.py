"""Compare the baseline, a length penalty and the scalar head on the test part.

The penalty's lambda is chosen on head_train + calib only. The test part is then
scored once per method, and each gain is judged with a bootstrap that resamples whole articles.

  PYTHONPATH=src python -m qa_baseline.evaluate_rerank \
      --exports exports/v1/heldout --table exports/v1/heldout/table.npz \
      --validation runs/baseline/validation-v1.1.json \
      --head exports/v1/heldout/heads_v2/head_scalars.npz \
      --output-dir exports/v1/heldout/rerank --write-predictions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .candidate_table import PARTS, load_table
from .data import write_squad_subset
from .span_head import part_layout, standardize

LAMBDA_GRID = tuple(round(0.01 * i, 2) for i in range(31))
TUNING_PARTS = ("head_train", "calib")


def chosen_rows(scores, starts, sizes):
    """Index of the best-scoring row of each question; ties pick the earlier row (rank 1 comes first)."""
    return np.array([a + int(np.argmax(scores[a : a + n])) for a, n in zip(starts, sizes)], dtype=np.int64)


def part_outcomes(table, number, row_scores):
    """EM and F1 of the top-scored candidate for every question of a part.

    row_scores aligns with the part's rows. Returns (question indices, em, f1, table rows chosen).
    Questions without any candidate score 0 and have no chosen row."""
    rows, (starts, sizes), _ = part_layout(table, number)
    pick = chosen_rows(np.asarray(row_scores, dtype=np.float64), starts, sizes)
    questions = np.where(table["question_part"] == number)[0]
    em_all = np.zeros(int(table["num_questions"]))
    f1_all = np.zeros_like(em_all)
    q = table["q_index"][rows][pick]
    em_all[q] = table["em"][rows][pick]
    f1_all[q] = table["f1"][rows][pick]
    return questions, em_all[questions], f1_all[questions], rows[pick]


def baseline_scores(table, rows):
    return -table["rank"][rows].astype(np.float64)


def penalty_scores(table, rows, lam):
    names = list(table["scalar_names"])
    s = table["scalars"][rows].astype(np.float64)
    return s[:, names.index("score")] - lam * s[:, names.index("span_tokens")]


def head_scores(table, rows, head):
    if head["w"].shape[0] != table["scalars"].shape[1]:
        raise ValueError("This head was not trained on the scalar features alone")
    return standardize(table["scalars"][rows], head["mean"], head["std"]) @ head["w"]


def select_lambda(table, grid=LAMBDA_GRID, parts=TUNING_PARTS):
    """Pick lambda by EM on the tuning parts (ties: higher F1, then the smaller lambda)."""
    curve = []
    for lam in grid:
        em_sum = f1_sum = count = 0.0
        for name in parts:
            number = PARTS.index(name)
            rows = np.where(table["part"] == number)[0]
            _, em, f1, _ = part_outcomes(table, number, penalty_scores(table, rows, lam))
            em_sum, f1_sum, count = em_sum + em.sum(), f1_sum + f1.sum(), count + len(em)
        curve.append((lam, 100 * em_sum / count, 100 * f1_sum / count))
    best = max(curve, key=lambda c: (round(c[1], 6), round(c[2], 6), -c[0]))
    return best[0], curve


def article_bootstrap(diff, articles, reps=10000, seed=0):
    """Mean per-question difference with a 95% interval from resampling whole articles.

    Returns (mean, low, high, share of resamples above zero)."""
    diff = np.asarray(diff, dtype=np.float64)
    _, inverse = np.unique(np.asarray(articles), return_inverse=True)
    n_articles = int(inverse.max()) + 1
    sums = np.bincount(inverse, weights=diff, minlength=n_articles)
    counts = np.bincount(inverse, minlength=n_articles).astype(np.float64)
    draws = np.random.default_rng(seed).integers(0, n_articles, size=(reps, n_articles))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high), float((means > 0).mean())


def article_net(diff, articles):
    """Articles where the net number of questions gained is positive, negative, or zero."""
    _, inverse = np.unique(np.asarray(articles), return_inverse=True)
    net = np.bincount(inverse, weights=np.asarray(diff, dtype=np.float64))
    return int((net > 0).sum()), int((net < 0).sum()), int((net == 0).sum())


def load_titles(validation_path):
    document = json.loads(Path(validation_path).read_text(encoding="utf-8"))
    return {
        qa["id"]: article["title"]
        for article in document["data"]
        for paragraph in article["paragraphs"]
        for qa in paragraph["qas"]
    }


def evaluate_methods(table, number, scores, articles):
    """Score each method on one part and compare it with scores["baseline"] (which must be present)."""
    outcomes = {name: part_outcomes(table, number, row_scores) for name, row_scores in scores.items()}
    _, base_em, base_f1, base_rows = outcomes["baseline"]
    report = {}
    for name, (_, em, f1, picked) in outcomes.items():
        entry = {"em": 100 * float(em.mean()), "f1": 100 * float(f1.mean())}
        if name != "baseline":
            for key, new, old in (("em", em, base_em), ("f1", f1, base_f1)):
                mean, low, high, share = article_bootstrap(new - old, articles)
                entry[f"d_{key}"] = (100 * mean, 100 * low, 100 * high)
                entry[f"share_{key}_above_0"] = share
            entry["changed"] = int((picked != base_rows).sum())
            entry["helped"] = int(((em == 1) & (base_em == 0)).sum())
            entry["hurt"] = int(((em == 0) & (base_em == 1)).sum())
            entry["articles_up_down_same"] = article_net(em - base_em, articles)
        report[name] = entry
    return report, outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-predictions", action="store_true")
    args = parser.parse_args()

    table = load_table(args.table)
    ids = [question["id"] for question in json.loads((args.exports / "questions.json").read_text(encoding="utf-8"))]
    titles = load_titles(args.validation)

    lam, curve = select_lambda(table)
    print("Length penalty on head_train + calib:")
    for value, em, f1 in curve:
        print(f"  lambda {value:.2f}  EM {em:.2f}  F1 {f1:.2f}" + ("   <-- chosen" if value == lam else ""))

    number = PARTS.index("test")
    rows = np.where(table["part"] == number)[0]
    scores = {
        "baseline": baseline_scores(table, rows),
        "length penalty": penalty_scores(table, rows, lam),
        "scalar head": head_scores(table, rows, np.load(args.head)),
    }
    questions = np.where(table["question_part"] == number)[0]
    articles = np.array([titles[ids[q]] for q in questions])
    report, outcomes = evaluate_methods(table, number, scores, articles)

    print(f"\nTest part: {len(questions)} questions, {len(set(articles))} articles (penalty lambda = {lam})")
    for name, entry in report.items():
        line = f"{name:15s} EM {entry['em']:.2f}  F1 {entry['f1']:.2f}"
        if name != "baseline":
            d, f = entry["d_em"], entry["d_f1"]
            line += (
                f"   dEM {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]  dF1 {f[0]:+.2f} [{f[1]:+.2f}, {f[2]:+.2f}]"
                f"   changed {entry['changed']} helped {entry['helped']} hurt {entry['hurt']}"
                f"   articles up/down/same {entry['articles_up_down_same']}"
            )
        print(line)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"lambda": lam, "curve": curve, "test": report}, indent=2) + "\n", encoding="utf-8"
    )
    if args.write_predictions:
        with (args.exports / "candidates.jsonl").open(encoding="utf-8") as stream:
            texts = [json.loads(line)["text"] for line in stream]
        test_ids = [ids[q] for q in questions]
        write_squad_subset(args.validation, set(test_ids), args.output_dir / "test-v1.1.json")
        for name, (_, _, _, picked) in outcomes.items():
            answers = {qid: "" for qid in test_ids}
            for row in picked:
                answers[str(table["qid"][row])] = texts[row]
            slug = name.replace(" ", "-")
            (args.output_dir / f"predictions-{slug}.json").write_text(json.dumps(answers, indent=2) + "\n", encoding="utf-8")
        print("\nCheck with the official scorer, for example:")
        print(f"  PYTHONPATH=src python -m qa_baseline.evaluate --data {args.output_dir}/test-v1.1.json "
              f"--predictions {args.output_dir}/predictions-baseline.json --output {args.output_dir}/metrics-baseline.json")


if __name__ == "__main__":
    main()