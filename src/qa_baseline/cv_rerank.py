"""Article-grouped cross-validation of the reranking variants (a development estimate).

Protocol, fixed before running:
  * 5 folds of whole articles, balanced by question count (seed 42).
  * Outer run k scores fold k, calibrates on fold k+1 (mod 5) and trains on the other three folds.
  * Every setting (length-penalty lambda, head L2 and stopping step, mixing weight alpha) is chosen on the
    calibration fold only. Each question is scored in exactly one run, so it gets one out-of-fold prediction.
  * Variants: length penalty; scalar head and scalar+span head, each alone and mixed with the baseline.
  * Adopt a variant only if (1) the 95% article-bootstrap interval of its EM gain has a lower bound above 0 and
    (2) its F1 gain has a point estimate of at least 0. Folds improved (EM, F1) are reported separately.
  * Not a final test: the earlier 16-article test part has been looked at. Dev stays untouched until the end.

  PYTHONPATH=src python -m qa_baseline.cv_rerank --exports exports/v1/heldout \
      --table exports/v1/heldout/table.npz --validation runs/baseline/validation-v1.1.json \
      --output-dir exports/v1/heldout/cv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .candidate_table import load_table
from .evaluate_rerank import article_bootstrap, article_net, chosen_rows, load_titles
from .span_head import features, group_layout, standardize, standardizer, train_listwise

LAMBDA_GRID = tuple(round(0.01 * i, 2) for i in range(31))
ALPHA_GRID = tuple(round(0.1 * i, 1) for i in range(11))
DEFAULT_L2 = {"scalars": (0.0001, 0.001, 0.01), "scalars+span": (0.03, 0.1, 0.3)}
# Pairs compared head to head. Added AFTER the first cross-validation run, so they are post hoc.
PAIRS = (
    ("scalars+span + mix", "scalars + mix"),
    ("scalars+span + mix", "length penalty"),
    ("scalars + mix", "length penalty"),
)


def assign_folds(question_titles, n_folds=5, seed=42):
    """Fold of every question; whole articles stay together and fold sizes are balanced."""
    sizes = Counter(question_titles)
    order = sorted(sizes, key=lambda title: hashlib.sha256(f"{seed}:{title}".encode()).digest())
    target = len(question_titles) / n_folds
    totals, fold_of = [0] * n_folds, {}
    for title in order:  # each article goes to the fold furthest below its target size
        fold = max(range(n_folds), key=lambda i: target - totals[i])
        fold_of[title] = fold
        totals[fold] += sizes[title]
    return np.array([fold_of[title] for title in question_titles], dtype=np.int8)


def group_softmax(scores, starts, sizes):
    scores = np.asarray(scores, dtype=np.float64)
    e = np.exp(scores - np.repeat(np.maximum.reduceat(scores, starts), sizes))
    return e / np.repeat(np.add.reduceat(e, starts), sizes)


def mixed_scores(base, head, alpha, starts, sizes):
    """(1 - alpha) * baseline probabilities + alpha * head probabilities, per question. alpha 0 is the baseline."""
    return (1 - alpha) * group_softmax(base, starts, sizes) + alpha * group_softmax(head, starts, sizes)


def outcomes(table, rows, row_scores, num_questions):
    """EM and F1 arrays (one entry per question, 0 where the question has no rows) of the top-scored candidate."""
    q_rows = table["q_index"][rows]
    starts, sizes = group_layout(q_rows)
    pick = chosen_rows(np.asarray(row_scores, dtype=np.float64), starts, sizes)
    em, f1 = np.zeros(num_questions), np.zeros(num_questions)
    em[q_rows[pick]] = table["em"][rows][pick]
    f1[q_rows[pick]] = table["f1"][rows][pick]
    return em, f1


def pick_best(options, evaluate):
    """Option with the highest EM; ties go to higher F1, then to the earlier option."""
    best = None
    for option in options:
        em, f1 = evaluate(option)
        key = (round(em, 6), round(f1, 6))
        if best is None or key > best[1]:
            best = (option, key)
    return best[0]


def fit_head(table, config, train_rows, calib_rows, pooled, h_cls, l2_grid, steps, lr):
    """Train on train_rows; choose L2 and the stopping step by loss on calib_rows."""
    X = features(table, train_rows, config, pooled, h_cls)
    mean, std = standardizer(X)
    X = standardize(X, mean, std)
    Xc = standardize(features(table, calib_rows, config, pooled, h_cls), mean, std)
    groups, groups_c = group_layout(table["q_index"][train_rows]), group_layout(table["q_index"][calib_rows])
    correct, correct_c = table["em"][train_rows].astype(np.float64), table["em"][calib_rows].astype(np.float64)
    best = None
    for l2 in l2_grid:
        w, loss, step = train_listwise(X, correct, groups, Xc, correct_c, groups_c, l2, steps, lr)
        if best is None or loss < best["loss"]:
            best = {"w": w, "loss": loss, "step": step, "l2": l2}
    best.update(mean=mean, std=std, calib_scores=Xc @ best["w"])
    return best


def compare_pair(em, f1, a, b, articles, folds):
    """Paired difference a - b over all questions, with an article-level bootstrap interval."""
    result = {}
    for key, arrays in (("em", em), ("f1", f1)):
        diff = arrays[a] - arrays[b]
        mean, low, high, _ = article_bootstrap(diff, articles)
        result[f"d_{key}"] = (100 * mean, 100 * low, 100 * high)
        result[f"folds_{key}"] = int(sum(diff[folds == k].mean() > 0 for k in range(int(folds.max()) + 1)))
    return result


def run_cv(table, pooled, h_cls, question_titles, configs=("scalars", "scalars+span"), n_folds=5, seed=42,
           l2_grids=None, steps=1000, lr=0.02):
    l2_grids = {**DEFAULT_L2, **(l2_grids or {})}
    num_q = int(table["num_questions"])
    folds = assign_folds(question_titles, n_folds, seed)
    row_fold = folds[table["q_index"]]
    names = list(table["scalar_names"])
    base_col = table["scalars"][:, names.index("score")].astype(np.float64)
    tokens_col = table["scalars"][:, names.index("span_tokens")].astype(np.float64)
    variants = ["baseline", "length penalty"] + [v for c in configs for v in (c, f"{c} + mix")]
    em = {v: np.zeros(num_q) for v in variants}
    f1 = {v: np.zeros(num_q) for v in variants}
    scored = np.zeros(num_q, dtype=int)
    settings, sizes_used = [], []

    def metrics(result, questions):
        return 100 * result[0][questions].mean(), 100 * result[1][questions].mean()

    for k in range(n_folds):
        calib_fold = (k + 1) % n_folds
        score_rows, calib_rows = np.where(row_fold == k)[0], np.where(row_fold == calib_fold)[0]
        train_rows = np.where((row_fold != k) & (row_fold != calib_fold))[0]
        score_q, calib_q = np.where(folds == k)[0], np.where(folds == calib_fold)[0]
        train_questions = int(((folds != k) & (folds != calib_fold)).sum())
        scored[score_q] += 1
        sizes_used.append((train_questions, len(calib_q), len(score_q)))
        chosen = {"fold": k, "calibration_fold": calib_fold}

        def record(variant, rows, row_scores):
            result = outcomes(table, rows, row_scores, num_q)
            em[variant][score_q], f1[variant][score_q] = result[0][score_q], result[1][score_q]

        record("baseline", score_rows, -table["rank"][score_rows].astype(np.float64))

        lam = pick_best(LAMBDA_GRID, lambda lam: metrics(
            outcomes(table, calib_rows, base_col[calib_rows] - lam * tokens_col[calib_rows], num_q), calib_q))
        record("length penalty", score_rows, base_col[score_rows] - lam * tokens_col[score_rows])
        chosen["lambda"] = lam

        starts_c, sizes_c = group_layout(table["q_index"][calib_rows])
        starts_s, sizes_s = group_layout(table["q_index"][score_rows])
        for config in configs:
            head = fit_head(table, config, train_rows, calib_rows, pooled, h_cls, l2_grids[config], steps, lr)
            score_head = standardize(features(table, score_rows, config, pooled, h_cls), head["mean"], head["std"]) @ head["w"]
            record(config, score_rows, score_head)
            alpha = pick_best(ALPHA_GRID, lambda a: metrics(outcomes(
                table, calib_rows, mixed_scores(base_col[calib_rows], head["calib_scores"], a, starts_c, sizes_c), num_q), calib_q))
            record(f"{config} + mix", score_rows,
                   mixed_scores(base_col[score_rows], score_head, alpha, starts_s, sizes_s))
            chosen[config] = {"l2": head["l2"], "step": head["step"], "alpha": alpha}
        settings.append(chosen)

    articles = np.array(question_titles)
    report = {}
    base_em, base_f1 = em["baseline"], f1["baseline"]
    for variant in variants:
        entry = {"em": 100 * float(em[variant].mean()), "f1": 100 * float(f1[variant].mean())}
        if variant != "baseline":
            for key, new, old in (("em", em[variant], base_em), ("f1", f1[variant], base_f1)):
                mean, low, high, _ = article_bootstrap(new - old, articles)
                entry[f"d_{key}"] = (100 * mean, 100 * low, 100 * high)
                entry[f"folds_improved_{key}"] = int(sum(
                    (new[folds == k] - old[folds == k]).mean() > 0 for k in range(n_folds)))
            entry["helped"] = int(((em[variant] == 1) & (base_em == 0)).sum())
            entry["hurt"] = int(((em[variant] == 0) & (base_em == 1)).sum())
            entry["articles_up_down_same"] = article_net(em[variant] - base_em, articles)
            entry["adopt"] = bool(entry["d_em"][1] > 0 and entry["d_f1"][0] >= 0)
        report[variant] = entry
    pairs = {f"{a} vs {b}": compare_pair(em, f1, a, b, articles, folds)
             for a, b in PAIRS if a in em and b in em}
    return {"report": report, "settings": settings, "fold_sizes": sizes_used, "pairs": pairs,
            "scored_once": bool((scored == 1).all()), "n_articles": len(set(question_titles)),
            "outcomes": {"folds": folds, "em": em, "f1": f1}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--configs", default="scalars,scalars+span")
    parser.add_argument("--l2-scalars", default=None, help="comma-separated L2 grid for the scalar head")
    parser.add_argument("--l2-span", default=None, help="comma-separated L2 grid for the scalars+span head")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.02)
    args = parser.parse_args()

    table = load_table(args.table)
    pooled = np.load(args.exports / "span_pooled.npy", mmap_mode="r")
    h_cls = np.load(args.exports / "windows.npz")["h_cls"]
    ids = [q["id"] for q in json.loads((args.exports / "questions.json").read_text(encoding="utf-8"))]
    titles = load_titles(args.validation)
    grids = {}
    if args.l2_scalars:
        grids["scalars"] = tuple(float(v) for v in args.l2_scalars.split(","))
    if args.l2_span:
        grids["scalars+span"] = tuple(float(v) for v in args.l2_span.split(","))
    result = run_cv(table, pooled, h_cls, [titles[i] for i in ids], tuple(args.configs.split(",")),
                    args.folds, l2_grids=grids, steps=args.steps, lr=args.lr)

    train_n, calib_n, score_n = (int(np.mean(col)) for col in zip(*result["fold_sizes"]))
    print(f"{args.folds} folds of whole articles ({result['n_articles']} articles). Each run trains on about "
          f"{train_n} questions, calibrates on about {calib_n}, scores about {score_n}. "
          f"Every question scored exactly once: {result['scored_once']}")
    for setting in result["settings"]:
        print("  fold", setting["fold"], "(calibrated on", f"{setting['calibration_fold']}):",
              {k: v for k, v in setting.items() if k not in ("fold", "calibration_fold")})
    base = result["report"]["baseline"]
    print(f"\nbaseline: EM {base['em']:.2f}  F1 {base['f1']:.2f}")
    for name, e in result["report"].items():
        if name == "baseline":
            continue
        print(f"{name:22s} EM {e['em']:.2f} F1 {e['f1']:.2f}  dEM {e['d_em'][0]:+.2f} [{e['d_em'][1]:+.2f}, {e['d_em'][2]:+.2f}]"
              f"  dF1 {e['d_f1'][0]:+.2f} [{e['d_f1'][1]:+.2f}, {e['d_f1'][2]:+.2f}]"
              f"  folds up EM {e['folds_improved_em']}/{args.folds} F1 {e['folds_improved_f1']}/{args.folds}"
              f"  helped {e['helped']} hurt {e['hurt']}  articles up/down/same {e['articles_up_down_same']}"
              f"  -> {'ADOPT' if e['adopt'] else 'no'}")
    if result["pairs"]:
        print("\nPaired comparisons (post hoc: chosen after seeing the first run), first minus second:")
        for name, e in result["pairs"].items():
            print(f"  {name:45s} dEM {e['d_em'][0]:+.2f} [{e['d_em'][1]:+.2f}, {e['d_em'][2]:+.2f}] folds up {e['folds_em']}/{args.folds}"
                  f"   dF1 {e['d_f1'][0]:+.2f} [{e['d_f1'][1]:+.2f}, {e['d_f1'][2]:+.2f}] folds up {e['folds_f1']}/{args.folds}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outcomes = result.pop("outcomes")
    slug = lambda name: re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    np.savez_compressed(args.output_dir / "cv_outcomes.npz", folds=outcomes["folds"],
                        **{f"em_{slug(v)}": a for v, a in outcomes["em"].items()},
                        **{f"f1_{slug(v)}": a for v, a in outcomes["f1"].items()})
    (args.output_dir / "cv_summary.json").write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()