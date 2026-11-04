"""Freeze the declared candidate head and the length penalty BEFORE dev is scored.

Protocol (declared in advance):
  * The same 5 article folds as cv_rerank (seed 42). Train on folds 1-4, calibrate on fold 0.
  * L2 and stopping step: lowest loss on the calibration fold. Mixing weight alpha: best EM on that fold.
  * Length-penalty lambda: best EM over all held-out questions (ties: higher F1, then smaller lambda).
  * Outputs final_head.npz (weights and settings) and final_settings.json. Commit and tag both before running
    dev_score, so the dev run provably used frozen settings.

  PYTHONPATH=src python -m qa_baseline.final_head --exports exports/v1/heldout \
      --table exports/v1/heldout/table.npz --text-features exports/v1/heldout/text_features.npz \
      --validation runs/baseline/validation-v1.1.json --output-dir exports/v1/final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .candidate_table import load_table
from .cv_rerank import ALPHA_GRID, DEFAULT_L2, LAMBDA_GRID, assign_folds, fit_head, mixed_scores, outcomes, pick_best
from .evaluate_rerank import load_titles
from .span_head import group_layout

ARRAYS = ("w", "mean", "std")


def em_f1(result, questions):
    return 100 * float(result[0][questions].mean()), 100 * float(result[1][questions].mean())


def freeze(table, pooled, question_titles, config="scalars+text+span", n_folds=5, seed=42, calib_fold=0,
           l2_grid=None, steps=1000, lr=0.02):
    """Train on every fold but calib_fold; choose L2, step and alpha there; choose lambda on all questions."""
    num_q = int(table["num_questions"])
    folds = assign_folds(question_titles, n_folds, seed)
    row_fold = folds[table["q_index"]]
    calib_rows, train_rows = np.where(row_fold == calib_fold)[0], np.where(row_fold != calib_fold)[0]
    calib_q = np.where(folds == calib_fold)[0]
    names = list(table["scalar_names"])
    base = table["scalars"][:, names.index("score")].astype(np.float64)
    tokens = table["scalars"][:, names.index("span_tokens")].astype(np.float64)

    head = fit_head(table, config, train_rows, calib_rows, pooled, None, l2_grid or DEFAULT_L2[config], steps, lr)
    starts, sizes = group_layout(table["q_index"][calib_rows])
    mix = lambda a: mixed_scores(base[calib_rows], head["calib_scores"], a, starts, sizes)
    alpha = pick_best(ALPHA_GRID, lambda a: em_f1(outcomes(table, calib_rows, mix(a), num_q), calib_q))
    everyone = np.arange(len(table["q_index"]))
    lam = pick_best(LAMBDA_GRID, lambda v: em_f1(outcomes(table, everyone, base - v * tokens, num_q), np.arange(num_q)))

    on_calib = {
        "baseline": em_f1(outcomes(table, calib_rows, -table["rank"][calib_rows].astype(np.float64), num_q), calib_q),
        "length penalty": em_f1(outcomes(table, calib_rows, base[calib_rows] - lam * tokens[calib_rows], num_q), calib_q),
        "frozen head + mix": em_f1(outcomes(table, calib_rows, mix(alpha), num_q), calib_q),
    }
    return {"config": config, "w": head["w"], "mean": head["mean"], "std": head["std"],
            "l2": float(head["l2"]), "step": int(head["step"]), "alpha": float(alpha), "lambda": float(lam),
            "calib_fold": int(calib_fold), "seed": int(seed), "n_folds": int(n_folds),
            "train_questions": int((folds != calib_fold).sum()), "calib_questions": int(len(calib_q)),
            "on_calibration_fold": {k: [round(x, 2) for x in v] for k, v in on_calib.items()}}


def save_final(directory, frozen):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(frozen[k]) for k in ARRAYS}
    scalars = {k: v for k, v in frozen.items() if k not in ARRAYS and k != "on_calibration_fold"}
    np.savez(directory / "final_head.npz", **arrays, **{k: np.array(v) for k, v in scalars.items()})
    (directory / "final_settings.json").write_text(
        json.dumps({**scalars, "on_calibration_fold": frozen["on_calibration_fold"],
                    "n_features": int(frozen["w"].shape[0])}, indent=2) + "\n", encoding="utf-8")


def load_final(directory):
    data = np.load(Path(directory) / "final_head.npz")
    out = {k: data[k] for k in ARRAYS}
    out["config"] = str(data["config"])
    for key in ("l2", "alpha", "lambda"):
        out[key] = float(data[key])
    for key in ("step", "calib_fold", "seed", "n_folds", "train_questions", "calib_questions"):
        out[key] = int(data[key])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", default="scalars+text+span")
    parser.add_argument("--calib-fold", type=int, default=0)
    parser.add_argument("--l2", default=None, help="comma-separated L2 grid (default: the cross-validation grid)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.02)
    args = parser.parse_args()

    from .text_features import load_text_features
    table = load_table(args.table)
    table["text"] = load_text_features(args.text_features, len(table["q_index"]))
    pooled = np.load(args.exports / "span_pooled.npy", mmap_mode="r")
    ids = [q["id"] for q in json.loads((args.exports / "questions.json").read_text(encoding="utf-8"))]
    titles = load_titles(args.validation)
    grid = tuple(float(v) for v in args.l2.split(",")) if args.l2 else None
    frozen = freeze(table, pooled, [titles[i] for i in ids], args.config, calib_fold=args.calib_fold,
                    l2_grid=grid, steps=args.steps, lr=args.lr)
    save_final(args.output_dir, frozen)

    print(f"Trained on {frozen['train_questions']} questions, calibrated on fold {frozen['calib_fold']} "
          f"({frozen['calib_questions']} questions). {frozen['w'].shape[0]} features.")
    print(f"Frozen: config={frozen['config']}  l2={frozen['l2']:g}  step={frozen['step']}  "
          f"alpha={frozen['alpha']:g}  lambda={frozen['lambda']:g}")
    print("On the calibration fold (optimistic: settings were chosen here), EM / F1:")
    for name, (em, f1) in frozen["on_calibration_fold"].items():
        print(f"  {name:20s} {em:.2f} / {f1:.2f}")
    print(f"\nWrote {args.output_dir}/final_head.npz and final_settings.json. Commit and tag them BEFORE running dev_score.")


if __name__ == "__main__":
    main()