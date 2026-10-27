"""Fit a temperature on held-out data and report calibration metrics.

Subcommands:

    fit     Fit T on the VALIDATION partition scores; write calibration.json.
    report  Report Brier/ECE/NLL + plots on DEV, one-shot, with hard
            assertions (see HANDOFF-confidence-v1.md section 6).

`report` needs matplotlib for the plots (runs locally, not in the pinned
training environment). `fit` needs only numpy.

Design notes:
- y (correctness) is computed with the vendor scorer's compute_exact, max
  over golds — never a reimplementation.
- The dev file has ~3 golds per question and val has 1; never compare ECE
  across the two files (base rate of y differs).
- Hard assertions in report:
    1. EM and F1 identical to the baseline run's metrics.json (temperature
       scaling is a monotone map over the confidence; the ANSWER must not
       change — failing this means an offset/mask/argmax bug).
    2. NLL decreased vs uncalibrated (T was fitted to minimise NLL).
    3. Brier and ECE both decreased vs uncalibrated (the point of the exercise).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

from .confidence import (
    brier,
    brier_skill_score,
    correctness,
    ece,
    fit_temperature,
    nll,
    reliability_bins,
    risk_coverage,
    span_probability,
)


def _load_scores(path: Path) -> dict:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _labels(scores: dict, examples: list[dict], predictions: dict[str, str]) -> np.ndarray:
    by_id = {example["id"]: example for example in examples}
    y = np.zeros(len(scores["qid"]), dtype=np.float64)
    for row, qid in enumerate(scores["qid"]):
        y[row] = correctness(by_id[qid], predictions[qid])
    return y


def _confidence(scores: dict, T: float = 1.0) -> np.ndarray:
    s, e = scores["s"], scores["e"]
    rows = np.arange(s.shape[0])
    ps = np.exp(s.astype(np.float64) / T)
    pe = np.exp(e.astype(np.float64) / T)
    ps /= ps.sum(axis=1, keepdims=True)
    pe /= pe.sum(axis=1, keepdims=True)
    return ps[rows, scores["best_i"]] * pe[rows, scores["best_j"]]


def _signal(scores: dict, name: str, T: float = 1.0) -> np.ndarray:
    """Confidence ranking signal per question, from the scores cache.

    prob: v1 span probability (optionally tempered). Higher = more confident.
    neg_margin / neg_entropy: ablation signals oriented so that HIGHER =
    more confident (margin and entropy themselves are lower = more confident).
    """
    s, e, mask = scores["s"], scores["e"], scores["mask"]
    if name == "prob":
        return _confidence(scores, T=T)
    if name in ("margin", "neg_margin"):
        from .confidence import span_margin

        vals = np.array([
            span_margin(s[k], e[k], mask[k], int(scores["best_i"][k]), int(scores["best_j"][k]), T=T)
            for k in range(s.shape[0])
        ])
        return -vals  # higher margin = more confident -> negate to keep 'higher = confident'
    if name in ("entropy", "neg_entropy"):
        from .confidence import entropy_signal

        vals = np.array([entropy_signal(s[k], e[k], mask[k], T=T) for k in range(s.shape[0])])
        return -vals
    raise ValueError(f"Unknown signal: {name}")


def _threshold_for_error(c: np.ndarray, y: np.ndarray, target_error: float) -> float:
    """Largest threshold tau (over the val ranking) whose retained set has
    error <= target_error. Retained = {c >= tau}. Returns the midpoint of the
    boundary gap so the threshold transfers between monotone transforms.
    """
    order = np.argsort(c)[::-1]
    y_sorted = y[order]
    c_sorted = c[order]
    errors = 1.0 - np.cumsum(y_sorted) / np.arange(1, len(y) + 1)
    eligible = np.nonzero(errors <= target_error)[0]
    if eligible.size == 0:
        raise ValueError(
            f"No confidence prefix reaches target error {target_error:.2%} "
            f"(best prefix error: {errors.min():.2%} at k={int(np.argmin(errors)) + 1})"
        )
    k = int(eligible[-1]) + 1  # largest retained-set size meeting the target
    if k == len(c_sorted):
        return float("-inf")
    return float((c_sorted[k - 1] + c_sorted[k]) / 2.0)


def cmd_target(args: argparse.Namespace) -> None:
    """Stretch goal (handoff section 8): the most persuasive single result.

    Pick a threshold on the VALIDATION partition such that the retained set
    has <= target error; then report the ACHIEVED error and coverage on dev
    for each signal. Note: for the temperature-scaled vs uncalibrated
    comparison the retained set is IDENTICAL (temperature scaling is a
    monotone map, so rankings are unchanged) — the interesting comparison is
    ACROSS SIGNALS: span probability vs margin vs entropy.
    """
    from .data import load_squad

    val_scores = _load_scores(args.val_scores)
    dev_scores = _load_scores(args.scores)
    val_examples = load_squad(args.val_data)
    dev_examples = load_squad(args.data)
    val_preds = json.loads(args.val_predictions.read_text(encoding="utf-8"))
    dev_preds = json.loads(args.predictions.read_text(encoding="utf-8"))
    y_val = _labels(val_scores, val_examples, val_preds)
    y_dev = _labels(dev_scores, dev_examples, dev_preds)

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    T = calibration["temperature"]

    rows = []
    for signal in ("prob", "margin", "entropy"):
        c_val = _signal(val_scores, signal)
        c_dev_uncal = _signal(dev_scores, signal, T=1.0)
        try:
            tau = _threshold_for_error(c_val, y_val, args.target_error)
        except ValueError as exc:
            # A signal that cannot reach the target is a FINDING, not a crash
            # (e.g. margin may simply be uninformative enough).
            rows.append({"signal": signal, "threshold": None,
                         "dev_coverage": None, "dev_achieved_error": None,
                         "dev_n_retained": None, "note": str(exc)})
            continue
        retained = c_dev_uncal >= tau
        coverage = float(retained.mean())
        achieved = float(1 - y_dev[retained].mean()) if retained.any() else float("nan")
        rows.append({
            "signal": signal,
            "threshold": tau,
            "dev_coverage": coverage,
            "dev_achieved_error": achieved,
            "dev_n_retained": int(retained.sum()),
        })

    print(f"Target error on val: {args.target_error:.0%} (n_val={len(y_val)}, n_dev={len(y_dev)})")
    print(f"{'signal':>10s} {'threshold':>10s} {'dev cov':>9s} {'dev err':>9s} {'n':>7s}")
    for r in rows:
        if r["threshold"] is None:
            print(f"{r['signal']:>10s}   UNREACHABLE: {r['note']}")
        else:
            print(f"{r['signal']:>10s} {r['threshold']:>10.4f} {r['dev_coverage']:>9.2%} "
                  f"{r['dev_achieved_error']:>9.2%} {r['dev_n_retained']:>7d}")

    output = {
        "target_error": args.target_error,
        "temperature": T,
        "fitted_on": str(args.val_data),
        "reported_on": str(args.data),
        "signals": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


def cmd_fit(args: argparse.Namespace) -> None:
    scores = _load_scores(args.scores)
    from .data import load_squad

    flat = load_squad(args.data)
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    y = _labels(scores, flat, predictions)
    T = fit_temperature(scores["s"], scores["e"], scores["mask"], scores["best_i"], scores["best_j"], y)

    result = {
        "method": "temperature scaling",
        "temperature": T,
        "fitted_on": str(args.data),
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "threshold": 0.5,
        "package_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }

    # Half-split sanity check (handoff gotcha 9): fit T on each half of the
    # val partition, split BY ARTICLE; T_A ≈ T_B means one parameter on 8.7k
    # rows has nothing to overfit.
    articles = {}
    for example in flat:
        articles.setdefault(example["article_title"], []).append(example["id"])
    titles = sorted(articles)
    half = len(titles) // 2
    groups = [set().union(*(articles[t] for t in titles[:half])),
              set().union(*(articles[t] for t in titles[half:]))]
    t_half = []
    for group in groups:
        rows = np.array([qid in group for qid in scores["qid"]])
        t_half.append(fit_temperature(
            scores["s"][rows], scores["e"][rows], scores["mask"][rows],
            scores["best_i"][rows], scores["best_j"][rows], y[rows],
        ))
    result["half_split"] = {"T_A": t_half[0], "T_B": t_half[1], "delta": abs(t_half[0] - t_half[1])}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Fitted T={T:.4f} on {len(y)} questions "
          f"(half-split: T_A={t_half[0]:.4f}, T_B={t_half[1]:.4f}) -> {args.output}")


def cmd_report(args: argparse.Namespace) -> None:
    scores = _load_scores(args.scores)
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    T = calibration["temperature"]

    # EM / F1 straight from the vendor scorer (gold first, max over golds).
    import subprocess

    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[2] / "vendor" / "evaluate-v2.0.py"),
         str(args.data), str(args.predictions)],
        capture_output=True, text=True, check=True,
    )
    metrics = json.loads(result.stdout)

    examples = json.loads(args.data.read_text(encoding="utf-8"))["data"]
    from .data import load_squad

    flat = load_squad(args.data)
    y = _labels(scores, flat, predictions)
    c_uncal = _confidence(scores, T=1.0)
    c_cal = _confidence(scores, T=T)

    row_uncal = {
        "EM": metrics["exact"], "F1": metrics["f1"],
        "mean_c": float(c_uncal.mean()), "base": float(y.mean()),
        "Brier": brier(c_uncal, y), "BSS": brier_skill_score(c_uncal, y),
        "ECE": ece(c_uncal, y), "NLL": nll(c_uncal, y),
    }
    row_cal = {
        "EM": metrics["exact"], "F1": metrics["f1"],
        "mean_c": float(c_cal.mean()), "base": float(y.mean()),
        "Brier": brier(c_cal, y), "BSS": brier_skill_score(c_cal, y),
        "ECE": ece(c_cal, y), "NLL": nll(c_cal, y),
    }

    # --- hard assertions (handoff section 6) ---
    baseline_metrics = None
    if args.metrics is not None and args.metrics.exists():
        baseline_metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
        for key, metric_key in (("EM", "exact"), ("F1", "f1")):
            if abs(row_uncal[key] - baseline_metrics[metric_key]) > 1e-6:
                raise AssertionError(
                    f"Assertion 1 FAILED: {key} {row_uncal[key]} != baseline "
                    f"{baseline_metrics[metric_key]} — answers changed; "
                    "suspect an offset/mask/argmax bug."
                )
    if row_cal["NLL"] >= row_uncal["NLL"]:
        raise AssertionError(
            f"Assertion 2 FAILED: NLL did not decrease "
            f"({row_uncal['NLL']:.4f} -> {row_cal['NLL']:.4f})."
        )
    if row_cal["Brier"] >= row_uncal["Brier"] or row_cal["ECE"] >= row_uncal["ECE"]:
        raise AssertionError(
            f"Assertion 3 FAILED: Brier/ECE did not both decrease "
            f"(Brier {row_uncal['Brier']:.4f} -> {row_cal['Brier']:.4f}, "
            f"ECE {row_uncal['ECE']:.4f} -> {row_cal['ECE']:.4f})."
        )

    header = f"{'':12s}" + "".join(f"{k:>9s}" for k in row_uncal)
    print(header)
    print(f"{'uncalib.':12s}" + "".join(f"{v:>9.4f}" if isinstance(v, float) else f"{v:>9.2f}" for v in row_uncal.values()))
    print(f"{'+T (T=' + format(T, '.2f') + ')':12s}"
          + "".join(f"{v:>9.4f}" if isinstance(v, float) else f"{v:>9.2f}" for v in row_cal.values()))

    report = {
        "temperature": T,
        "n": int(len(y)),
        "uncalibrated": row_uncal,
        "calibrated": row_cal,
        "reliability_bins_uncalibrated": reliability_bins(c_uncal, y),
        "reliability_bins_calibrated": reliability_bins(c_cal, y),
        "risk_coverage_uncalibrated": risk_coverage(c_uncal, y),
        "risk_coverage_calibrated": risk_coverage(c_cal, y),
        "assertions": {
            "em_f1_match_baseline": baseline_metrics is not None,
            "nll_decreased": True,
            "brier_ece_decreased": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote report to {args.output}")

    if args.plot_dir is not None:
        _plots(args.plot_dir, c_uncal, c_cal, y, T, row_uncal, row_cal)


def _plots(plot_dir: Path, c_uncal, c_cal, y, T, row_uncal, row_cal) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(0, 1, 16)

    # 1. Reliability diagram.
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
    for c, label, color in ((c_uncal, "uncalibrated", "tab:red"),
                            (c_cal, f"calibrated (T={T:.2f})", "tab:blue")):
        rb = reliability_bins(c, y)
        ax.plot([b["mean_c"] for b in rb], [b["mean_y"] for b in rb],
                "o-", color=color, label=label)
    ax.set(xlabel="confidence", ylabel="accuracy", title="Reliability (dev)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "reliability.png", dpi=150)
    plt.close(fig)

    # 2. Risk-coverage curve with the 81.5 EM human ceiling and base-rate floor.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for c, label, color in ((c_uncal, "uncalibrated", "tab:red"),
                            (c_cal, f"calibrated (T={T:.2f})", "tab:blue")):
        rc = risk_coverage(c, y, coverages=np.arange(0.5, 1.001, 0.05))
        ax.plot([r["coverage"] for r in rc], [r["em"] for r in rc], "o-", color=color, label=label)
    ax.axhline(81.5, color="k", linestyle="--", linewidth=1, label="human ceiling 81.5 EM")
    ax.axhline(row_uncal["base"] * 100, color="grey", linestyle=":", linewidth=1,
               label=f"base rate {row_uncal['base']*100:.1f} EM")
    ax.set(xlabel="coverage (fraction of questions answered)", ylabel="EM on retained set",
           title="Risk–coverage (dev)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "risk_coverage.png", dpi=150)
    plt.close(fig)

    # 3. Confidence histograms.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(c_uncal, bins=bins, alpha=0.5, label="uncalibrated", color="tab:red")
    ax.hist(c_cal, bins=bins, alpha=0.5, label=f"calibrated (T={T:.2f})", color="tab:blue")
    ax.set(xlabel="confidence", ylabel="questions", title="Confidence distribution (dev)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "confidence_hist.png", dpi=150)
    plt.close(fig)
    print(f"Wrote plots to {plot_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="fit T on the validation partition")
    fit.add_argument("--scores", type=Path, default=Path("runs/baseline/val-scores.npz"))
    fit.add_argument("--data", type=Path, default=Path("runs/baseline/validation-v1.1.json"))
    fit.add_argument("--predictions", type=Path, default=Path("runs/baseline/val-predictions.json"))
    fit.add_argument("--output", type=Path, default=Path("runs/baseline/calibration.json"))
    fit.set_defaults(func=cmd_fit)

    report = sub.add_parser("report", help="report calibration metrics on dev")
    report.add_argument("--scores", type=Path, default=Path("runs/baseline/dev-scores.npz"))
    report.add_argument("--data", type=Path, default=Path("data/raw/dev-v1.1.json"))
    report.add_argument("--predictions", type=Path, default=Path("runs/baseline/predictions.json"))
    report.add_argument("--calibration", type=Path, default=Path("runs/baseline/calibration.json"))
    report.add_argument("--metrics", type=Path, default=Path("runs/baseline/metrics.json"))
    report.add_argument("--output", type=Path, default=Path("runs/baseline/calibration-report.json"))
    report.add_argument("--plot-dir", type=Path, default=Path("runs/baseline/plots"))
    report.set_defaults(func=cmd_report)

    target = sub.add_parser(
        "target",
        help="error-rate targeting: val threshold for a target error, achieved on dev, per signal",
    )
    target.add_argument("--val-scores", type=Path, default=Path("runs/baseline/val-scores.npz"))
    target.add_argument("--val-data", type=Path, default=Path("runs/baseline/validation-v1.1.json"))
    target.add_argument("--val-predictions", type=Path, default=Path("runs/baseline/val-predictions.json"))
    target.add_argument("--scores", type=Path, default=Path("runs/baseline/dev-scores.npz"))
    target.add_argument("--data", type=Path, default=Path("data/raw/dev-v1.1.json"))
    target.add_argument("--predictions", type=Path, default=Path("runs/baseline/predictions.json"))
    target.add_argument("--calibration", type=Path, default=Path("runs/baseline/calibration.json"))
    target.add_argument("--target-error", type=float, default=0.05)
    target.add_argument("--output", type=Path, default=Path("runs/baseline/target-report.json"))
    target.set_defaults(func=cmd_target)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
