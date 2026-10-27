"""Confidence signals and calibration metrics for the SQuAD baseline.

Pure numpy + stdlib; no torch. The vendor scorer (`vendor/evaluate-v2.0.py`)
is imported dynamically so ground-truth labels never come from a
reimplementation.

Conventions:
- `s`, `e` are raw start/end logit rows for one question's winning window,
  each of length max_length (padded rows included).
- `mask` is the context mask (offset_mapping[k] is not None).
- v1 confidence is the model's own span probability over the FULL window row
  (softmax over all positions, the standard signal); `mask` is unused in v1
  but kept in the signature for v2's renormalisation over legal spans.
- Metrics take binary `y` ("was the emitted answer correct") and confidence
  `c` in [0, 1]. Brier is the primary number (proper scoring rule); ECE is
  diagnostic only (not proper). See HANDOFF-confidence-v1.md section 5.1.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_VENDOR_DIR = Path(__file__).resolve().parents[2] / "vendor"


_vendor_module = None


def _load_vendor():
    global _vendor_module
    if _vendor_module is None:
        spec = importlib.util.spec_from_file_location(
            "evaluate_v2_0", _VENDOR_DIR / "evaluate-v2.0.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["evaluate_v2_0"] = module
        spec.loader.exec_module(module)
        _vendor_module = module
    return _vendor_module


def softmax_rows(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax (logits reach ±20, so shift by the row max)."""
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def span_probability(
    s: np.ndarray, e: np.ndarray, mask: np.ndarray, i: int, j: int, T: float = 1.0
) -> float:
    """v1 signal: softmax(s/T)[i] * softmax(e/T)[j] over the full window row.

    NOTE (v2): this product sums to well below 1 over legal spans because it
    is not renormalised over the legal grid; mask is unused here but required
    by v2's `Z` over legal spans.
    """
    if T <= 0:
        raise ValueError("Temperature must be positive")
    p_start = softmax_rows(s / T)[i]
    p_end = softmax_rows(e / T)[j]
    return float(p_start * p_end)


def span_margin(
    s: np.ndarray, e: np.ndarray, mask: np.ndarray, i: int, j: int, T: float = 1.0
) -> float:
    """top1 - top2 score gap over legal spans in the n_best grid (ablation).

    Cheap margin signal computed over the legal (mask-true, ordered) token
    pairs whose product probability exceeds all others; uses the same 20x20
    grid limitation as the search (see handoff gotcha 6).
    """
    s = np.asarray(s, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    scores = (s[:, None] + e[None, :]) / T  # outer sum: grid[start, end]
    legal = mask[:, None] & mask[None, :]
    legal &= np.triu(np.ones_like(legal, dtype=bool))  # end >= start
    scores = np.where(legal, scores, -np.inf)
    flat = scores.ravel()
    if not np.isfinite(flat).any():
        return float("nan")
    top2 = np.sort(flat[~np.isnan(flat)])[-2:]
    return float(top2[1] - top2[0]) if len(top2) == 2 else 0.0


def correctness(example: dict, prediction: str) -> int:
    """y in {0,1} via the vendor scorer's compute_exact, max over golds."""
    vendor = _load_vendor()
    golds = [a["text"] for a in example["answers"] if vendor.normalize_answer(a["text"])]
    if not golds:
        golds = [""]
    return int(max(vendor.compute_exact(g, prediction) for g in golds))


def ece(c: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error with EQUAL-MASS bins.

    Equal-mass (np.array_split on sorted confidences), not equal-width:
    confidences cluster near 0.8 so equal-width bins go sparse. Report
    n_bins alongside every ECE.
    """
    c = np.asarray(c, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if c.shape != y.shape or c.size == 0:
        raise ValueError("c and y must be non-empty arrays of equal shape")
    order = np.argsort(c)
    c_sorted, y_sorted = c[order], y[order]
    bins = np.array_split(np.arange(c.size), n_bins)
    error = 0.0
    for idx in bins:
        if idx.size == 0:
            continue
        error += (idx.size / c.size) * abs(float(y_sorted[idx].mean()) - float(c_sorted[idx].mean()))
    return float(error)


def brier(c: np.ndarray, y: np.ndarray) -> float:
    """Brier score: mean (c - y)^2. Proper scoring rule; lower is better."""
    c = np.asarray(c, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean((c - y) ** 2))


def nll(c: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Binary negative log-likelihood (log loss). Clipped for stability."""
    c = np.clip(np.asarray(c, dtype=np.float64), eps, 1.0 - eps)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(c) + (1 - y) * np.log(1 - c)))


def brier_skill_score(c: np.ndarray, y: np.ndarray) -> float:
    """1 - B / base(1-base): 0 = no better than the base-rate predictor."""
    c = np.asarray(c, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    base = float(y.mean())
    denom = base * (1 - base)
    if denom == 0:
        raise ValueError("y is constant; Brier skill score undefined")
    return float(1.0 - brier(c, y) / denom)


def reliability_bins(c: np.ndarray, y: np.ndarray, n_bins: int = 15) -> list[dict]:
    """Per-bin mean confidence / mean accuracy / count, equal-mass bins."""
    c = np.asarray(c, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(c)
    c_sorted, y_sorted = c[order], y[order]
    bins = np.array_split(np.arange(c.size), n_bins)
    out = []
    for idx in bins:
        if idx.size == 0:
            continue
        out.append(
            {
                "mean_c": float(c_sorted[idx].mean()),
                "mean_y": float(y_sorted[idx].mean()),
                "n": int(idx.size),
            }
        )
    return out


def risk_coverage(
    c: np.ndarray, y: np.ndarray, coverages=(0.95, 0.9, 0.8, 0.7, 0.6, 0.5)
) -> list[dict]:
    """EM (accuracy) on the top-`coverage` fraction of questions by confidence."""
    c = np.asarray(c, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(c)[::-1]  # most confident first
    y_sorted = y[order]
    n = c.size
    out = []
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        out.append({"coverage": cov, "n": k, "em": float(y_sorted[:k].mean())})
    return out


def entropy_signal(s: np.ndarray, e: np.ndarray, mask: np.ndarray, T: float = 1.0) -> float:
    """Ablation signal: mean of the start/end softmax entropies over the full row.

    LOWER entropy = more confident. For risk-coverage ranking, sort by the
    NEGATIVE of this value. Computed over the full window row (same scope as
    the v1 span probability, not just the n_best grid).
    """
    ps = softmax_rows(np.asarray(s, dtype=np.float64) / T)
    pe = softmax_rows(np.asarray(e, dtype=np.float64) / T)
    h_s = float(-np.sum(ps * np.log(np.clip(ps, 1e-12, None))))
    h_e = float(-np.sum(pe * np.log(np.clip(pe, 1e-12, None))))
    return (h_s + h_e) / 2.0


def renormalised_span_probability(
    s: np.ndarray, e: np.ndarray, mask: np.ndarray, i: int, j: int, T: float = 1.0
) -> float:
    """v2 signal (deferred in v1, implemented here for the cache experiments):
    p = exp(s_i + e_j) / Z over ALL legal spans of the window, with
    Z = sum over legal (a, b), a <= b, of exp(s_a + e_b).

    Unlike the v1 product, this IS a distribution over answers — it sums to 1
    over legal spans. Requires full logit rows + context mask, which is why
    predict.py caches complete rows (no n_best truncation here).
    """
    if T <= 0:
        raise ValueError("Temperature must be positive")
    s = np.asarray(s, dtype=np.float64) / T
    e = np.asarray(e, dtype=np.float64) / T
    scores = s[:, None] + e[None, :]
    legal = mask[:, None] & mask[None, :]
    legal &= np.triu(np.ones_like(legal, dtype=bool))  # a <= b
    if not legal.any():
        raise ValueError("No legal spans in window")
    legal_scores = scores[legal]
    max_score = legal_scores.max()
    log_z = max_score + np.log(np.exp(legal_scores - max_score).sum())
    return float(np.exp(s[i] + e[j] - log_z))


def fit_two_temperature(
    s: np.ndarray,
    e: np.ndarray,
    mask: np.ndarray,
    best_i: np.ndarray,
    best_j: np.ndarray,
    y: np.ndarray,
    grid_s: np.ndarray = np.linspace(0.5, 3.0, 26),
    grid_e: np.ndarray = np.linspace(0.5, 3.0, 26),
) -> tuple[float, float]:
    """Stretch goal: separate start/end temperatures (T_s, T_e), 2 parameters.

    Returns (T_s, T_e) minimising NLL of softmax(s/T_s)[i] * softmax(e/T_e)[j].
    Precomputes per-T vectors and combines via outer products, so the 2-D
    search costs little more than the 1-D one.
    """
    s = np.asarray(s, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rows = np.arange(s.shape[0])

    def unit_vector(logits: np.ndarray, T: float, idx: np.ndarray) -> np.ndarray:
        p = softmax_rows(logits / T)
        return p[rows, idx]

    ps_all = {T: unit_vector(s, T, np.asarray(best_i, dtype=int)) for T in grid_s}
    pe_all = {T: unit_vector(e, T, np.asarray(best_j, dtype=int)) for T in grid_e}

    best = (None, None)
    best_loss = float("inf")
    for ts, ps in ps_all.items():
        for te, pe in pe_all.items():
            p = np.clip(ps * pe, 1e-12, 1 - 1e-12)
            loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
            if loss < best_loss:
                best, best_loss = (float(ts), float(te)), loss
    if best[0] is None:
        raise ValueError("Empty temperature grids")
    return best


def fit_temperature(
    s: np.ndarray,
    e: np.ndarray,
    mask: np.ndarray,
    best_i: np.ndarray,
    best_j: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray = np.linspace(0.25, 5.0, 476),
) -> float:
    """1-D search over T minimising NLL of the derived span probability.

    `s`/`e` are (n, max_length) logit-row caches, `best_i`/`best_j` the
    winning token positions, `y` the binary correctness labels. Returns the
    best float T. mask is unused in v1 (full-row softmax) but kept for
    signature stability with v2.
    """
    s = np.asarray(s, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    best_i = np.asarray(best_i, dtype=int)
    best_j = np.asarray(best_j, dtype=int)
    best_t, best_loss = None, float("inf")
    for T in grid:
        ps = softmax_rows(s / T)[np.arange(s.shape[0]), best_i]
        pe = softmax_rows(e / T)[np.arange(e.shape[0]), best_j]
        p = np.clip(ps * pe, 1e-12, 1 - 1e-12)
        loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        if loss < best_loss:
            best_t, best_loss = float(T), loss
    if best_t is None:
        raise ValueError("Empty temperature grid")
    return best_t
