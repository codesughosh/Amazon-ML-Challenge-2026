"""The metric and the decision layer.

Everything that turns pair probabilities into a submission lives here, because
this is where the score is actually won:

* `macro_f05`          - the competition metric, exactly
* `apply_assignment`   - the many-to-one constraint (each S2/S3 ID used once)
* `select_topk_threshold` - simple global-threshold decision (the floor)
* `select_expected_f05`   - per-entity k chosen to maximise expected F_0.5
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Metric
# --------------------------------------------------------------------------

def f05_single(pred: set, true: set) -> float:
    """F_0.5 for one Source-1 entity.

    Algebraically the official definition collapses to

        F_0.5 = 1.25 * TP / (0.25 * |T| + k)

    with k = |pred|. Verified against the statement's worked example:
    TP=2, k=3, |T|=2 -> 2.5/3.5 = 0.7143.
    """
    k, t = len(pred), len(true)
    if k == 0 and t == 0:
        return 1.0          # correctly identified singleton
    if k == 0 or t == 0:
        return 0.0          # missed everything, or invented matches
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    return 1.25 * tp / (0.25 * t + k)


def macro_f05(pred: dict[str, set], true: dict[str, set]) -> float:
    """Macro-average over every Source-1 entity in `true` (singletons included)."""
    if not true:
        return 0.0
    return float(np.mean([f05_single(pred.get(s, set()), t) for s, t in true.items()]))


# --------------------------------------------------------------------------
# Decision layer
# --------------------------------------------------------------------------

def apply_assignment(s1_idx: np.ndarray, cand_idx: np.ndarray, prob: np.ndarray,
                     ) -> np.ndarray:
    """Enforce many-to-one: each S2/S3 record goes to at most one S1 entity.

    Verified in training: 7,638,365 matched IDs, zero reuse. So whenever two S1
    entities claim the same candidate, at least one is wrong. Resolve greedily
    by probability - highest claim wins, later claims are dropped.

    Only ever removes predictions, so it strictly raises precision, which the
    metric weights double.

    Returns a boolean mask of rows to keep.
    """
    order = np.argsort(-prob, kind="stable")
    seen = np.zeros(int(cand_idx.max()) + 1 if len(cand_idx) else 1, dtype=bool)
    keep = np.zeros(len(prob), dtype=bool)
    for r in order:
        c = cand_idx[r]
        if not seen[c]:
            seen[c] = True
            keep[r] = True
    return keep


def select_topk_threshold(s1_idx: np.ndarray, prob: np.ndarray,
                          threshold: float, max_k: int = 12) -> np.ndarray:
    """Baseline decision: keep candidates above `threshold`, at most `max_k`.

    max_k is bounded by the data - the training ground truth never exceeds 11
    matches for one entity.
    """
    keep = prob >= threshold
    if max_k:
        order = np.lexsort((-prob, s1_idx))
        run = np.zeros(len(prob), dtype=np.int32)
        _, starts, counts = np.unique(s1_idx[order], return_index=True, return_counts=True)
        for s, c in zip(starts, counts):
            run[order[s:s + c]] = np.arange(c)
        keep &= run < max_k
    return keep


def _expected_f05_for_k(p_sorted: np.ndarray) -> int:
    """Choose k maximising E[F_0.5] for one entity.

    Under independence the optimal prediction is a top-k set, so only k has to
    be searched. With F_0.5 = 1.25*TP/(0.25|T| + k), and TP/|T| both
    Poisson-binomial, we evaluate

        E[F_0.5(k)] = 1.25 * E[ TP_k / (0.25 * (TP_k + FN_k) + k) ]

    exactly by convolving the two Poisson-binomial distributions. n is small
    (tens of candidates), so the O(n^3) worst case is cheap.
    """
    n = len(p_sorted)
    if n == 0:
        return 0

    # pmf_prefix[k] = distribution of the number of true matches among the top k
    pmf_prefix = [np.array([1.0])]
    cur = np.array([1.0])
    for p in p_sorted:
        cur = np.convolve(cur, [1 - p, p])
        pmf_prefix.append(cur.copy())

    # pmf_suffix[k] = distribution of true matches among candidates below k
    pmf_suffix = [None] * (n + 1)
    cur = np.array([1.0])
    pmf_suffix[n] = cur.copy()
    for j in range(n - 1, -1, -1):
        cur = np.convolve(cur, [1 - p_sorted[j], p_sorted[j]])
        pmf_suffix[j] = cur.copy()

    best_k, best_v = 0, -1.0
    for k in range(n + 1):
        a, b = pmf_prefix[k], pmf_suffix[k]
        if k == 0:
            # Empty prediction scores 1.0 only when the entity truly has no
            # matches - probability b[0].
            v = float(b[0])
        else:
            tp = np.arange(len(a))
            fn = np.arange(len(b))
            denom = 0.25 * (tp[:, None] + fn[None, :]) + k
            v = float(1.25 * np.sum(np.outer(a, b) * (tp[:, None] / denom)))
        if v > best_v:
            best_k, best_v = k, v
    return best_k


def select_expected_f05(s1_idx: np.ndarray, prob: np.ndarray,
                        max_cand: int = 25) -> np.ndarray:
    """Per-entity k chosen by expected-F_0.5 maximisation.

    Requires *calibrated* probabilities - run isotonic regression on a held-out
    fold first. With uncalibrated scores this is actively worse than a tuned
    global threshold.
    """
    keep = np.zeros(len(prob), dtype=bool)
    order = np.lexsort((-prob, s1_idx))
    s_sorted = s1_idx[order]
    _, starts, counts = np.unique(s_sorted, return_index=True, return_counts=True)

    for s, c in zip(starts, counts):
        rows = order[s:s + min(c, max_cand)]
        k = _expected_f05_for_k(prob[rows])
        if k:
            keep[rows[:k]] = True
    return keep


def tune_threshold(s1_idx: np.ndarray, prob: np.ndarray, cand_ids: np.ndarray,
                   truth: dict[str, set], s1_ids: np.ndarray,
                   grid: np.ndarray | None = None, max_k: int = 12):
    """Sweep the global threshold and return (best_threshold, best_score, table)."""
    if grid is None:
        grid = np.arange(0.10, 0.96, 0.025)
    rows = []
    best = (0.5, -1.0)
    for t in grid:
        keep = select_topk_threshold(s1_idx, prob, t, max_k)
        pred: dict[str, set] = {}
        for i, c in zip(s1_idx[keep], cand_ids[keep]):
            pred.setdefault(s1_ids[i], set()).add(c)
        sc = macro_f05(pred, truth)
        rows.append((float(t), sc, int(keep.sum())))
        if sc > best[1]:
            best = (float(t), sc)
    return best[0], best[1], rows
