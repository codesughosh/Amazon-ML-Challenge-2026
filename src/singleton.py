"""Entity-level singleton detection.

5.6% of Source-1 entities have no match at all. Under the metric each one is
worth a full 1.0 when predicted as an empty list, and 0.0 otherwise - so
getting them right is worth up to ~0.05 of macro F_0.5, comparable to a large
modelling improvement.

The pairwise model cannot deliver this on its own. To conclude "no match" from
independent pair probabilities, *every* candidate must be improbable: with 50
candidates at even p=0.1 each, P(none is a match) = 0.9^50 ~ 0.5%. The
independence assumption makes the empty set nearly unreachable no matter how
well calibrated the pair model is.

So we ask the question directly at the entity level instead: given the *shape*
of an entity's candidate score distribution, does it look like an entity whose
true match is present, or like one whose isn't? A singleton's candidates are
uniformly mediocre; a matched entity has at least one standout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def entity_features(s1_row: np.ndarray, prob: np.ndarray,
                    sim: np.ndarray | None = None,
                    n_entities: int | None = None) -> pd.DataFrame:
    """Aggregate pair-level scores into one feature row per S1 entity.

    `s1_row` must be the integer S1 index for each candidate pair.
    Entities with no candidates at all get an all-zero row (and are almost
    certainly singletons).
    """
    n = n_entities if n_entities is not None else int(s1_row.max()) + 1
    df = pd.DataFrame({"i": s1_row, "p": prob})
    if sim is not None:
        df["s"] = sim

    g = df.groupby("i", sort=True)
    out = pd.DataFrame(index=np.arange(n, dtype=np.int64))

    agg = g["p"].agg(["max", "mean", "std", "count", "sum"])
    out["p_max"] = agg["max"]
    out["p_mean"] = agg["mean"]
    out["p_std"] = agg["std"]
    out["n_cand"] = agg["count"]
    out["p_sum"] = agg["sum"]

    # Counts above thresholds: how many candidates look plausible at all.
    for t in (0.2, 0.4, 0.6, 0.8, 0.9):
        out[f"n_above_{t}"] = g["p"].apply(lambda x, t=t: (x >= t).sum())

    # Shape of the top of the distribution. A matched entity usually has a
    # clear leader; a singleton has a flat, mediocre profile.
    top = g["p"].apply(lambda x: np.sort(x.to_numpy())[::-1][:5])
    for r in range(5):
        out[f"p_top{r+1}"] = top.apply(lambda v, r=r: v[r] if len(v) > r else 0.0)
    out["gap_1_2"] = out["p_top1"] - out["p_top2"]
    out["gap_1_3"] = out["p_top1"] - out["p_top3"]
    out["ratio_2_1"] = out["p_top2"] / np.maximum(out["p_top1"], 1e-6)

    if sim is not None:
        sagg = g["s"].agg(["max", "mean"])
        out["sim_max"] = sagg["max"]
        out["sim_mean"] = sagg["mean"]
        out["sim_gap"] = out["sim_max"] - out["sim_mean"]

    return out.fillna(0.0).astype(np.float32)


def fit_predict(train_X: pd.DataFrame, train_y: np.ndarray,
                test_X: pd.DataFrame, n_estimators: int = 300):
    """Train the singleton classifier and return P(singleton) for test rows."""
    import lightgbm as lgb

    # Singletons are ~5.6% of entities, so rebalance or the model predicts
    # "never a singleton" and scores well on accuracy while being useless.
    m = lgb.LGBMClassifier(
        n_estimators=n_estimators, learning_rate=0.05, num_leaves=63,
        min_child_samples=40, colsample_bytree=0.8, subsample=0.8,
        subsample_freq=1, class_weight="balanced", n_jobs=-1, verbose=-1,
    )
    m.fit(train_X, train_y)
    return m, m.predict_proba(test_X)[:, 1]


def tune_singleton_threshold(p_singleton: np.ndarray, base_pred: dict,
                             s1_ids: np.ndarray, truth: dict,
                             grid=None):
    """Find the P(singleton) cut-off that maximises macro F_0.5.

    Forcing an entity to empty is a gamble: +1.0 if it really is a singleton,
    but it throws away whatever the pair model would have scored otherwise. The
    optimum is usually a high threshold - only override when confident.
    """
    from .er import macro_f05

    if grid is None:
        grid = np.arange(0.50, 0.99, 0.02)

    best = (1.01, macro_f05(base_pred, truth))   # 1.01 = never override
    rows = [(1.01, best[1], 0)]
    for t in grid:
        forced = {s: (set() if p >= t else base_pred.get(s, set()))
                  for s, p in zip(s1_ids, p_singleton)}
        for s in base_pred:
            forced.setdefault(s, base_pred[s])
        sc = macro_f05(forced, truth)
        n_forced = int((p_singleton >= t).sum())
        rows.append((float(t), sc, n_forced))
        if sc > best[1]:
            best = (float(t), sc)
    return best[0], best[1], rows
