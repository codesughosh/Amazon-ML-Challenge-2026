"""Cross-validation harness.

The public leaderboard is a trap: final rank comes from a private leaderboard
on the complete test set. Trust the OOF score this produces, not the board.
If local CV and the public board disagree in direction, believe local CV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from .metrics import GREATER_IS_BETTER, score
from .utils import SEED, free, timer


def make_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    strategy: str = "kfold",
    target_col: str | None = None,
    group_col: str | None = None,
    seed: int = SEED,
) -> np.ndarray:
    """Return a fold index per row.

    strategy:
      kfold       - plain random. Default for regression.
      stratified  - for classification, or binned regression targets.
      group       - when an entity (product / seller / brand) must not straddle
                    folds. Use this if the test set holds out whole groups.
    """
    folds = np.full(len(df), -1, dtype=np.int8)

    if strategy == "kfold":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(df)
    elif strategy == "stratified":
        assert target_col, "stratified needs target_col"
        y = df[target_col]
        if pd.api.types.is_float_dtype(y):
            # Bin a continuous target so each fold has the same price profile.
            y = pd.qcut(y.rank(method="first"), q=10, labels=False)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(df, y)
    elif strategy == "group":
        assert group_col, "group needs group_col"
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(df, groups=df[group_col])
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    for i, (_, val_idx) in enumerate(splits):
        folds[val_idx] = i
    assert (folds >= 0).all(), "some rows were not assigned a fold"
    return folds


def run_cv(
    df: pd.DataFrame,
    folds: np.ndarray,
    fit_predict,
    target_col: str,
    metric: str,
    test_df: pd.DataFrame | None = None,
    verbose: bool = True,
):
    """Run out-of-fold CV.

    `fit_predict(train_df, val_df, test_df, fold) -> (val_pred, test_pred|None)`

    Returns (oof_pred, test_pred_mean, per_fold_scores, oof_score).
    Bag the per-fold test predictions rather than refitting on all data - it is
    strictly more robust and costs nothing extra.
    """
    n_splits = int(folds.max()) + 1
    oof = np.zeros(len(df), dtype=np.float64)
    test_preds, fold_scores = [], []

    for fold in range(n_splits):
        tr_idx, va_idx = np.where(folds != fold)[0], np.where(folds == fold)[0]
        with timer(f"fold {fold}"):
            val_pred, test_pred = fit_predict(
                df.iloc[tr_idx], df.iloc[va_idx], test_df, fold
            )
        oof[va_idx] = val_pred
        if test_pred is not None:
            test_preds.append(np.asarray(test_pred))

        s = score(metric, df.iloc[va_idx][target_col].values, val_pred)
        fold_scores.append(s)
        if verbose:
            print(f"  fold {fold} {metric} = {s:.5f}")
        free()

    oof_score = score(metric, df[target_col].values, oof)
    test_mean = np.mean(test_preds, axis=0) if test_preds else None

    if verbose:
        print(f"\n{'=' * 46}")
        print(f"OOF {metric} = {oof_score:.5f}")
        print(f"fold mean   = {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
        print(f"{'=' * 46}")
        if np.std(fold_scores) > 0.15 * abs(np.mean(fold_scores)):
            print("WARNING: high fold variance - CV is unstable, treat gains "
                  "smaller than the spread as noise.")
    return oof, test_mean, fold_scores, oof_score


class Tracker:
    """Append-only experiment log. Keeps the team honest about what helped."""

    def __init__(self, path="experiments.csv", metric="smape"):
        self.path, self.metric = path, metric
        self.rows = []

    def log(self, name: str, oof_score: float, note: str = "") -> None:
        import datetime

        self.rows.append({
            "time": datetime.datetime.now().strftime("%m-%d %H:%M"),
            "name": name,
            self.metric: round(oof_score, 5),
            "note": note,
        })
        pd.DataFrame(self.rows).to_csv(self.path, index=False)
        best = (max if GREATER_IS_BETTER[self.metric] else min)(
            r[self.metric] for r in self.rows
        )
        flag = "  <-- BEST" if oof_score == best else ""
        print(f"logged: {name} = {oof_score:.5f}{flag}")

    def leaderboard(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows).sort_values(
            self.metric, ascending=not GREATER_IS_BETTER[self.metric]
        )
