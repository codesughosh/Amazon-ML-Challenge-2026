"""Competition metrics.

The exact metric is announced at kickoff. Implement it here FIRST, verify it
against any worked example in the problem statement, then never optimise
anything else. Every past edition used one of these families.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error


# --------------------------------------------------------------------------
# Regression (2025: price prediction used SMAPE)
# --------------------------------------------------------------------------

def smape(y_true, y_pred) -> float:
    """Symmetric MAPE, as a percentage in [0, 200]. Lower is better.

    2025 reference points: winning score 39.7, AIR-80 score 43.28.

    Note the denominator convention: (|A| + |F|) / 2. Some definitions omit
    the /2 (halving the score). If the problem statement gives a worked
    example, check against it before trusting this.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # Where both are zero the error is zero, not NaN.
    ratio = np.where(denom == 0, 0.0, np.abs(y_pred - y_true) / np.where(denom == 0, 1.0, denom))
    return float(100.0 * np.mean(ratio))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = y_true != 0
    return float(100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def rmsle(y_true, y_pred) -> float:
    y_true = np.clip(np.asarray(y_true, dtype=np.float64), 0, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


# --------------------------------------------------------------------------
# Classification / extraction (2024: entity extraction scored with F1)
# --------------------------------------------------------------------------

def f1_macro(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def f1_micro(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="micro", zero_division=0))


def f1_weighted(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def exact_match(y_true, y_pred) -> float:
    """Entity-extraction style: string-equality accuracy after normalisation."""
    y_true = [str(v).strip().lower() for v in y_true]
    y_pred = [str(v).strip().lower() for v in y_pred]
    return float(np.mean([t == p for t, p in zip(y_true, y_pred)]))


# --------------------------------------------------------------------------
# Registry + direction
# --------------------------------------------------------------------------

METRICS = {
    "smape": smape,
    "mape": mape,
    "rmsle": rmsle,
    "rmse": rmse,
    "mae": mae,
    "f1_macro": f1_macro,
    "f1_micro": f1_micro,
    "f1_weighted": f1_weighted,
    "exact_match": exact_match,
}

# True when a higher score is better.
GREATER_IS_BETTER = {
    "smape": False, "mape": False, "rmsle": False, "rmse": False, "mae": False,
    "f1_macro": True, "f1_micro": True, "f1_weighted": True, "exact_match": True,
}


def score(name: str, y_true, y_pred) -> float:
    return METRICS[name](y_true, y_pred)


def is_better(name: str, new: float, old: float) -> bool:
    return new > old if GREATER_IS_BETTER[name] else new < old


# --------------------------------------------------------------------------
# Gradient-boosting hooks
# --------------------------------------------------------------------------

def smape_eval_lgb(y_pred, dataset):
    """LightGBM custom eval: feval=smape_eval_lgb."""
    return "smape", smape(dataset.get_label(), y_pred), False


def smape_eval_xgb(y_pred, dtrain):
    """XGBoost custom eval: feval=smape_eval_xgb, maximize=False."""
    return "smape", smape(dtrain.get_label(), y_pred)


if __name__ == "__main__":
    # Sanity checks. A perfect prediction scores 0; a 2x over-prediction
    # scores 100*(1/1.5) = 66.67 under the /2 convention.
    assert abs(smape([100, 200], [100, 200])) < 1e-9
    assert abs(smape([100.0], [200.0]) - 66.6666667) < 1e-4
    assert abs(smape([0.0], [0.0])) < 1e-9
    print("metrics.py self-check passed")
    print("smape([100],[200]) =", round(smape([100.0], [200.0]), 4))
