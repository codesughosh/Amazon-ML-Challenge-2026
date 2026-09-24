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

    VERIFIED against the official 2025 problem statement, which defines

        SMAPE = (1/n) * sum |pred - actual| / ((|actual| + |pred|) / 2)

    and gives the worked example actual=100, pred=120 -> 18.18%. This
    implementation reproduces that exactly (see the self-check at the bottom
    of this file). The /2 matters: the convention without it doubles the score.

    2025 reference points: winning 39.7, AIR-80 43.28.
    Re-verify against the 2026 statement's own example before trusting it.
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

def optimal_multiplier(y_true, oof_pred, metric: str = "smape",
                       lo: float = 0.85, hi: float = 1.25, steps: int = 81):
    """Find the global scaling factor that minimises the metric on OOF.

    A genuinely free win on relative-error metrics, but only when fitted on
    out-of-fold predictions. Fit it on train predictions and you are just
    measuring your own overfitting.

    Returns (best_multiplier, best_score, baseline_score).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    oof_pred = np.asarray(oof_pred, dtype=np.float64)
    baseline = score(metric, y_true, oof_pred)

    best_m, best_s = 1.0, baseline
    for m in np.linspace(lo, hi, steps):
        s = score(metric, y_true, oof_pred * m)
        if is_better(metric, s, best_s):
            best_m, best_s = float(m), s

    gain = abs(best_s - baseline)
    print(f"multiplier {best_m:.3f}: {metric} {baseline:.5f} -> {best_s:.5f} "
          f"({gain:.5f} gain)")
    if gain < 0.01:
        print("  gain is negligible - skip it, it is probably fold noise")
    return best_m, best_s, baseline


def smape_eval_lgb(y_pred, dataset):
    """LightGBM custom eval: feval=smape_eval_lgb."""
    return "smape", smape(dataset.get_label(), y_pred), False


def smape_eval_xgb(y_pred, dtrain):
    """XGBoost custom eval: feval=smape_eval_xgb, maximize=False."""
    return "smape", smape(dtrain.get_label(), y_pred)


if __name__ == "__main__":
    # The official 2025 worked example: actual 100, predicted 120 -> 18.18%.
    # This is the check that proves the convention is right.
    assert abs(smape([100.0], [120.0]) - 18.1818) < 1e-3, "official example mismatch"

    assert abs(smape([100, 200], [100, 200])) < 1e-9          # perfect -> 0
    assert abs(smape([100.0], [200.0]) - 66.6666667) < 1e-4   # 2x over -> 66.67
    assert abs(smape([0.0], [0.0])) < 1e-9                    # 0/0 -> 0, not NaN
    assert abs(smape([1.0], [0.0]) - 200.0) < 1e-9            # upper bound

    # SMAPE is NOT symmetric in the error direction, despite the name. For the
    # same absolute error, UNDER-predicting costs more, because the denominator
    # shrinks with the prediction:
    #     actual 100, pred 150 -> 50/125 = 40.00
    #     actual 100, pred  50 -> 50/ 75 = 66.67
    # So the SMAPE-optimal point estimate sits slightly ABOVE the conditional
    # median, and a global multiplier a little over 1.0 is the thing to test.
    # Sweep it on OOF (see optimal_multiplier) rather than assuming a value.
    over, under = smape([100.0], [150.0]), smape([100.0], [50.0])
    assert under > over, "under-prediction should cost more than over-prediction"
    print(f"over-predicting by 50  -> {over:.2f}")
    print(f"under-predicting by 50 -> {under:.2f}")
    print("metrics.py self-check passed (official example reproduced)")
