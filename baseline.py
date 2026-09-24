"""Hour-1 baseline. Get a score on the board before building anything clever.

    python baseline.py --data data --text-col catalog_content --target price

Runs two models in ascending order of ambition and writes a submission for each:
  1. constant (train median)      - proves the submission format is accepted
  2. TF-IDF word+char -> LightGBM - a real floor, usually surprisingly strong

Both are CPU-only and finish in minutes. Adjust the column names via flags once
the 2026 statement is out; the defaults match the 2025 schema.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cv import Tracker, make_folds, run_cv  # noqa: E402
from src.metrics import optimal_multiplier, score  # noqa: E402
from src.submit import write_submission  # noqa: E402
from src.utils import describe_df, load_cached, seed_everything, timer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data", help="directory holding train/test csv")
    p.add_argument("--train", default="train.csv")
    p.add_argument("--test", default="test.csv")
    p.add_argument("--sample-sub", default="sample_test_out.csv",
                   help="organisers' sample output; used to validate format")
    p.add_argument("--id-col", default="sample_id")
    p.add_argument("--text-col", default="catalog_content")
    p.add_argument("--target", default="price")
    p.add_argument("--metric", default="smape")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--log-target", action="store_true", default=True,
                   help="train on log1p(target); right default for relative-error metrics")
    p.add_argument("--no-log-target", dest="log_target", action="store_false")
    p.add_argument("--max-features-word", type=int, default=200_000)
    p.add_argument("--max-features-char", type=int, default=100_000)
    p.add_argument("--sample", type=int, default=0,
                   help="use only N rows, for a fast smoke run")
    return p.parse_args()


def main():
    a = parse_args()
    seed_everything()
    d = Path(a.data)

    # ---------------------------------------------------------------- load
    with timer("load"):
        train = load_cached(d / a.train)
        test = load_cached(d / a.test)
        if a.sample:
            train = train.sample(a.sample, random_state=42).reset_index(drop=True)

    describe_df(train, "train")
    print(f"\ntest: {test.shape[0]:,} rows")

    for col in (a.id_col, a.text_col, a.target):
        if col not in train.columns:
            sys.exit(f"\nColumn '{col}' not in train. Columns are: {list(train.columns)}\n"
                     f"Re-run with the right --id-col / --text-col / --target.")

    sample_path = d / a.sample_sub
    sample_path = sample_path if sample_path.exists() else None
    if sample_path is None:
        print(f"\nWARNING: {a.sample_sub} not found - submission format will not be "
              f"validated against the organisers' file. Find it before submitting.")

    y = train[a.target].values
    print(f"\ntarget: min={y.min():.2f} median={np.median(y):.2f} "
          f"mean={y.mean():.2f} max={y.max():.2f}")
    print(f"skew={pd.Series(y).skew():.2f}  "
          f"({'log transform recommended' if pd.Series(y).skew() > 1 else 'roughly symmetric'})")

    tracker = Tracker(metric=a.metric)

    # ------------------------------------------------------- 1. constant
    print("\n" + "=" * 60 + "\n MODEL 1: constant (train median)\n" + "=" * 60)
    const = float(np.median(y))
    oof_const = np.full(len(train), const)
    s = score(a.metric, y, oof_const)
    tracker.log("constant_median", s, f"value={const:.2f}")

    write_submission(test[a.id_col].values, np.full(len(test), const),
                     a.id_col, a.target, "01_constant",
                     sample_path=sample_path, clip_min=0.01)

    # ------------------------------------------- 2. TF-IDF -> LightGBM
    print("\n" + "=" * 60 + "\n MODEL 2: TF-IDF (word+char) -> LightGBM\n" + "=" * 60)

    txt_tr = train[a.text_col].fillna("").astype(str)
    txt_te = test[a.text_col].fillna("").astype(str)

    with timer("tfidf"):
        # Word n-grams catch brand and product-type tokens; char n-grams catch
        # pack sizes, units and typos that word tokenisation destroys.
        vw = TfidfVectorizer(ngram_range=(1, 2), max_features=a.max_features_word,
                             min_df=3, sublinear_tf=True, strip_accents="unicode")
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                             max_features=a.max_features_char, min_df=3,
                             sublinear_tf=True)
        Xw_tr, Xw_te = vw.fit_transform(txt_tr), vw.transform(txt_te)
        Xc_tr, Xc_te = vc.fit_transform(txt_tr), vc.transform(txt_te)

    # Cheap hand features that text vectorisers miss entirely.
    def extras(s: pd.Series) -> np.ndarray:
        return np.vstack([
            s.str.len().values,
            s.str.count(r"\s").values,
            s.str.count(r"\d").values,
            s.str.count(r"[A-Z]").values,
        ]).T.astype(np.float32)

    X_tr = hstack([Xw_tr, Xc_tr, extras(txt_tr)]).tocsr()
    X_te = hstack([Xw_te, Xc_te, extras(txt_te)]).tocsr()
    print(f"feature matrix: {X_tr.shape}")

    folds = make_folds(train, n_splits=a.folds, strategy="stratified",
                       target_col=a.target)

    def fit_predict(tr, va, te, fold):
        import lightgbm as lgb

        tr_i, va_i = tr.index.values, va.index.values
        ytr = np.log1p(tr[a.target].values) if a.log_target else tr[a.target].values

        m = lgb.LGBMRegressor(
            n_estimators=3000, learning_rate=0.05, num_leaves=127,
            colsample_bytree=0.4, subsample=0.8, subsample_freq=1,
            min_child_samples=20, verbose=-1, n_jobs=-1,
        )
        yva = np.log1p(va[a.target].values) if a.log_target else va[a.target].values
        m.fit(X_tr[tr_i], ytr,
              eval_set=[(X_tr[va_i], yva)],
              callbacks=[lgb.early_stopping(100, verbose=False)])

        vp = m.predict(X_tr[va_i])
        tp = m.predict(X_te)
        if a.log_target:
            vp, tp = np.expm1(vp), np.expm1(tp)
        return np.clip(vp, 0.01, None), np.clip(tp, 0.01, None)

    train = train.reset_index(drop=True)
    oof, test_pred, _, oof_score = run_cv(
        train, folds, fit_predict, a.target, a.metric, test_df=test
    )
    tracker.log("tfidf_lgbm", oof_score,
                f"log_target={a.log_target}, {X_tr.shape[1]} features")

    # ---------------------------------------------- 3. multiplier sweep
    print("\n--- post-processing ---")
    mult, mult_score, _ = optimal_multiplier(train[a.target].values, oof, a.metric)
    if abs(mult_score - oof_score) > 0.01:
        tracker.log("tfidf_lgbm_x_mult", mult_score, f"multiplier={mult:.3f}")
        test_pred = test_pred * mult

    write_submission(test[a.id_col].values, test_pred, a.id_col, a.target,
                     "02_tfidf_lgbm", sample_path=sample_path, clip_min=0.01)

    np.save("oof_tfidf_lgbm.npy", oof)   # keep for blending later

    print("\n" + "=" * 60)
    print(tracker.leaderboard().to_string(index=False))
    print("=" * 60)
    print("\nSubmit 01_constant FIRST to confirm the format is accepted.")
    print("Then submit 02_tfidf_lgbm. That score is your floor - every later")
    print("model must beat it on OOF, not on the public leaderboard.")


if __name__ == "__main__":
    main()
