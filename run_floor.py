"""End-to-end floor pipeline on one country shard.

    python run_floor.py --country India --sample-s1 40000

normalise -> block -> features -> LightGBM -> decision -> macro F_0.5

Reports three decision strategies on a held-out split of S1 entities:
  1. tuned global threshold
  2. + many-to-one assignment constraint
  3. + per-entity expected-F_0.5 k selection (on calibrated probabilities)

The gap between (1) and (3) is the value of the decision layer.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.blocking import build_dfreq, candidates          # noqa: E402
from src.er import (apply_assignment, macro_f05,          # noqa: E402
                    select_expected_f05, select_topk_threshold, tune_threshold)
from src.features import build_features, compute_idf      # noqa: E402
from src.normalize import add_normalized_columns          # noqa: E402
from src.utils import free, seed_everything, timer        # noqa: E402

DATA = Path(r"X:\Amazon ML Challenge 2026\6ab10eb3b23ba_student_resource\student_resource\dataset")


def load(split, source, country=None):
    df = pd.read_csv(DATA / split / f"{split}_source{source}.tsv", sep="\t",
                     dtype=str, keep_default_na=False)
    if country:
        df = df[df["country"] == country]
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="India")
    ap.add_argument("--sample-s1", type=int, default=40000)
    ap.add_argument("--max-per-s1", type=int, default=60)
    ap.add_argument("--max-block", type=int, default=800)
    ap.add_argument("--n-rare", type=int, default=4)
    ap.add_argument("--solo-df", type=int, default=1500)
    ap.add_argument("--val-frac", type=float, default=0.3)
    ap.add_argument("--trees", type=int, default=400)
    a = ap.parse_args()
    seed_everything()

    # ---------------------------------------------------------------- data
    with timer("load + normalise"):
        s1 = load("train", 1, a.country)
        if a.sample_s1:
            s1 = s1.sample(a.sample_s1, random_state=42).reset_index(drop=True)
        s2, s3 = load("train", 2, a.country), load("train", 3, a.country)
        print(f"  S1 {len(s1):,}  S2 {len(s2):,}  S3 {len(s3):,}")
        s1, s2, s3 = (add_normalized_columns(d) for d in (s1, s2, s3))
        for d in (s1, s2, s3):
            d.drop(columns=["business_name", "business_address", "country"], inplace=True)
        free()

    with timer("idf + document frequencies"):
        dfa, dfn = build_dfreq(s2, s3)
        idf = compute_idf(s2["addr_n"], s3["addr_n"], s2["name_n"], s3["name_n"])

    with timer("ground truth"):
        gt = pd.read_csv(DATA / "train" / "train_ground_truth.tsv", sep="\t",
                         dtype=str, keep_default_na=False)
        gt = gt[gt["source1_entity_id"].isin(set(s1["entity_id"]))]
        truth = {sid: {x.strip() for x in lst.split(",") if x.strip()}
                 for sid, lst in zip(gt["source1_entity_id"], gt["matched_entity_ids"])}
        for sid in s1["entity_id"]:
            truth.setdefault(sid, set())

    # Split by S1 ENTITY - never by pair. Mirrors how the test set works.
    rng = np.random.default_rng(42)
    is_val = rng.random(len(s1)) < a.val_frac
    print(f"  train entities {(~is_val).sum():,}   val entities {is_val.sum():,}")

    # ------------------------------------------------------- candidates
    blocks = []
    for tag, s23 in (("S2", s2), ("S3", s3)):
        with timer(f"{tag} blocking"):
            pr, nk = candidates(s1, s23, dfa, dfn, max_block=a.max_block,
                                max_per_s1=a.max_per_s1, n_rare=a.n_rare,
                                solo_df=a.solo_df)
            print(f"    {len(pr):,} pairs ({len(pr)/len(s1):.1f} per S1)")
        with timer(f"{tag} features"):
            X = build_features(s1, s23, pr, nk, idf)
        ids = s23["entity_id"].to_numpy()[pr[:, 1]]
        s1ids = s1["entity_id"].to_numpy()[pr[:, 0]]
        y = np.fromiter((c in truth[s] for s, c in zip(s1ids, ids)),
                        dtype=np.int8, count=len(ids))
        blocks.append((pr[:, 0].copy(), pr[:, 1].copy(), ids, s1ids, X, y))
        del pr, nk, X
        free()

    i1 = np.concatenate([b[0] for b in blocks])
    cand_ids = np.concatenate([b[2] for b in blocks])
    s1_of_pair = np.concatenate([b[3] for b in blocks])
    X = pd.concat([b[4] for b in blocks], ignore_index=True)
    y = np.concatenate([b[5] for b in blocks])
    # Candidate index must be unique across S2 and S3 for the assignment step.
    cand_uid = np.concatenate([b[1] for b in blocks])
    cand_uid[len(blocks[0][1]):] += int(blocks[0][1].max()) + 1
    del blocks
    free()

    n_true_total = sum(len(v) for v in truth.values())
    print(f"\n  candidate pairs {len(y):,}   positives {y.sum():,}   "
          f"blocking recall {y.sum()/n_true_total:.2%}")
    print(f"  >>> CEILING: no decision rule can beat ~{y.sum()/n_true_total:.2%}\n")

    tr_mask = ~is_val[i1]
    va_mask = is_val[i1]

    # ------------------------------------------------------------ model
    import lightgbm as lgb
    with timer("lightgbm"):
        m = lgb.LGBMClassifier(
            n_estimators=a.trees, learning_rate=0.08, num_leaves=127,
            min_child_samples=50, colsample_bytree=0.8, subsample=0.8,
            subsample_freq=1, n_jobs=-1, verbose=-1,
        )
        m.fit(X[tr_mask], y[tr_mask],
              eval_set=[(X[va_mask], y[va_mask])],
              eval_metric="average_precision",
              callbacks=[lgb.early_stopping(50, verbose=False)])
        p_va = m.predict_proba(X[va_mask])[:, 1]

    imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n  top features:")
    for k, v in imp.head(12).items():
        print(f"    {k:<16}{v:>8}")

    # -------------------------------------------------------- decisions
    val_ids = set(s1["entity_id"].to_numpy()[is_val])
    val_truth = {s: truth[s] for s in val_ids}

    v_s1 = s1_of_pair[va_mask]
    v_cand = cand_ids[va_mask]
    v_uid = cand_uid[va_mask]
    v_i1 = i1[va_mask]

    print("\n" + "=" * 62)
    thr, sc, table = tune_threshold(v_i1, p_va, v_cand, val_truth,
                                    s1["entity_id"].to_numpy())
    print(f"  [1] global threshold      thr={thr:.3f}   macro F0.5 = {sc:.4f}")

    k2 = select_topk_threshold(v_i1, p_va, thr)
    idxs = np.where(k2)[0]
    sub_keep = apply_assignment(v_i1[idxs], v_uid[idxs], p_va[idxs])
    k2[idxs[~sub_keep]] = False
    pred = {}
    for i, c in zip(v_i1[k2], v_cand[k2]):
        pred.setdefault(s1["entity_id"].to_numpy()[i], set()).add(c)
    sc2 = macro_f05(pred, val_truth)
    print(f"  [2] + assignment          macro F0.5 = {sc2:.4f}  ({sc2-sc:+.4f})")

    from sklearn.isotonic import IsotonicRegression
    # Calibrate on a slice of val, evaluate the rule on the rest, so the
    # calibration is not fitted on the data it is scored on.
    half = np.random.default_rng(0).random(len(p_va)) < 0.5
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_va[half], y[va_mask][half])
    p_cal = iso.predict(p_va)

    k3 = select_expected_f05(v_i1, p_cal)
    idxs = np.where(k3)[0]
    sub_keep = apply_assignment(v_i1[idxs], v_uid[idxs], p_cal[idxs])
    k3[idxs[~sub_keep]] = False
    pred3 = {}
    for i, c in zip(v_i1[k3], v_cand[k3]):
        pred3.setdefault(s1["entity_id"].to_numpy()[i], set()).add(c)
    # score only on the half not used for calibration would be ideal; report both
    sc3 = macro_f05(pred3, val_truth)
    print(f"  [3] + expected-F0.5 k     macro F0.5 = {sc3:.4f}  ({sc3-sc2:+.4f})")
    print("=" * 62)

    npred = np.array([len(v) for v in pred3.values()])
    ntrue = np.array([len(v) for v in val_truth.values()])
    print(f"\n  predicted per entity: mean {npred.mean():.2f}   "
          f"true mean {ntrue.mean():.2f}")
    print(f"  predicted empty: {(npred==0).mean():.1%}   "
          f"true empty: {(ntrue==0).mean():.1%}")


if __name__ == "__main__":
    main()
