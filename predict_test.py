"""Full test-set inference -> matching_results.tsv + candidate_pairs.tsv.

    python -u predict_test.py --train-sample 60000 --chunk 50000

Two phases:

  1. Train one model on a sample of the TRAINING data, pooled across countries.
     Pooled deliberately: the test set contains France, which never appears in
     training, so the model must not learn country-specific behaviour. No
     country feature is used anywhere.

  2. Stream the test set country by country, in chunks of S1 entities, so peak
     memory stays bounded. Each country writes its partial results to disk
     immediately, making the run resumable if it is interrupted.

Output is written to output/ in the exact format the validator expects.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.blocking import (build_dfreq, candidates,        # noqa: E402
                          prepare_index, rescore_and_cut)
from src.er import apply_assignment, select_topk_threshold, tune_threshold  # noqa: E402
from src.features import build_features, compute_idf     # noqa: E402
from src.normalize import add_normalized_columns         # noqa: E402
from src.utils import free, seed_everything, timer       # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
OUT = ROOT / "output"


def load(split, source, country=None, cols=None):
    df = pd.read_csv(DATA / split / f"{split}_source{source}.tsv", sep="\t",
                     dtype=str, keep_default_na=False)
    if country:
        df = df[df["country"] == country]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Phase 1: train
# --------------------------------------------------------------------------

def train_model(a):
    import lightgbm as lgb

    frames_s1, frames_s2, frames_s3 = [], [], []
    per_country = a.train_sample // 2
    for c in ("India", "US"):
        s1 = load("train", 1, c)
        s1 = s1.sample(min(per_country, len(s1)), random_state=42)
        frames_s1.append(s1)
    s1 = pd.concat(frames_s1, ignore_index=True)

    gt = pd.read_csv(DATA / "train" / "train_ground_truth.tsv", sep="\t",
                     dtype=str, keep_default_na=False)
    gt = gt[gt["source1_entity_id"].isin(set(s1["entity_id"]))]
    truth = {s: {x.strip() for x in l.split(",") if x.strip()}
             for s, l in zip(gt["source1_entity_id"], gt["matched_entity_ids"])}
    for s in s1["entity_id"]:
        truth.setdefault(s, set())

    Xs, ys, i1s, cids, sims = [], [], [], [], []
    for c in ("India", "US"):
        s1c = s1[s1["country"] == c].reset_index(drop=True)
        s2, s3 = load("train", 2, c), load("train", 3, c)
        s1c, s2, s3 = (add_normalized_columns(d) for d in (s1c, s2, s3))
        for d in (s1c, s2, s3):
            d.drop(columns=["business_name", "business_address", "country"], inplace=True)
        dfa, dfn = build_dfreq(s2, s3)
        idf = compute_idf(s2["addr_n"], s3["addr_n"], s2["name_n"], s3["name_n"])

        for s23 in (s2, s3):
            with timer(f"  train {c} block"):
                pr, nk = candidates(s1c, s23, dfa, dfn, max_block=a.max_block,
                                    max_per_s1=0, n_rare=a.n_rare, solo_df=a.solo_df)
                pr, nk, sim = rescore_and_cut(s1c, s23, pr, nk, top_k=a.top_k)
            X = build_features(s1c, s23, pr, nk, idf)
            X["blk_sim"] = sim
            ids = s23["entity_id"].to_numpy()[pr[:, 1]]
            sid = s1c["entity_id"].to_numpy()[pr[:, 0]]
            ys.append(np.fromiter((c2 in truth[s] for s, c2 in zip(sid, ids)),
                                  dtype=np.int8, count=len(ids)))
            Xs.append(X)
            i1s.append(pr[:, 0].copy())
            cids.append(ids)
            del pr, nk, X
            free()
        del s2, s3
        free()

    X = pd.concat(Xs, ignore_index=True)
    y = np.concatenate(ys)
    del Xs, ys
    free()
    print(f"  training pairs {len(y):,}  positives {y.sum():,} ({y.mean():.2%})")

    m = lgb.LGBMClassifier(n_estimators=a.trees, learning_rate=0.08,
                           num_leaves=127, min_child_samples=50,
                           colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                           n_jobs=-1, verbose=-1)
    m.fit(X, y)

    # Tune the decision threshold on the same pooled sample. Slightly
    # optimistic, but the earlier held-out run put the optimum at 0.70 and the
    # curve is flat nearby, so the risk is small.
    p = m.predict_proba(X)[:, 1]
    cols = list(X.columns)
    del X
    free()
    return m, cols, float(a.threshold), p, y


# --------------------------------------------------------------------------
# Phase 2: inference
# --------------------------------------------------------------------------

def predict_country(a, model, cols, country):
    print(f"\n{'='*64}\n  {country}\n{'='*64}")
    with timer(f"{country} load+normalise"):
        s1 = load("test", 1, country)
        s2, s3 = load("test", 2, country), load("test", 3, country)
        print(f"    S1 {len(s1):,}  S2 {len(s2):,}  S3 {len(s3):,}")
        s1, s2, s3 = (add_normalized_columns(d) for d in (s1, s2, s3))
        for d in (s1, s2, s3):
            d.drop(columns=["business_name", "business_address", "country"], inplace=True)
        free()

    with timer(f"{country} dfreq+idf"):
        dfa, dfn = build_dfreq(s2, s3)
        idf = compute_idf(s2["addr_n"], s3["addr_n"], s2["name_n"], s3["name_n"])

    indexes = {}
    for tag, s23 in (("S2", s2), ("S3", s3)):
        with timer(f"{country} {tag} key index"):
            indexes[tag] = prepare_index(s23, dfa, dfn, max_block=a.max_block,
                                         n_rare=a.n_rare, solo_df=a.solo_df)

    s1_ids = s1["entity_id"].to_numpy()
    matches = {s: [] for s in s1_ids}
    cands = {s: [] for s in s1_ids}
    # Persisted so a later cross-encoder pass can rescore the uncertain band
    # without repeating blocking, which is the expensive part of this run.
    keep_s1, keep_cid, keep_p = [], [], []

    n = len(s1)
    bar = tqdm(total=n, desc=f"  {country}", unit="ent", unit_scale=True,
               ascii=True, ncols=78, mininterval=0.5)
    for start in range(0, n, a.chunk):
        stop = min(start + a.chunk, n)
        chunk = s1.iloc[start:stop].reset_index(drop=True)
        rows_s1, rows_c, rows_p, rows_uid = [], [], [], []

        for off, (tag, s23) in enumerate((("S2", s2), ("S3", s3))):
            pr, nk = candidates(chunk, s23, dfa, dfn, max_block=a.max_block,
                                max_per_s1=0, n_rare=a.n_rare, solo_df=a.solo_df,
                                k2=indexes[tag])
            if not len(pr):
                continue
            pr, nk, sim = rescore_and_cut(chunk, s23, pr, nk, top_k=a.top_k)
            X = build_features(chunk, s23, pr, nk, idf)
            X["blk_sim"] = sim
            p = model.predict_proba(X[cols])[:, 1]
            ids23 = s23["entity_id"].to_numpy()[pr[:, 1]]

            rows_s1.append(pr[:, 0].copy())
            rows_c.append(ids23)
            rows_p.append(p.astype(np.float32))
            keep_s1.append(chunk["entity_id"].to_numpy()[pr[:, 0]])
            keep_cid.append(ids23)
            keep_p.append(p.astype(np.float32))
            # Unique candidate id space across S2/S3 for the assignment step.
            rows_uid.append(pr[:, 1].astype(np.int64) + off * 10_000_000)

            # candidate_pairs.tsv records what the model actually scored.
            for i, cid in zip(pr[:, 0], ids23):
                cands[chunk["entity_id"].iat[int(i)]].append(cid)
            del pr, nk, X, sim
            free()

        if rows_s1:
            ci = np.concatenate(rows_s1)
            cc = np.concatenate(rows_c)
            cp = np.concatenate(rows_p)
            cu = np.concatenate(rows_uid)
            keep = select_topk_threshold(ci, cp, a.threshold)
            idxs = np.where(keep)[0]
            if len(idxs):
                sub = apply_assignment(ci[idxs], cu[idxs], cp[idxs])
                keep[idxs[~sub]] = False
            for i, cid in zip(ci[keep], cc[keep]):
                matches[chunk["entity_id"].iat[int(i)]].append(cid)
            del ci, cc, cp, cu, keep
        bar.update(stop - start)
        free()
    bar.close()

    if keep_s1:
        scored = pd.DataFrame({
            "s1_id": np.concatenate(keep_s1),
            "cand_id": np.concatenate(keep_cid),
            "prob": np.concatenate(keep_p),
        })
        sp = ROOT / "data" / f"scored_{country}.parquet"
        sp.parent.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(sp, index=False)
        print(f"    saved {len(scored):,} scored pairs -> {sp.name}", flush=True)
        del scored, keep_s1, keep_cid, keep_p

    del s2, s3, indexes
    free()
    return s1_ids, matches, cands


def write_tsv(path: Path, ids, mapping, col: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s in ids:
            v = mapping.get(s, [])
            # De-duplicate while preserving order; the validator rejects dupes.
            seen, outv = set(), []
            for x in v:
                if x not in seen:
                    seen.add(x)
                    outv.append(x)
            f.write(f"{s}\t{','.join(outv)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-sample", type=int, default=60000)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-block", type=int, default=1500)
    ap.add_argument("--n-rare", type=int, default=8)
    ap.add_argument("--solo-df", type=int, default=4000)
    ap.add_argument("--trees", type=int, default=400)
    ap.add_argument("--threshold", type=float, default=0.675)
    ap.add_argument("--countries", default="France,US,India",
                    help="smallest first, so a partial run still yields output")
    a = ap.parse_args()
    seed_everything()

    with timer("PHASE 1: train"):
        model, cols, thr, _, _ = train_model(a)
    print(f"  model trained, threshold {thr}")

    all_ids, all_m, all_c = [], {}, {}
    for country in a.countries.split(","):
        ids, m, c = predict_country(a, model, cols, country)
        all_ids.append(ids)
        all_m.update(m)
        all_c.update(c)
        # Write after every country so an interrupted run still leaves output.
        ids_so_far = np.concatenate(all_ids)
        write_tsv(OUT / "matching_results.tsv", ids_so_far, all_m, "matched_entity_ids")
        write_tsv(OUT / "candidate_pairs.tsv", ids_so_far, all_c, "candidate_entity_ids")
        print(f"  wrote partial output ({len(ids_so_far):,} entities)")
        free()

    ids = np.concatenate(all_ids)
    n_match = sum(1 for s in ids if all_m.get(s))
    npred = np.array([len(all_m.get(s, [])) for s in ids])
    print(f"\n{'='*64}")
    print(f"  entities      {len(ids):,}")
    print(f"  with matches  {n_match:,} ({n_match/len(ids):.1%})")
    print(f"  empty         {len(ids)-n_match:,} ({1-n_match/len(ids):.1%})  "
          f"(train singleton rate 5.6%)")
    print(f"  mean matches  {npred.mean():.2f}  (train mean 3.46)")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
