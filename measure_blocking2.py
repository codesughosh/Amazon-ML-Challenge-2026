"""Measure recall of the rare-token blocker.

    python measure_blocking2.py --country India --sample-s1 40000
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.blocking import build_dfreq, candidates  # noqa: E402
from src.normalize import add_normalized_columns  # noqa: E402
from src.utils import timer  # noqa: E402

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
    ap.add_argument("--max-block", type=int, default=300)
    ap.add_argument("--n-rare", type=int, default=3)
    ap.add_argument("--n-digit", type=int, default=3)
    ap.add_argument("--solo-df", type=int, default=400)
    ap.add_argument("--max-df", type=int, default=100000)
    a = ap.parse_args()

    with timer("load + normalise"):
        s1 = load("train", 1, a.country)
        if a.sample_s1:
            s1 = s1.sample(a.sample_s1, random_state=42).reset_index(drop=True)
        s2 = load("train", 2, a.country)
        s3 = load("train", 3, a.country)
        print(f"  S1 {len(s1):,}  S2 {len(s2):,}  S3 {len(s3):,}")
        s1, s2, s3 = (add_normalized_columns(d) for d in (s1, s2, s3))
        for d in (s1, s2, s3):
            d.drop(columns=["business_name", "business_address"], inplace=True)
        gc.collect()

    with timer("document frequencies"):
        # Built from S2+S3 (the pool we search), which is what determines
        # whether a token is a discriminating key.
        dfa, dfn = build_dfreq(s2, s3)
        print(f"  addr vocab {len(dfa):,}   name vocab {len(dfn):,}")

    with timer("ground truth"):
        gt = pd.read_csv(DATA / "train" / "train_ground_truth.tsv", sep="\t",
                         dtype=str, keep_default_na=False)
        gt = gt[gt["source1_entity_id"].isin(set(s1["entity_id"]))]
        pos = dict(zip(s1["entity_id"], range(len(s1))))
        truth = {"S2": set(), "S3": set()}
        for sid, lst in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
            i1 = pos[sid]
            for x in lst.split(","):
                x = x.strip()
                if x:
                    truth[x[:2]].add((i1, x))
        print(f"  true pairs  S2 {len(truth['S2']):,}  S3 {len(truth['S3']):,}")

    kw = dict(n_rare=a.n_rare, n_digit=a.n_digit,
              solo_df=a.solo_df, max_df=a.max_df)
    total_pairs = 0
    print()
    for tag, s23 in (("S2", s2), ("S3", s3)):
        with timer(f"{tag} candidates"):
            pr, nk = candidates(s1, s23, dfa, dfn, max_block=a.max_block, **kw)
        ids = s23["entity_id"].to_numpy()
        T = truth[tag]
        print(f"  {tag}: {len(pr):,} raw pairs ({len(pr)/len(s1):.0f} per S1)")
        order = np.lexsort((-nk, pr[:, 0]))
        prs = pr[order]; nks = nk[order]
        _, starts, counts = np.unique(prs[:, 0], return_index=True, return_counts=True)
        for K in (10, 20, 30, 50, 80, 0):
            if K:
                keep = np.concatenate([np.arange(s, s + min(c, K)) for s, c in zip(starts, counts)])
                sub = prs[keep]
            else:
                sub = prs
            got = {(int(i1), ids[i2]) for i1, i2 in sub}
            rec = len(got & T) / max(1, len(T))
            lbl = f"top-{K}" if K else "all"
            print(f"      {lbl:>7}: recall {rec:6.2%}   {len(sub)/len(s1):6.1f} per S1")
        total_pairs += len(pr)
        del pr, nk, prs, got
        gc.collect()

    print(f"\n  total {total_pairs:,} candidates = {total_pairs/len(s1):.1f} per S1")
    print("  target: recall >= 97%, <= ~60 candidates per S1")


if __name__ == "__main__":
    main()
