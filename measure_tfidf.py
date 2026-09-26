"""Compare blocking recall: rare-token vs TF-IDF vs union.

    python measure_tfidf.py --country India --sample-s1 20000 --top-k 30
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
from src.normalize import add_normalized_columns          # noqa: E402
from src.tfidf_block import build_vectorizer, topk_neighbours, union_pairs  # noqa: E402
from src.utils import free, timer                         # noqa: E402

from src.paths import DATA  # portable: see src/paths.py
def load(split, source, country=None):
    df = pd.read_csv(DATA / split / f"{split}_source{source}.tsv", sep="\t",
                     dtype=str, keep_default_na=False)
    if country:
        df = df[df["country"] == country]
    return df.reset_index(drop=True)


def recall_of(pairs, ids23, truth_set):
    got = {(int(i1), ids23[i2]) for i1, i2 in pairs}
    return len(got & truth_set) / max(1, len(truth_set))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="India")
    ap.add_argument("--sample-s1", type=int, default=20000)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=0.18)
    ap.add_argument("--max-features", type=int, default=300000)
    ap.add_argument("--max-per-s1", type=int, default=50)
    a = ap.parse_args()

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

    with timer("document frequencies"):
        dfa, dfn = build_dfreq(s2, s3)

    # Combined text: one representation carrying both fields, so a record with
    # a mangled name can still be found through its address and vice versa.
    def combo(d):
        return (d["name_n"] + " " + d["addr_n"]).tolist()

    with timer("fit tfidf (on S2+S3 pool)"):
        vec = build_vectorizer(combo(s2) + combo(s3), max_features=a.max_features)
        print(f"  vocab {len(vec.vocabulary_):,}")

    print()
    for tag, s23 in (("S2", s2), ("S3", s3)):
        ids = s23["entity_id"].to_numpy()
        T = truth[tag]
        print(f"--- {tag} ---")

        with timer(f"{tag} rare-token"):
            pr_rt, _ = candidates(s1, s23, dfa, dfn, max_block=800,
                                  max_per_s1=a.max_per_s1, n_rare=4, solo_df=1500)
        r_rt = recall_of(pr_rt, ids, T)
        print(f"  rare-token : recall {r_rt:6.2%}   {len(pr_rt)/len(s1):5.1f} per S1")

        with timer(f"{tag} tfidf"):
            pr_tf, sims = topk_neighbours(vec, combo(s1), combo(s23),
                                          top_k=a.top_k, threshold=a.threshold)
        r_tf = recall_of(pr_tf, ids, T)
        print(f"  tfidf      : recall {r_tf:6.2%}   {len(pr_tf)/len(s1):5.1f} per S1")

        pr_u = union_pairs(pr_rt, pr_tf)
        r_u = recall_of(pr_u, ids, T)
        print(f"  UNION      : recall {r_u:6.2%}   {len(pr_u)/len(s1):5.1f} per S1")
        print(f"               tfidf adds {r_u - r_rt:+.2%} over rare-token alone")

        del pr_rt, pr_tf, pr_u, sims
        free()
        print()


if __name__ == "__main__":
    main()
