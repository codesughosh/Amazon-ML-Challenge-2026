"""Does similarity re-ranking beat shared-key-count ranking?

    python -u measure_rescore.py --country India --sample-s1 20000

Compares, at several top-k cuts:
  A) rank by shared-key count   (current)
  B) rank by string similarity  (rescore_and_cut)
against the raw "keep everything" ceiling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.blocking import build_dfreq, candidates, rescore_and_cut  # noqa: E402
from src.normalize import add_normalized_columns                   # noqa: E402
from src.utils import free, timer                                  # noqa: E402

from src.paths import DATA  # portable: see src/paths.py
def load(split, source, country=None):
    df = pd.read_csv(DATA / split / f"{split}_source{source}.tsv", sep="\t",
                     dtype=str, keep_default_na=False)
    if country:
        df = df[df["country"] == country]
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="India")
    ap.add_argument("--sample-s1", type=int, default=20000)
    ap.add_argument("--max-block", type=int, default=800)
    ap.add_argument("--n-rare", type=int, default=4)
    ap.add_argument("--solo-df", type=int, default=1500)
    a = ap.parse_args()

    with timer("load + normalise"):
        s1 = load("train", 1, a.country)
        if a.sample_s1:
            s1 = s1.sample(a.sample_s1, random_state=42).reset_index(drop=True)
        s2, s3 = load("train", 2, a.country), load("train", 3, a.country)
        s1, s2, s3 = (add_normalized_columns(d) for d in (s1, s2, s3))
        for d in (s1, s2, s3):
            d.drop(columns=["business_name", "business_address", "country"], inplace=True)
        free()

    with timer("dfreq"):
        dfa, dfn = build_dfreq(s2, s3)

    gt = pd.read_csv(DATA / "train" / "train_ground_truth.tsv", sep="\t",
                     dtype=str, keep_default_na=False)
    gt = gt[gt["source1_entity_id"].isin(set(s1["entity_id"]))]
    pos = dict(zip(s1["entity_id"], range(len(s1))))
    truth = {"S2": set(), "S3": set()}
    for sid, lst in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        for x in lst.split(","):
            x = x.strip()
            if x:
                truth[x[:2]].add((pos[sid], x))

    def recall(pr, ids, T):
        return len({(int(i), ids[j]) for i, j in pr} & T) / max(1, len(T))

    for tag, s23 in (("S2", s2), ("S3", s3)):
        ids = s23["entity_id"].to_numpy()
        T = truth[tag]
        print(f"\n--- {tag} ({len(T):,} true pairs) ---")

        with timer("  block (raw, no cut)"):
            pr, nk = candidates(s1, s23, dfa, dfn, max_block=a.max_block,
                                max_per_s1=0, n_rare=a.n_rare, solo_df=a.solo_df)
        print(f"  raw: {len(pr)/len(s1):.0f} per S1   CEILING recall {recall(pr, ids, T):.2%}")

        # A) shared-key-count ranking
        order = np.lexsort((-nk, pr[:, 0]))
        pr_k = pr[order]
        _, st, ct = np.unique(pr_k[:, 0], return_index=True, return_counts=True)

        # B) similarity ranking
        with timer("  rescore"):
            pr_s, _, sc = rescore_and_cut(s1, s23, pr, nk, top_k=200)
        _, st2, ct2 = np.unique(pr_s[:, 0], return_index=True, return_counts=True)

        print(f"  {'k':>5} {'by key-count':>14} {'by similarity':>15}   gain")
        for K in (20, 30, 50, 80, 120):
            kA = np.concatenate([np.arange(s, s + min(c, K)) for s, c in zip(st, ct)])
            kB = np.concatenate([np.arange(s, s + min(c, K)) for s, c in zip(st2, ct2)])
            rA = recall(pr_k[kA], ids, T)
            rB = recall(pr_s[kB], ids, T)
            print(f"  {K:>5} {rA:>13.2%} {rB:>14.2%}   {rB-rA:+.2%}")
        del pr, nk, pr_k, pr_s
        free()


if __name__ == "__main__":
    main()
