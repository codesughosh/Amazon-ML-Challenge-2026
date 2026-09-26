"""Export labelled candidate pairs for cross-encoder training.

    python -u export_pairs.py --country India --sample-s1 25000

Writes data/ce_pairs.parquet with columns text1, text2, label.

Run this BEFORE the long CPU inference job: it needs the CPU for ~10 minutes,
after which the cross-encoder trains on the GPU while inference uses the CPU,
so both resources work at once instead of queueing.

The text pair fed to the model keeps the raw (non-romanised) strings as well as
the normalised ones. A multilingual encoder can align Bengali with Latin
directly, which is the whole point of using one - romanising first would throw
away the signal we are buying it for.
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
from src.utils import free, seed_everything, timer                 # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"


def load(split, source, country=None):
    df = pd.read_csv(DATA / split / f"{split}_source{source}.tsv", sep="\t",
                     dtype=str, keep_default_na=False)
    if country:
        df = df[df["country"] == country]
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="India")
    ap.add_argument("--sample-s1", type=int, default=25000)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-block", type=int, default=1500)
    ap.add_argument("--n-rare", type=int, default=8)
    ap.add_argument("--solo-df", type=int, default=4000)
    ap.add_argument("--neg-per-pos", type=float, default=4.0,
                    help="negatives kept per positive; the raw ratio is ~30:1")
    ap.add_argument("--out", default="data/ce_pairs.parquet")
    a = ap.parse_args()
    seed_everything()

    with timer("load + normalise"):
        s1 = load("train", 1, a.country).sample(a.sample_s1, random_state=42).reset_index(drop=True)
        s2, s3 = load("train", 2, a.country), load("train", 3, a.country)
        s1, s2, s3 = (add_normalized_columns(d) for d in (s1, s2, s3))
        free()

    with timer("dfreq"):
        dfa, dfn = build_dfreq(s2, s3)

    gt = pd.read_csv(DATA / "train" / "train_ground_truth.tsv", sep="\t",
                     dtype=str, keep_default_na=False)
    gt = gt[gt["source1_entity_id"].isin(set(s1["entity_id"]))]
    truth = {s: {x.strip() for x in l.split(",") if x.strip()}
             for s, l in zip(gt["source1_entity_id"], gt["matched_entity_ids"])}
    for s in s1["entity_id"]:
        truth.setdefault(s, set())

    rng = np.random.default_rng(0)
    parts = []
    for tag, s23 in (("S2", s2), ("S3", s3)):
        with timer(f"{tag} candidates"):
            pr, nk = candidates(s1, s23, dfa, dfn, max_block=a.max_block,
                                max_per_s1=0, n_rare=a.n_rare, solo_df=a.solo_df)
            pr, nk, _ = rescore_and_cut(s1, s23, pr, nk, top_k=a.top_k)

        sid = s1["entity_id"].to_numpy()[pr[:, 0]]
        cid = s23["entity_id"].to_numpy()[pr[:, 1]]
        y = np.fromiter((c in truth[s] for s, c in zip(sid, cid)),
                        dtype=np.int8, count=len(cid))

        # Keep every positive; subsample negatives. The raw candidate set is
        # ~30:1 negative, which wastes most of the GPU budget on easy negatives.
        # Hard negatives survive preferentially because they are what blocking
        # ranked highly in the first place.
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        n_keep = min(len(neg), int(len(pos) * a.neg_per_pos))
        neg = rng.choice(neg, n_keep, replace=False)
        sel = np.concatenate([pos, neg])
        rng.shuffle(sel)

        # Raw strings, not romanised: the multilingual encoder aligns scripts
        # itself, and romanising first destroys exactly that signal.
        n1 = s1["business_name"].to_numpy()[pr[sel, 0]]
        a1 = s1["business_address"].to_numpy()[pr[sel, 0]]
        n2 = s23["business_name"].to_numpy()[pr[sel, 1]]
        a2 = s23["business_address"].to_numpy()[pr[sel, 1]]

        parts.append(pd.DataFrame({
            "text1": [f"{x} | {y}" for x, y in zip(n1, a1)],
            "text2": [f"{x} | {y}" for x, y in zip(n2, a2)],
            "label": y[sel].astype(np.int8),
            "s1_id": sid[sel],
        }))
        print(f"    {tag}: {len(pos):,} pos + {n_keep:,} neg")
        del pr, nk, sel
        free()

    out = pd.concat(parts, ignore_index=True)
    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p, index=False)
    print(f"\nwrote {p}  {len(out):,} pairs  ({out.label.mean():.1%} positive)")
    print(out.head(3).to_string())


if __name__ == "__main__":
    main()
