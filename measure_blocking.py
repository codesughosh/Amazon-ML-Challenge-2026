"""Measure blocking recall - the hard ceiling on the final score.

    python measure_blocking.py --country India --max-block 200

No model can recover a true pair that blocking never proposed, so this number
is measured before any modelling happens. Reports, per blocking key and for the
union: recall (fraction of true pairs captured) and candidates per S1 entity.

Runs one country shard at a time to stay inside 16 GB of RAM.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.normalize import add_normalized_columns  # noqa: E402
from src.utils import timer  # noqa: E402

DATA = Path(r"X:\Amazon ML Challenge 2026\6ab10eb3b23ba_student_resource\student_resource\dataset")


def load(split: str, source: int, country: str | None) -> pd.DataFrame:
    df = pd.read_csv(DATA / split / f"{split}_source{source}.tsv", sep="\t",
                     dtype=str, keep_default_na=False)
    if country:
        df = df[df["country"] == country]
    return df.reset_index(drop=True)


# Each entry: (name, list of columns whose concatenation forms the key).
# A key is skipped for a record when any component is empty.
KEY_DEFS = [
    ("addr_full",      ["addr_n"]),
    ("digits_tok1",    ["addr_digits", "addr_tok1"]),
    ("house_tok1",     ["addr_house", "addr_tok1"]),
    ("pin_tok1",       ["addr_pin", "addr_tok1"]),
    ("name_exact",     ["name_n"]),
    ("nametok_house",  ["name_tok1", "addr_house"]),
]


def build_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    k = df[cols[0]].astype(str)
    for c in cols[1:]:
        k = k + "|" + df[c].astype(str)
    # Blank out keys with any empty component - they would collide en masse.
    bad = np.zeros(len(df), dtype=bool)
    for c in cols:
        bad |= (df[c].astype(str).str.len() == 0).to_numpy()
    k = k.where(~bad, "")
    return k


def pairs_for_key(s1: pd.DataFrame, s23: pd.DataFrame, cols: list[str],
                  max_block: int) -> np.ndarray:
    """Return an (n, 2) int32 array of (s1_idx, s23_idx) candidate pairs."""
    k1 = build_key(s1, cols)
    k2 = build_key(s23, cols)

    left = pd.DataFrame({"i1": np.arange(len(s1), dtype=np.int32), "k": k1})
    right = pd.DataFrame({"i2": np.arange(len(s23), dtype=np.int32), "k": k2})
    left = left[left["k"] != ""]
    right = right[right["k"] != ""]

    # Drop oversized blocks: they are generic keys that would blow up the pair
    # count without carrying information.
    sz = right.groupby("k", sort=False)["i2"].transform("size")
    right = right[sz <= max_block]
    if right.empty or left.empty:
        return np.empty((0, 2), dtype=np.int32)

    keep = right["k"].unique()
    left = left[left["k"].isin(keep)]
    if left.empty:
        return np.empty((0, 2), dtype=np.int32)

    m = left.merge(right, on="k", copy=False)
    return m[["i1", "i2"]].to_numpy(dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="India")
    ap.add_argument("--max-block", type=int, default=200)
    ap.add_argument("--sample-s1", type=int, default=0,
                    help="evaluate recall on N sampled S1 entities (0 = all)")
    a = ap.parse_args()

    with timer("load + normalise"):
        s1 = load("train", 1, a.country)
        if a.sample_s1:
            s1 = s1.sample(a.sample_s1, random_state=42).reset_index(drop=True)
        s2 = load("train", 2, a.country)
        s3 = load("train", 3, a.country)
        print(f"  S1 {len(s1):,}   S2 {len(s2):,}   S3 {len(s3):,}")
        s1 = add_normalized_columns(s1)
        s2 = add_normalized_columns(s2)
        s3 = add_normalized_columns(s3)
        for d in (s1, s2, s3):
            d.drop(columns=["business_name", "business_address"], inplace=True)
        gc.collect()

    with timer("ground truth"):
        gt = pd.read_csv(DATA / "train" / "train_ground_truth.tsv", sep="\t",
                         dtype=str, keep_default_na=False)
        gt = gt[gt["source1_entity_id"].isin(set(s1["entity_id"]))]
        s1_pos = dict(zip(s1["entity_id"], range(len(s1))))
        truth = set()
        n_true = 0
        for sid, lst in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
            i1 = s1_pos[sid]
            for x in lst.split(","):
                x = x.strip()
                if x:
                    truth.add((i1, x))
                    n_true += 1
        print(f"  {len(gt):,} S1 entities, {n_true:,} true pairs")

    results = {}
    union = {}   # source tag -> set of (i1, i2)

    for tag, s23 in (("S2", s2), ("S3", s3)):
        ids23 = s23["entity_id"].to_numpy()
        acc: set[tuple[int, int]] = set()
        truth_this = {(i, x) for (i, x) in truth if x.startswith(tag)}

        for name, cols in KEY_DEFS:
            with timer(f"{tag} block[{name}]"):
                pr = pairs_for_key(s1, s23, cols, a.max_block)
            got = {(int(i1), ids23[i2]) for i1, i2 in pr}
            rec = len(got & truth_this) / max(1, len(truth_this))
            results[(tag, name)] = (rec, len(pr))
            acc |= {(int(i1), int(i2)) for i1, i2 in pr}
            print(f"    recall {rec:7.2%}   pairs {len(pr):>12,}")
            del pr, got
            gc.collect()

        got_u = {(i1, ids23[i2]) for i1, i2 in acc}
        rec_u = len(got_u & truth_this) / max(1, len(truth_this))
        results[(tag, "UNION")] = (rec_u, len(acc))
        union[tag] = len(acc)
        print(f"  >>> {tag} UNION recall {rec_u:.2%}  pairs {len(acc):,}  "
              f"({len(acc)/len(s1):.1f} per S1)")
        del acc, got_u
        gc.collect()

    print("\n" + "=" * 68)
    print(f"{'key':<18}{'S2 recall':>12}{'S2 pairs':>14}{'S3 recall':>12}{'S3 pairs':>14}")
    print("-" * 68)
    for name, _ in KEY_DEFS + [("UNION", None)]:
        r2, p2 = results.get(("S2", name), (0, 0))
        r3, p3 = results.get(("S3", name), (0, 0))
        print(f"{name:<18}{r2:>11.2%}{p2:>14,}{r3:>11.2%}{p3:>14,}")
    print("=" * 68)
    tot = union.get("S2", 0) + union.get("S3", 0)
    print(f"total candidates {tot:,}  =  {tot/len(s1):.1f} per S1 entity")
    print("\nA union recall below ~0.97 means the ceiling is too low - add a "
          "blocker before touching the model.")


if __name__ == "__main__":
    main()
