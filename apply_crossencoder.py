"""Rescore the uncertain band with the cross-encoder and rebuild the output.

    python -u apply_crossencoder.py --countries France,US,India

Reads data/scored_{country}.parquet (pair probabilities saved by predict_test.py)
so blocking never has to run again - that is the expensive part, and it is
already done.

Only pairs whose LightGBM probability sits in the uncertain band are sent to the
cross-encoder. Confident pairs are left alone: rescoring a pair at p=0.99 cannot
change the decision, and scoring all 173M pairs would take ~26 hours versus ~3
for the band.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", "X:/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(ROOT))

DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
OUT = ROOT / "output"


def load_records(country: str) -> dict[str, str]:
    """entity_id -> 'name | address' for every record of one country."""
    rec = {}
    for i in (1, 2, 3):
        df = pd.read_csv(DATA / "test" / f"test_source{i}.tsv", sep="\t",
                         dtype=str, keep_default_na=False)
        df = df[df["country"] == country]
        for e, n, a in zip(df["entity_id"], df["business_name"],
                           df["business_address"]):
            rec[e] = f"{n} | {a}"
        del df
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", default="France,US,India")
    ap.add_argument("--model", default="models/crossencoder")
    ap.add_argument("--lo", type=float, default=0.05)
    ap.add_argument("--hi", type=float, default=0.95)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--weight", type=float, default=0.5,
                    help="blend weight on the cross-encoder score")
    ap.add_argument("--threshold", type=float, default=0.675)
    ap.add_argument("--max-k", type=int, default=12)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from tqdm.auto import tqdm

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ROOT / a.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        ROOT / a.model).to(dev).eval()
    print(f"device {dev}; band [{a.lo}, {a.hi}]")

    all_pred: dict[str, list[str]] = {}

    for country in a.countries.split(","):
        p = ROOT / "data" / f"scored_{country}.parquet"
        if not p.exists():
            print(f"  {country}: {p.name} missing, skipping")
            continue
        df = pd.read_parquet(p)
        band = (df["prob"] >= a.lo) & (df["prob"] <= a.hi)
        print(f"\n{country}: {len(df):,} pairs, {band.sum():,} in band "
              f"({band.mean():.1%})")

        rec = load_records(country)
        idx = np.where(band.to_numpy())[0]
        s1 = df["s1_id"].to_numpy()
        cd = df["cand_id"].to_numpy()
        ce = np.zeros(len(df), dtype=np.float32)

        with torch.no_grad():
            for lo in tqdm(range(0, len(idx), a.batch), desc=f"  {country} CE",
                           ascii=True, ncols=78):
                sl = idx[lo:lo + a.batch]
                t1 = [rec.get(x, "") for x in s1[sl]]
                t2 = [rec.get(x, "") for x in cd[sl]]
                enc = tok(t1, t2, padding=True, truncation=True,
                          max_length=a.max_len, return_tensors="pt").to(dev)
                with torch.autocast(dev, dtype=torch.float16):
                    out = torch.sigmoid(model(**enc).logits.squeeze(-1))
                ce[sl] = out.float().cpu().numpy()

        # Outside the band the LightGBM probability is kept as-is.
        final = df["prob"].to_numpy().astype(np.float32).copy()
        final[idx] = (1 - a.weight) * final[idx] + a.weight * ce[idx]

        # Threshold + cap, per S1 entity.
        order = np.lexsort((-final, s1))
        s_sorted, f_sorted, c_sorted = s1[order], final[order], cd[order]
        keep = f_sorted >= a.threshold
        _, starts, counts = np.unique(s_sorted, return_index=True, return_counts=True)
        for st, ct in zip(starts, counts):
            if ct > a.max_k:
                keep[st + a.max_k:st + ct] = False
        for s, c in zip(s_sorted[keep], c_sorted[keep]):
            all_pred.setdefault(s, []).append(c)
        del df, rec, ce, final
        print(f"  {country}: {sum(1 for _ in all_pred):,} entities with matches so far")

    ids = pd.read_csv(DATA / "test" / "test_source1.tsv", sep="\t", dtype=str,
                      keep_default_na=False, usecols=["entity_id"])["entity_id"]
    dst = OUT / "matching_results_ce.tsv"
    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s in ids:
            v = all_pred.get(s, [])
            seen, o = set(), []
            for x in v:
                if x not in seen:
                    seen.add(x); o.append(x)
            f.write(f"{s}\t{','.join(o)}\n")
    n = sum(1 for s in ids if all_pred.get(s))
    print(f"\nwrote {dst}  {len(ids):,} rows, {n:,} non-empty ({n/len(ids):.1%})")


if __name__ == "__main__":
    main()
