"""Emit a valid all-empty submission, to prove the format before spending hours.

An all-empty file scores ~0.056 (the singleton rate), but the point is not the
score: it confirms the column names, row count, row order and ID universe are
accepted, so a real run is not wasted on a format rejection.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
from src.paths import DATA  # portable: see src/paths.py
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

s1 = pd.read_csv(DATA / "test" / "test_source1.tsv", sep="\t", dtype=str,
                 keep_default_na=False, usecols=["entity_id"])
print(f"test S1 entities: {len(s1):,}")

for fname, col in (("matching_results.tsv", "matched_entity_ids"),
                   ("candidate_pairs.tsv", "candidate_entity_ids")):
    p = OUT / fname
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s in s1["entity_id"]:
            f.write(f"{s}\t\n")
    print(f"wrote {p}  ({p.stat().st_size/1024**2:.1f} MB)")
