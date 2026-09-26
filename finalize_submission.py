"""Pad a partial submission up to the full test entity list.

    python -u finalize_submission.py

predict_test.py writes output after each country, but only for the entities it
has processed so far. The rules require EVERY test Source-1 entity to appear or
the submission is rejected, so a partial file is not uploadable as-is.

This reads whatever predictions exist, adds an empty row for every unprocessed
entity, and writes the result in the required order. Safe to run at any time,
including while the main job is still going - it only reads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
OUT = ROOT / "output"


def pad(fname: str, col: str) -> None:
    src = OUT / fname
    ids = pd.read_csv(DATA / "test" / "test_source1.tsv", sep="\t", dtype=str,
                      keep_default_na=False, usecols=["entity_id"])["entity_id"]

    have = {}
    if src.exists():
        df = pd.read_csv(src, sep="\t", dtype=str, keep_default_na=False)
        if len(df.columns) >= 2:
            have = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        print(f"  {fname}: {len(have):,} predicted rows found")
    else:
        print(f"  {fname}: not present, writing all-empty")

    dst = OUT / fname.replace(".tsv", "_full.tsv")
    n_pred = 0
    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s in ids:
            v = have.get(s, "")
            if v:
                n_pred += 1
            f.write(f"{s}\t{v}\n")
    print(f"  -> {dst.name}: {len(ids):,} rows, {n_pred:,} non-empty "
          f"({n_pred/len(ids):.1%})")


def main():
    OUT.mkdir(exist_ok=True)
    pad("matching_results.tsv", "matched_entity_ids")
    pad("candidate_pairs.tsv", "candidate_entity_ids")
    print("\nUpload the *_full.tsv files. Validate first:")
    print("  cd 6ab10eb3b23ba_student_resource/student_resource")
    print("  python utils/validate_submission.py \\")
    print("    --matching ../../output/matching_results_full.tsv \\")
    print("    --candidate ../../output/candidate_pairs_full.tsv \\")
    print("    --test-dir dataset/test")


if __name__ == "__main__":
    main()
