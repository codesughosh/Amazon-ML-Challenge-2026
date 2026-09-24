"""Submission writer with pre-flight checks.

Format bugs are the single most common way a strong model scores zero. Every
submission goes through `write_submission`, which refuses to write a file that
would be rejected.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SUB_DIR = Path(__file__).resolve().parent.parent / "submissions"


def write_submission(
    ids,
    preds,
    id_col: str,
    target_col: str,
    name: str,
    sample_path: str | Path | None = None,
    clip_min: float | None = None,
    clip_max: float | None = None,
    round_int: bool = False,
) -> Path:
    """Validate and write a submission CSV. Returns the path.

    Pass `sample_path` (the organisers' sample_submission.csv) whenever one
    exists - it is the only reliable source of truth for column names, row
    count and row order.
    """
    preds = np.asarray(preds, dtype=np.float64)
    ids = np.asarray(ids)

    if len(ids) != len(preds):
        raise ValueError(f"length mismatch: {len(ids)} ids vs {len(preds)} preds")

    n_nan = int(np.isnan(preds).sum())
    n_inf = int(np.isinf(preds).sum())
    if n_nan or n_inf:
        raise ValueError(f"predictions contain {n_nan} NaN and {n_inf} inf values")

    if clip_min is not None or clip_max is not None:
        before = preds.copy()
        preds = np.clip(preds, clip_min, clip_max)
        n_clipped = int((before != preds).sum())
        if n_clipped:
            print(f"clipped {n_clipped:,} predictions to [{clip_min}, {clip_max}]")

    if round_int:
        preds = np.rint(preds).astype(np.int64)

    sub = pd.DataFrame({id_col: ids, target_col: preds})

    if sample_path:
        sample = pd.read_csv(sample_path)
        if list(sample.columns) != list(sub.columns):
            raise ValueError(
                f"column mismatch\n  expected: {list(sample.columns)}\n"
                f"  got:      {list(sub.columns)}"
            )
        if len(sample) != len(sub):
            raise ValueError(f"row count mismatch: expected {len(sample):,}, got {len(sub):,}")
        # Reorder to the sample's exact id order. Many graders join on position.
        sub = sub.set_index(id_col).reindex(sample[id_col]).reset_index()
        if sub[target_col].isnull().any():
            n = int(sub[target_col].isnull().sum())
            raise ValueError(f"{n:,} ids in sample_submission have no prediction")
        print("validated against sample_submission")

    if sub[id_col].duplicated().any():
        raise ValueError(f"{int(sub[id_col].duplicated().sum()):,} duplicate ids")

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%m%d_%H%M")
    path = SUB_DIR / f"{stamp}_{name}.csv"
    sub.to_csv(path, index=False)

    print(f"\nwrote {path}")
    print(f"  rows: {len(sub):,}")
    print(f"  {target_col}: min={sub[target_col].min():.4f} "
          f"median={sub[target_col].median():.4f} max={sub[target_col].max():.4f}")
    print("\nfirst 3 rows:")
    print(sub.head(3).to_string(index=False))
    return path


def blend(paths, weights=None, id_col: str = None, target_col: str = None, name: str = "blend"):
    """Weighted average of existing submission files.

    Cheap and reliably worth 1-3% late in a competition. Only blend models whose
    OOF scores you have actually measured.
    """
    dfs = [pd.read_csv(p) for p in paths]
    id_col = id_col or dfs[0].columns[0]
    target_col = target_col or dfs[0].columns[1]

    base = dfs[0][[id_col]].copy()
    weights = np.asarray(weights if weights is not None else [1 / len(dfs)] * len(dfs), dtype=float)
    weights = weights / weights.sum()

    stacked = np.zeros(len(base))
    for df, w in zip(dfs, weights):
        aligned = df.set_index(id_col).reindex(base[id_col])[target_col].values
        if np.isnan(aligned).any():
            raise ValueError("id mismatch between submissions being blended")
        stacked += w * aligned

    return write_submission(base[id_col].values, stacked, id_col, target_col, name)
