"""Hackathon utilities: seeding, timing, memory control, fast IO.

With 15.6 GB of RAM, `reduce_mem` and parquet caching are not optional niceties
on a large catalog dataset - they are what keeps the machine from swapping.
"""

from __future__ import annotations

import gc
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Leave cudnn.benchmark on: the speed is worth more than bitwise
        # reproducibility over a 72-hour window.
        torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


@contextmanager
def timer(name: str):
    t0 = time.time()
    print(f"[{name}] start")
    yield
    print(f"[{name}] done in {time.time() - t0:.1f}s")


def reduce_mem(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns in place. Typically cuts memory 50-70%.

    Floats go to float32, not float16 - float16 silently destroys precision on
    price-like targets and will cost you leaderboard points.
    """
    start = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        t = df[col].dtype
        if pd.api.types.is_integer_dtype(t):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(t):
            df[col] = df[col].astype(np.float32)
    end = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"reduce_mem: {start:.1f} MB -> {end:.1f} MB ({100 * (start - end) / start:.0f}% saved)")
    return df


def load_cached(csv_path, parquet_path=None, **read_kwargs) -> pd.DataFrame:
    """Read a CSV once, then serve every later read from parquet.

    Re-parsing a multi-GB CSV on every notebook restart is the quietest way to
    lose an hour a day.
    """
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path) if parquet_path else csv_path.with_suffix(".parquet")
    if parquet_path.exists():
        print(f"loading cached parquet: {parquet_path.name}")
        return pd.read_parquet(parquet_path)
    print(f"parsing csv: {csv_path.name}")
    df = pd.read_csv(csv_path, **read_kwargs)
    df = reduce_mem(df)
    df.to_parquet(parquet_path, index=False)
    print(f"cached -> {parquet_path.name}")
    return df


def free() -> None:
    """Drop Python and CUDA garbage. Call between model fits."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def gpu_mem() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "cuda unavailable"
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"GPU {alloc:.2f}/{reserved:.2f}/{total:.1f} GB (alloc/reserved/total)"
    except ImportError:
        return "torch not installed"


def describe_df(df: pd.DataFrame, name: str = "df") -> None:
    """The first thing to run on the dataset. Answers most Day-1 questions."""
    print(f"=== {name}: {df.shape[0]:,} rows x {df.shape[1]} cols ===")
    print(f"memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB\n")
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "nulls": df.isnull().sum(),
        "null_%": (100 * df.isnull().mean()).round(2),
        "nunique": df.nunique(),
    })
    print(info.to_string())
