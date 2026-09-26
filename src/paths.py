"""Portable path resolution - no machine-specific paths anywhere else.

Every script imports DATA from here, so the repo runs unchanged on Windows,
Linux or a Kaggle notebook. Override with environment variables when the data
sits outside the repo:

    export ML_DATA=/scratch/student_resource/dataset
    export HF_HOME=~/hf_cache
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _find_data() -> Path:
    env = os.environ.get("ML_DATA")
    if env:
        return Path(env)
    # The organisers' zip extracts to a hashed folder name that differs per
    # download, so locate it rather than hardcoding it.
    for p in ROOT.glob("*student_resource*/student_resource/dataset"):
        return p
    for p in ROOT.glob("**/dataset/train/train_source1.tsv"):
        return p.parent.parent
    return ROOT / "dataset"


DATA = _find_data()
HF_HOME = os.environ.get("HF_HOME") or str(ROOT / "hf_cache")
os.environ.setdefault("HF_HOME", HF_HOME)

if __name__ == "__main__":
    print(f"ROOT    {ROOT}")
    print(f"DATA    {DATA}   exists={DATA.exists()}")
    print(f"HF_HOME {HF_HOME}")
