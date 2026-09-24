"""End-to-end GPU smoke test. Run this TONIGHT, not at 1 AM tomorrow.

    python smoke_test.py

Proves, in about a minute: CUDA works, AMP works, a real transformer fine-tunes
on the 4060, and the metric/CV/submission plumbing all runs. If anything here
fails, it fails now while there is still time to fix it.
"""

from __future__ import annotations

import sys
import time

import numpy as np

print("=" * 60)
print("AMAZON ML CHALLENGE 2026 - ENVIRONMENT SMOKE TEST")
print("=" * 60)

failures = []


def check(label, fn):
    try:
        result = fn()
        print(f"  PASS  {label}" + (f"  ({result})" if result else ""))
        return True
    except Exception as e:
        print(f"  FAIL  {label}\n        {type(e).__name__}: {e}")
        failures.append(label)
        return False


# ---------------------------------------------------------------- 1. imports
print("\n[1/5] Core imports")
check("numpy / pandas", lambda: f"numpy {np.__version__}")
check("scikit-learn", lambda: __import__("sklearn").__version__)
check("lightgbm", lambda: __import__("lightgbm").__version__)
check("xgboost", lambda: __import__("xgboost").__version__)
check("transformers", lambda: __import__("transformers").__version__)
check("pyarrow (parquet)", lambda: __import__("pyarrow").__version__)


# ------------------------------------------------------------------ 2. cuda
print("\n[2/5] CUDA")
import torch

print(f"  torch {torch.__version__}")
if not torch.cuda.is_available():
    print("  FAIL  torch.cuda.is_available() is False")
    print("        You are on a CPU-only build. Reinstall with:")
    print("        pip install torch torchvision "
          "--index-url https://download.pytorch.org/whl/cu128")
    failures.append("cuda")
else:
    props = torch.cuda.get_device_properties(0)
    print(f"  PASS  {props.name}")
    print(f"        VRAM {props.total_memory / 1024**3:.1f} GB | "
          f"CUDA {torch.version.cuda} | capability {props.major}.{props.minor}")

    def matmul_bench():
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            a @ a
        torch.cuda.synchronize()
        tflops = 20 * 2 * 4096**3 / (time.time() - t0) / 1e12
        del a
        torch.cuda.empty_cache()
        return f"{tflops:.1f} TFLOPS fp16"

    check("fp16 matmul throughput", matmul_bench)
    check("AMP autocast", lambda: bool(
        torch.autocast("cuda", dtype=torch.float16).__enter__() is None or True))


# ------------------------------------------------- 3. real transformer step
print("\n[3/5] Transformer fine-tune step on GPU")
if torch.cuda.is_available():
    def finetune_step():
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        name = "distilbert-base-uncased"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=1
        ).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
        scaler = torch.amp.GradScaler("cuda")

        texts = ["wireless bluetooth headphones over ear 40h battery"] * 16
        batch = tok(texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=128)
        batch = {k: v.cuda() for k, v in batch.items()}
        labels = torch.randn(16, 1).cuda()

        t0 = time.time()
        for _ in range(3):
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(**batch).logits
                loss = torch.nn.functional.mse_loss(out, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        dt = (time.time() - t0) / 3
        del model, opt
        torch.cuda.empty_cache()
        return f"{dt * 1000:.0f} ms/step @ bs16, peak VRAM {peak:.2f} GB"

    check("distilbert fwd+bwd with AMP", finetune_step)
else:
    print("  SKIP  (no CUDA)")


# ------------------------------------------------------- 4. project plumbing
print("\n[4/5] Project plumbing")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

def plumbing():
    import pandas as pd

    from src.cv import make_folds, run_cv
    from src.metrics import smape
    from src.submit import write_submission
    from src.utils import seed_everything

    seed_everything(42)
    assert abs(smape([100.0], [200.0]) - 66.6667) < 1e-3

    df = pd.DataFrame({
        "id": range(500),
        "x1": np.random.randn(500),
        "x2": np.random.randn(500),
        "y": np.random.rand(500) * 100 + 10,
    })
    folds = make_folds(df, n_splits=3, strategy="kfold")

    def fit_predict(tr, va, te, fold):
        import lightgbm as lgb

        m = lgb.LGBMRegressor(n_estimators=30, verbose=-1)
        m.fit(tr[["x1", "x2"]], tr["y"])
        return m.predict(va[["x1", "x2"]]), None

    _, _, _, oof = run_cv(df, folds, fit_predict, "y", "smape", verbose=False)
    write_submission(df["id"].values, np.full(500, 50.0), "id", "y",
                     "smoketest", clip_min=0)
    return f"OOF smape {oof:.2f} on random data (~45-60 expected)"

check("metrics + CV + submission round-trip", plumbing)


# ------------------------------------------------------------- 5. disk / ram
print("\n[5/5] Resources")
import shutil

free_gb = shutil.disk_usage("X:\\").free / 1024**3
print(f"  X: free {free_gb:.1f} GB" + ("" if free_gb > 25 else "   <-- TIGHT"))
try:
    import psutil

    print(f"  RAM available {psutil.virtual_memory().available / 1024**3:.1f} GB "
          f"of {psutil.virtual_memory().total / 1024**3:.1f} GB")
except ImportError:
    pass


print("\n" + "=" * 60)
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {', '.join(failures)}")
    sys.exit(1)
print("ALL CHECKS PASSED - you are ready for kickoff.")
print("=" * 60)
