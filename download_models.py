"""Pre-download model weights so kickoff is not spent waiting on HuggingFace.

    python download_models.py

About 4 GB total. Run it tonight on good wifi. Weights land in X:\\hf_cache,
which keeps them off the system drive.
"""

import os
from pathlib import Path

CACHE = Path("X:/hf_cache")
CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE)

from huggingface_hub import snapshot_download  # noqa: E402

# Chosen to cover the plausible problem space with base-size models that fit
# in 8 GB of VRAM. The 2025 top solutions were built from exactly this shelf.
MODELS = [
    # --- text encoders -----------------------------------------------------
    ("microsoft/deberta-v3-base",      "strongest text encoder that fits; 2025 top-solution backbone"),
    ("distilbert-base-uncased",        "fast baseline, 2x faster than DeBERTa"),
    ("sentence-transformers/all-MiniLM-L6-v2", "cheap sentence embeddings for kNN/features"),
    # --- vision / multimodal ----------------------------------------------
    ("openai/clip-vit-base-patch32",   "image+text in one space; half of the 2025 winning fusion"),
    ("facebook/convnextv2-tiny-1k-224", "image encoder, pairs well with DeBERTa"),
    # --- extraction fallback ----------------------------------------------
    ("google/flan-t5-base",            "seq2seq for entity-extraction style tasks"),
]


def main():
    ok, failed = [], []
    for repo, why in MODELS:
        print(f"\n--- {repo}\n    {why}")
        try:
            snapshot_download(
                repo_id=repo,
                cache_dir=CACHE,
                ignore_patterns=["*.msgpack", "*.h5", "*.onnx", "*.tflite"],
            )
            ok.append(repo)
            print("    done")
        except Exception as e:
            failed.append((repo, str(e)[:120]))
            print(f"    FAILED: {e}")

    print("\n" + "=" * 60)
    print(f"downloaded {len(ok)}/{len(MODELS)}")
    for repo, err in failed:
        print(f"  FAILED {repo}: {err}")
    size = sum(f.stat().st_size for f in CACHE.rglob("*") if f.is_file()) / 1024**3
    print(f"cache size: {size:.2f} GB at {CACHE}")
    print("\nSet HF_HOME=X:\\hf_cache in any new shell to use this cache.")


if __name__ == "__main__":
    main()
