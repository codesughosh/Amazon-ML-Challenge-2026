"""Fine-tune a multilingual cross-encoder on candidate pairs.

    python -u train_crossencoder.py --epochs 2

Why this and not more string features: the pairs we still lose look like

    Blue Services Pvt Ltd   vs   ব্লু সার্ভিসেস প্রাইভেট লিমিটেড
    Ss Exports Limited      vs   ಎಸ್ಎಸ್ ಎಕ್ಸ್‌ಪೋರ್ಟ್ಸ್ ಲಿಮಿಟೆಡ್

As strings these share nothing, and romanisation only half-recovers them
(`blu saarbhises praaibhett limittedd`). A multilingual encoder aligns the
scripts in embedding space, which is a capability no character metric has.

Backbone: multilingual-e5-small (MIT, 118M params) - comfortably inside the
competition's MIT/Apache and 8B-parameter limits. Runs on the RTX 4060 at
batch 64 with AMP.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", "X:/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/ce_pairs.parquet")
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--out", default="models/crossencoder")
    a = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    if dev == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"  {p.name}  {p.total_memory/1024**3:.1f} GB")

    df = pd.read_parquet(ROOT / a.pairs)
    print(f"pairs: {len(df):,}  ({df.label.mean():.1%} positive)")

    # Split by S1 ENTITY, never by pair - the same entity's candidates must not
    # straddle the split or validation is optimistic.
    ents = df["s1_id"].unique()
    rng = np.random.default_rng(42)
    val_ents = set(rng.choice(ents, int(len(ents) * a.val_frac), replace=False))
    is_val = df["s1_id"].isin(val_ents).to_numpy()
    tr, va = df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)
    print(f"  train {len(tr):,} pairs / {len(ents)-len(val_ents):,} entities")
    print(f"  val   {len(va):,} pairs / {len(val_ents):,} entities")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=1).to(dev)

    class DS(Dataset):
        def __init__(self, d):
            self.t1 = d["text1"].tolist()
            self.t2 = d["text2"].tolist()
            self.y = d["label"].to_numpy(dtype=np.float32)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return self.t1[i], self.t2[i], self.y[i]

    def collate(batch):
        t1, t2, y = zip(*batch)
        enc = tok(list(t1), list(t2), padding=True, truncation=True,
                  max_length=a.max_len, return_tensors="pt")
        return enc, torch.tensor(y)

    dl_tr = DataLoader(DS(tr), batch_size=a.batch, shuffle=True,
                       collate_fn=collate, num_workers=0, drop_last=True)
    dl_va = DataLoader(DS(va), batch_size=a.batch * 2, shuffle=False,
                       collate_fn=collate, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    steps = len(dl_tr) * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(dev)
    lossf = torch.nn.BCEWithLogitsLoss()

    from tqdm.auto import tqdm
    for ep in range(a.epochs):
        model.train()
        run, t0 = 0.0, time.time()
        bar = tqdm(dl_tr, desc=f"  epoch {ep+1}/{a.epochs}", ascii=True, ncols=78)
        for i, (enc, y) in enumerate(bar):
            enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
            y = y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev, dtype=torch.float16):
                out = model(**enc).logits.squeeze(-1)
                loss = lossf(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            if i % 50 == 0:
                bar.set_postfix(loss=f"{run/(i+1):.4f}")
        print(f"  epoch {ep+1} loss {run/len(dl_tr):.4f}  ({time.time()-t0:.0f}s)")

        model.eval()
        preds = []
        with torch.no_grad():
            for enc, _ in tqdm(dl_va, desc="  validating", ascii=True, ncols=78):
                enc = {k: v.to(dev) for k, v in enc.items()}
                with torch.autocast(dev, dtype=torch.float16):
                    preds.append(torch.sigmoid(
                        model(**enc).logits.squeeze(-1)).float().cpu().numpy())
        p = np.concatenate(preds)
        from sklearn.metrics import average_precision_score, roc_auc_score
        print(f"  val AUC {roc_auc_score(va['label'], p):.4f}   "
              f"AP {average_precision_score(va['label'], p):.4f}")

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    np.save(out / "val_preds.npy", p)
    va[["s1_id", "label"]].to_parquet(out / "val_meta.parquet", index=False)
    print(f"\nsaved -> {out}")
    print("Compare the AP above against the LightGBM pair model's AP. If it is "
          "materially higher, the cross-encoder score is worth adding as a "
          "feature; if not, drop it and keep the string pipeline.")


if __name__ == "__main__":
    main()
