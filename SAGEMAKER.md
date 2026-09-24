# SageMaker setup — primary environment

Laptop (RTX 4060) is the fallback if credits run out. This is the main plan.

---

## The two settings that will bite you

**1. The default EBS volume is 5 GB. That is not enough and you set it at creation.**

You need room for: 150k product images (~1 GB downscaled, ~8 GB at full size),
train + test CSVs, parquet caches, model weights (~4 GB), and checkpoints.
**Ask for 100 GB.** It costs about $0.14/GB-month — roughly $1.40 for the whole
challenge, which is noise against your $200. Running out of disk at hour 30 is
not noise.

You *can* resize later, but only while the instance is **stopped**, and the
restart costs you momentum. Get it right the first time.

**2. GPU notebook-instance quota is a separate quota from training-job quota.**

Correcting what I told you earlier: I said to request quota for `ml.g5.xlarge`
**training jobs**. Under this plan you want:

> Service Quotas → Amazon SageMaker → `ml.g5.xlarge for notebook instance usage`

They are different limits and new accounts often have **0** for GPU notebook
instances. Request it **now** — approval is not instant, and the console will
happily let you try to create an instance that then fails.

Request both `ml.g5.xlarge` and `ml.g4dn.xlarge` so you have a fallback if one
is denied or capacity-constrained in the region.

---

## What to create

Region **us-east-1** (best service coverage, and what the AWS prep session used).

| Instance | GPU | VRAM | ~$/hr | 72h continuous |
|---|---|---|---|---|
| `ml.t3.medium` | — | — | ~$0.05 | free tier, 250 h |
| `ml.g4dn.xlarge` | T4 | 16 GB | ~$0.74 | ~$53 |
| **`ml.g5.xlarge`** | **A10G** | **24 GB** | **~$1.41** | **~$101** |
| `ml.g5.2xlarge` | A10G | 24 GB | ~$1.69 | ~$122 |

*us-east-1 approximate — check current pricing in the console.*

**Recommendation: `ml.g5.xlarge`.** 24 GB of VRAM is 3x your laptop, which means
DeBERTa-v3-large becomes reachable and you can run batch 32–64 instead of 16.
$101 for 72 hours fits inside $200 with room for the S3 and EBS line items.

**Settings at creation:**
- Name: `ml-challenge-2026`
- Instance type: `ml.g5.xlarge`
- **Volume size: 100 GB** ← the one people miss
- IAM role: Create a new role, defaults are fine
- Platform: latest Amazon Linux 2 / JupyterLab 4

**Budget math with a team.** Credits are **per participant**, so a team of 4 has
4 × $200 = $800 across four accounts. You cannot pool them, but you can
parallelise: each member runs a different experiment branch in their own
account and you compare OOF scores in a shared sheet. That is 4x the experiment
throughput for free, and it is the single biggest structural advantage available
to you. Decide who runs what before kickoff.

---

## Billing discipline

- A notebook instance bills for **wall-clock time while `InService`**, whether or
  not the GPU is doing anything. A forgotten instance overnight is ~$34.
- **Stop it when you sleep.** Stopping preserves `/home/ec2-user/SageMaker`.
  *Stop, never delete* — deleting wipes the volume.
- **Endpoints are not needed.** You submit a CSV, not an API. If you ever deploy
  one, delete it — it bills ~$0.12/hr even idle.
- Set a **billing alarm at $50 and $150** before you start.

---

## Bootstrap the instance

Only `/home/ec2-user/SageMaker` survives a stop/start. Everything else is
ephemeral, so the project, the venv-equivalent, and the HF cache all live there.

Open a terminal in JupyterLab and run:

```bash
cd /home/ec2-user/SageMaker
git clone <your-repo-url> ml-challenge-2026   # or upload this folder as a zip
cd ml-challenge-2026
bash sagemaker_bootstrap.sh
```

Then in a notebook, pick the **`conda_pytorch_p310`** kernel — it already has a
CUDA build of PyTorch, so you do not reinstall torch.

Verify before you trust it:

```python
import torch; print(torch.__version__, torch.cuda.is_available(),
                    torch.cuda.get_device_name(0))
```

---

## Day-1 order of operations on SageMaker

1. Start the instance (~3 min to `InService`).
2. Upload / download the dataset to `/home/ec2-user/SageMaker/ml-challenge-2026/data`.
3. **Kick off image download immediately, in a background terminal** — from inside
   AWS this is far faster than from home broadband:
   ```bash
   nohup python -c "
   import pandas as pd; from src.images import download_all
   df = pd.read_csv('data/train.csv')
   download_all(df['image_link'], 'data/train_images', max_workers=64, max_side=256)
   " > images.log 2>&1 &
   ```
   Use `max_workers=64` here — the EC2 network handles it. Watch `images.log`.
4. Do EDA and ship the TF-IDF + LightGBM floor while images download.
5. Only then start the transformer.

---

## S3

Your data does not *have* to go through S3 — local training inside the notebook
reads straight off the EBS volume, which is what the AWS prep session recommends
for this challenge. Use S3 for:

- **Backing up checkpoints and submissions.** If the instance dies, the EBS volume
  usually survives, but a `aws s3 sync submissions/ s3://<bucket>/submissions/`
  costs nothing and removes the risk entirely. Do this after every good model.
- Sharing artefacts between teammates' accounts.

```python
import sagemaker
bucket = sagemaker.Session().default_bucket()   # auto-created, no setup
```

---

## If credits run out → laptop fallback

The repo runs identically on the RTX 4060. `download_models.py` and
`smoke_test.py` detect the platform and pick the right cache path, so nothing
needs editing. Differences to plan for:

- 8 GB VRAM instead of 24 — halve the batch size, enable gradient checkpointing,
  and drop from DeBERTa-large to DeBERTa-base.
- 15.6 GB RAM — lean on `reduce_mem` and parquet caching.
- Keep the laptop **plugged in**; the GPU is capped at 95 W and throttles.
