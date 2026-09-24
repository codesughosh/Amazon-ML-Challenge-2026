# Amazon ML Challenge 2026 — 72-Hour Runbook

**Window:** 25 Sep 00:00 IST → 27 Sep 23:59 IST
**Deliverables:** leaderboard predictions + zipped code/notebook + **1–2 page approach doc**
**Stakes:** Top 50 → Amazon Applied Scientist Intern PPI. Top 10 → Grand Finale, 7 Oct.

---

## The three rules

1. **Get a scoring submission on the board within 4 hours.** However dumb. A mean
   baseline proves the format is right and gives you a floor. Teams that build the
   perfect architecture for two days and hit a format bug on Day 3 lose.
2. **Optimise the competition metric, nothing else.** Implement it in
   `src/metrics.py` first, verify it against any worked example in the problem
   statement, then never look at another number.
3. **Trust local CV, not the public leaderboard.** Final rank comes from a private
   leaderboard on the *complete* test set. When CV and the board disagree in
   direction, believe CV.

---

## Hour 0–1: triage

Before writing a single model, answer these in a shared doc:

- [ ] What is the **exact metric**? Is there a worked example to check against?
- [ ] What is the **submission format**? Column names, row count, row order.
      Is there a `sample_submission.csv`?
- [ ] **How many submissions per day?** This sets your whole experiment budget.
- [ ] Are **external data / pretrained weights / LLM APIs** allowed? 2025 banned
      external price lookups. Read the rules clause before you touch an API.
- [ ] What **modalities** ship? Text only, text + images, tabular? How big on disk?
- [ ] Is the test set a **random split** or held-out **groups** (brands, sellers,
      categories)? This decides `strategy=` in `make_folds`.
- [ ] Target distribution — skewed? Long-tailed? Zeros? Log-transform needed?

```python
from src.utils import describe_df, load_cached
train = load_cached("data/train.csv")
describe_df(train, "train")
```

---

## Hour 1–4: the floor

Ship in this order. Do not skip ahead.

1. **Constant baseline.** Predict the train median. Submit it. Confirms format.
2. **TF-IDF + LightGBM** on the text field. Char n-grams (3–5) plus word n-grams
   catch a surprising amount of catalog signal. Submit it.
3. Record both OOF scores in `experiments.csv` via `Tracker`. That is your floor.

If the metric is SMAPE or any relative error, **train on `log1p(target)` and
invert with `expm1`**. This alone is usually worth several points — relative
error metrics punish absolute-scale losses badly.

---

## Hour 4–36: the real model

Base-size transformers only. 8 GB VRAM comfortably fits DeBERTa-v3-base at
batch 16 with AMP; it will not fit anything much larger without quantization.

**Text branch:** `microsoft/deberta-v3-base`, `max_length` 256–384, AMP fp16,
gradient checkpointing if VRAM is tight, 2–3 epochs, lr 2e-5, cosine schedule.

**Image branch (only if images ship):** `openai/clip-vit-base-patch32` or
`facebook/convnextv2-tiny-1k-224`. Extract embeddings once, cache to disk, then
train the head on cached features — do **not** re-encode images every epoch.

**Fusion:** concatenate the two pooled vectors into a small MLP head. The 2025
top solutions used exactly this (CLIP+DistilBERT → 40.78 SMAPE; DeBERTa+ConvNeXT
with a gated fusion head). Winning score was 39.7.

**Always keep the GBDT alive.** Feed transformer embeddings plus handcrafted
features into LightGBM. It blends well with the neural model and is your
insurance if training goes sideways.

---

## Hour 36–60: squeeze

In descending order of value-per-hour:

1. **Blend.** Average your best 3–4 OOF-verified models (`src.submit.blend`).
   Reliably worth 1–3%. Only blend models whose OOF you have actually measured.
2. **Feature engineering on the text.** Lengths, digit/unit extraction, brand
   tokens, pack quantity, category keywords. Per the AWS prep session: "feature
   engineering often matters more than your choice of algorithm."
3. **Seed averaging.** Same architecture, 3 seeds, average. Nearly free variance
   reduction.
4. **Target post-processing.** Clip to the observed train range. Check whether
   the metric rewards a small systematic bias (SMAPE rewards under-prediction —
   test a multiplier between 0.92 and 1.0 on OOF).

Hyperparameter tuning is near the *bottom* of this list. It is the lowest
return on a 72-hour clock.

---

## Hour 60–72: land it

- [ ] **Write the approach document.** 1–2 pages. It is explicitly part of top-10
      selection alongside leaderboard rank. Structure: problem framing → data
      insights → architecture (a diagram helps) → what worked, what didn't (with
      OOF numbers) → results. **Do not write this at 11:30 PM.**
- [ ] **Clean the zip.** Code must run top-to-bottom. Strip absolute paths, dead
      cells, and API keys. Add a one-line "how to run".
- [ ] **Submit your best OOF model, not your best public-LB model.**
- [ ] Submit with **hours to spare.** Unstop gets slow at the deadline with 89,000
      registrants on it.

---

## Compute

| Resource | Use it for |
|---|---|
| **RTX 4060 (8 GB)** | Primary. Base transformers, GBDT, all iteration. |
| **AWS $200 credits** | `ml.g5.xlarge` (A10G 24 GB) when you need bigger batches or parallel runs. ~$1.4/hr. Top 500 get +$100 at the 48h mark. |
| **Kaggle** | Free fallback, ~30 hrs/week T4×2. Log in tonight. |

SageMaker free tier is **CPU-only** (`ml.t3.medium`). Any GPU work comes out of
credits. Request a GPU quota increase *before* you need it — approval is not
instant. Set a billing alert. Stop notebook instances when idle; delete endpoints
(they bill ~$0.12/hr even idle).

**Laptop discipline:** stay plugged in, Windows power mode on Performance. The
4060 is capped at 95 W and will thermal-throttle across a 72-hour run.

**RAM is your tightest resource (15.6 GB).** Use `load_cached` for parquet
caching and `reduce_mem` on every dataframe. Do not hold train, test, and
intermediate features in memory at once.

---

## Layout

```
X:\amazon-ml-2026\
├── .venv\                 Python 3.13 + CUDA 12.8 torch
├── data\                  raw + cached parquet (gitignored)
├── src\
│   ├── metrics.py         SMAPE / F1 / RMSLE + registry   <- edit this first
│   ├── utils.py           seeding, timing, reduce_mem, load_cached
│   ├── cv.py              make_folds, run_cv, Tracker
│   └── submit.py          write_submission (validates), blend
├── notebooks\             EDA and experiments
├── submissions\           timestamped, never overwritten
├── models\                checkpoints
├── smoke_test.py          run first, proves GPU + plumbing work
└── download_models.py     pre-cached weights in X:\hf_cache
```

## Team split (2–4 people)

| Role | Owns |
|---|---|
| **Infra** | submission pipeline, CV harness, experiment tracking, the zip |
| **Text** | tokenization, text encoder, text features |
| **Vision** | image loading, embedding extraction + caching (only if images ship) |
| **Analysis** | EDA, metric verification, error analysis, the approach doc |

One person owns the submission pipeline end-to-end. Diffusing that responsibility
is how format bugs survive to Day 3.

---

## Quick start

```powershell
cd X:\amazon-ml-2026
.\.venv\Scripts\Activate.ps1
python smoke_test.py
```
