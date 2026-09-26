# Resume on the college GPU box

State as of 26 Sep, ~08:30. **France and US are done and persisted. Only India remains.**

## What is already computed (do not redo)

| File | Contents | Why it matters |
|---|---|---|
| `data/scored_France.parquet` | 25.9M pairs + LightGBM probability | blocking for France never needs to run again |
| `data/scored_US.parquet` | ~66M pairs + probability | same for US |
| `output/matching_results.tsv` | France + US predictions | partial, **must be padded before upload** |
| `models/crossencoder/` | fine-tuned multilingual-e5-small | val AUC 0.9999 on held-out entities |
| `data/ce_pairs.parquet` | 411k labelled training pairs | for retraining / tuning the cross-encoder |

Blocking is the expensive stage (~7.5 h per large country). Both completed
shards have it banked. **Nothing from the overnight run is lost.**

## Setup on the box

```bash
git clone https://github.com/codesughosh/Amazon-ML-Challenge-2026.git
cd Amazon-ML-Challenge-2026
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt Unidecode rapidfuzz
```

Then copy across from the laptop (they are gitignored, being large):

- the dataset folder `6ab10eb3b23ba_student_resource/`
- `data/scored_France.parquet`, `data/scored_US.parquet`
- `models/crossencoder/` (only if running the cross-encoder pass)

Paths: `predict_test.py` and the other scripts resolve `DATA` relative to the
repo root, so no edits are needed as long as the dataset folder sits beside
them. `train_crossencoder.py` and `apply_crossencoder.py` default `HF_HOME` to
`X:/hf_cache` - override it on Linux:

```bash
export HF_HOME=~/hf_cache
```

## Step 1 - run the India shard

```bash
python -u predict_test.py --countries India --train-sample 30000 --chunk 20000
```

Took ~7.5 h for US (663k entities) on a 14-core laptop; India is 810k, so
expect ~9 h there and proportionally less on a bigger box. The stage is
CPU- and RAM-bound, **not** GPU-bound - what helps is cores and memory, not the
P5000s. Raise `--chunk` if the box has plenty of RAM; 20000 was sized for 16 GB.

It retrains the LightGBM model first (~20 min). That is deterministic
(`seed_everything`), so it reproduces the same model the other two shards used.

Output: `data/scored_India.parquet` plus an updated `output/matching_results.tsv`.

## Step 2 - build the uploadable file

`predict_test.py` only writes rows for entities it has processed. A partial file
is **rejected outright** - every test entity must appear. So always finish with:

```bash
python -u finalize_submission.py
```

which writes `output/matching_results_full.tsv` with all 1,732,544 rows, then
validate before uploading:

```bash
cd 6ab10eb3b23ba_student_resource/student_resource
python utils/validate_submission.py \
  --matching ../../output/matching_results_full.tsv \
  --candidate ../../output/candidate_pairs_full.tsv \
  --test-dir dataset/test
```

Expect `PASS` and roughly **0.93**.

## Step 3 - cross-encoder (only after step 2 is banked)

**Not yet validated end-to-end.** We know it separates pairs at AUC 0.9999 on
held-out training entities, but we have never measured whether that improves
macro F_0.5, and the blend weight is a guess. Validate before trusting it:

```bash
python -u run_floor.py --country India --sample-s1 25000 --max-per-s1 50 \
       --max-block 1500 --n-rare 8 --solo-df 4000
```

Baseline to beat: **0.9349**. Only if a cross-encoder-augmented run clears that
by a real margin is it worth applying:

```bash
python -u apply_crossencoder.py --countries France,US,India --weight 0.5
```

This reads the saved `scored_*.parquet`, so it does **not** re-run blocking
(~3 h instead of ~26). Sweep `--weight` and `--threshold` on validation first.

Also untested: the cross-encoder was trained on India only. **France is unseen
in training data entirely** and is 15% of the test set - check it holds there
before using it for the final submission.

## Where the score stands

```
0.056    all-empty baseline (submitted, confirms format)
0.9002   first working pipeline
0.9219   + similarity re-ranking of candidates
0.9349   + new blocking key families, + partial_ratio ranker   <- current
0.9869   leaderboard leader
```

Blocking recall is 95.06%; at that recall, perfect precision would score 0.9896.
The model currently converts 86.5% of the true pairs blocking hands it, so the
remaining gap is model quality, not candidate generation. That is what the
cross-encoder is meant to attack.

Things measured as worth nothing (do not spend time re-trying): the many-to-one
assignment constraint, per-entity expected-F_0.5 selection, a singleton
classifier, reverse-direction features, and 2.8x more training data.
