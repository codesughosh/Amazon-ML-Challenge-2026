#!/usr/bin/env bash
# Run inference, then publish the results so they are retrievable from anywhere.
#
# The box sits on a private IP, so SSH only works from campus. This gzips the
# submission (~96 MB -> ~25 MB, comfortably under GitHub's 100 MB limit) and
# pushes it, so the output survives losing network access to the machine.
set -u
export PATH=~/miniconda3/bin:$PATH
cd ~/Amazon-ML-Challenge-2026

echo "=== run started $(date) ==="
python -u predict_test.py --countries France,US,India \
       --train-sample 30000 --chunk 60000 2>&1 | tee run.log
STATUS=${PIPESTATUS[0]}
echo "=== inference exit=$STATUS $(date) ==="

# Pad to the full entity list regardless of how far it got: a partial file is
# rejected outright, so this is what makes even an interrupted run uploadable.
python -u finalize_submission.py 2>&1 | tee -a run.log

cd ~/Amazon-ML-Challenge-2026
mkdir -p results
for f in matching_results_full.tsv candidate_pairs_full.tsv; do
  [ -f "output/$f" ] && gzip -c "output/$f" > "results/$f.gz"
done
ls -la results/ | awk '{printf "%-34s %7.1f MB\n", $NF, $5/1048576}'

git add -f results/*.gz run.log 2>/dev/null
git -c user.name="college-box" -c user.email="noreply@example.com" \
    commit -q -m "Inference results $(date -u +%Y-%m-%dT%H:%MZ) (exit=$STATUS)" 2>&1 | tail -2
git push origin main 2>&1 | tail -3
echo "=== published $(date) ==="
