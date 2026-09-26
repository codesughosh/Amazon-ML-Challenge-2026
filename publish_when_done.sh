#!/usr/bin/env bash
# Wait for the running inference to exit, then publish results to GitHub.
# Separate from the run itself so an in-flight job does not need restarting.
set -u
export PATH=~/miniconda3/bin:$PATH
cd ~/Amazon-ML-Challenge-2026

echo "waiting for predict_test.py to finish... ($(date))"
while pgrep -f "python -u predict_test.py" >/dev/null; do sleep 60; done
echo "inference finished $(date)"

python -u finalize_submission.py 2>&1 | tee -a run.log

mkdir -p results
for f in matching_results_full.tsv candidate_pairs_full.tsv; do
  [ -f "output/$f" ] && gzip -c "output/$f" > "results/$f.gz"
done
ls -la results/ 2>/dev/null | awk "{printf \"%-34s %7.1f MB\n\", \$NF, \$5/1048576}"

git add -f results/*.gz run.log 2>/dev/null
git -c user.name="college-box" -c user.email="noreply@example.com" \
    commit -q -m "Inference results $(date -u +%Y-%m-%dT%H:%MZ)" 2>&1 | tail -2
git push origin main 2>&1 | tail -3
echo "=== published $(date) ==="
