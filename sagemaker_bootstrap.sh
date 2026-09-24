#!/usr/bin/env bash
# Bootstrap a SageMaker notebook instance for the ML Challenge.
#
#   bash sagemaker_bootstrap.sh
#
# Run once per instance, from the repo root inside /home/ec2-user/SageMaker.
# Safe to re-run. Does NOT reinstall torch - the conda_pytorch_p310 kernel
# already ships a CUDA build, and replacing it is a reliable way to break CUDA.

set -euo pipefail

PERSIST=/home/ec2-user/SageMaker
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================================="
echo " ML Challenge 2026 - SageMaker bootstrap"
echo " repo: $REPO"
echo "=============================================================="

if [[ "$REPO" != "$PERSIST"* ]]; then
  echo
  echo "WARNING: this repo is NOT under $PERSIST"
  echo "Only that directory survives a notebook stop/start. Move the repo"
  echo "there before you do any real work, or you will lose it."
  echo
fi

# --- caches on the persistent EBS volume ---------------------------------
export HF_HOME="$PERSIST/hf_cache"
mkdir -p "$HF_HOME" "$REPO/data" "$REPO/submissions" "$REPO/models"

# Persist the env vars for future shells and notebook kernels.
PROFILE="$PERSIST/.ml_challenge_env.sh"
cat > "$PROFILE" <<EOF
export HF_HOME="$PERSIST/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO:\${PYTHONPATH:-}"
EOF
grep -q "ml_challenge_env" ~/.bashrc 2>/dev/null || \
  echo "source $PROFILE" >> ~/.bashrc
echo "env written to $PROFILE (sourced from ~/.bashrc)"

# --- python deps ----------------------------------------------------------
# Use the pytorch conda env's pip so packages land where the kernel looks.
PIP="${CONDA_PREFIX:-/home/ec2-user/anaconda3/envs/pytorch_p310}/bin/pip"
[[ -x "$PIP" ]] || PIP="$(command -v pip3 || command -v pip)"
echo "using pip: $PIP"

"$PIP" install --quiet --upgrade pip

# torch/torchvision deliberately excluded - already present with CUDA.
"$PIP" install --quiet \
  lightgbm xgboost catboost \
  transformers tokenizers sentencepiece accelerate datasets huggingface_hub \
  sentence-transformers timm \
  pyarrow scikit-learn scipy \
  pillow opencv-python-headless \
  matplotlib seaborn tqdm psutil requests

echo "python deps installed"

# --- verify ---------------------------------------------------------------
echo
echo "--- verification ---"
python - <<'PY'
import torch, platform
print("python      ", platform.python_version())
print("torch       ", torch.__version__)
print("cuda avail  ", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu         ", p.name, f"{p.total_memory/1024**3:.1f} GB")
else:
    print("!! No CUDA. You are on a CPU notebook instance, or the wrong kernel.")
    print("   Select the conda_pytorch_p310 kernel, or check the instance type.")
PY

echo
echo "--- disk ---"
df -h "$PERSIST" | tail -1 | awk '{print "  "$4" free of "$2" on "$6}'

echo
echo "=============================================================="
echo " Next:"
echo "   python smoke_test.py        # full environment check"
echo "   python download_models.py   # ~4 GB of weights -> \$HF_HOME"
echo "=============================================================="
