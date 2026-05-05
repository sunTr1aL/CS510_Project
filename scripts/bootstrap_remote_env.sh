#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-relclip}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PROJECT_DIR="${PROJECT_DIR:-/shared/nas/data/m1/zixuans8/CS510_Project}"

if ! command -v conda >/dev/null 2>&1; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "conda is not available on PATH and no standard conda.sh was found" >&2
    exit 1
  fi
else
  eval "$(conda shell.bash hook)"
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1
python -m pip install -r "$PROJECT_DIR/requirements-experiment.txt"

python - <<'PY'
import importlib.util

for name in ["datasets", "huggingface_hub", "open_clip", "torch"]:
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"missing package after install: {name}")

import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device_count", torch.cuda.device_count())
PY
