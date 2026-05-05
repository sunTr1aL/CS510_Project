#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/shared/nas/data/m1/zixuans8/CS510_Project}"
BENCH_ROOT="${BENCH_ROOT:-/shared/nas/data/m1/zixuans8/relbottleneck/benchmarks/public_hf}"
ENV_NAME="${ENV_NAME:-relclip}"
PYTHON_BIN="${PYTHON_BIN:-/shared/nas/data/m1/zixuans8/miniconda3/envs/$ENV_NAME/bin/python}"
GPU_ID="${GPU_ID:-0}"
OUT_ROOT="${OUT_ROOT:-/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_topvenue_$(date +%Y%m%d_%H%M%S)}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

cd "$PROJECT_DIR"
mkdir -p "$OUT_ROOT"

echo "Prelaunch GPU status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

screen -dmS relclip_topvenue bash -lc "\
  cd '$PROJECT_DIR' && \
  PROJECT_DIR='$PROJECT_DIR' \
  BENCH_ROOT='$BENCH_ROOT' \
  OUT_ROOT='$OUT_ROOT' \
  PYTHON_BIN='$PYTHON_BIN' \
  GPU_ID='$GPU_ID' \
  TRAIN_EXAMPLES='${TRAIN_EXAMPLES:-2048}' \
  EVAL_LIMIT_PER_KEY='${EVAL_LIMIT_PER_KEY:-512}' \
  EPOCHS='${EPOCHS:-5}' \
  BATCH_SIZE='${BATCH_SIZE:-64}' \
  LR='${LR:-1e-4}' \
  PROTECTED_RANK='${PROTECTED_RANK:-64}' \
  ROP_LAMBDA='${ROP_LAMBDA:-0.1}' \
  FLICKR_LIMIT='${FLICKR_LIMIT:-1000}' \
  GUARDRAIL_RETRIEVAL_IMAGES='${GUARDRAIL_RETRIEVAL_IMAGES:-1000}' \
  GUARDRAIL_IMAGENET_EXAMPLES='${GUARDRAIL_IMAGENET_EXAMPLES:-5000}' \
  bash scripts/run_topvenue_batch.sh 2>&1 | tee '$OUT_ROOT/topvenue_batch.log'"

echo "Launched relclip_topvenue"
echo "OUT_ROOT=$OUT_ROOT"
screen -ls
