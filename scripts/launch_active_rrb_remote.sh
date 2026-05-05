#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/shared/nas/data/m1/zixuans8/CS510_Project}"
BENCH_ROOT="${BENCH_ROOT:-/shared/nas/data/m1/zixuans8/relbottleneck/benchmarks/public_hf}"
ENV_NAME="${ENV_NAME:-relclip}"
PYTHON_BIN="${PYTHON_BIN:-/shared/nas/data/m1/zixuans8/miniconda3/envs/$ENV_NAME/bin/python}"
OUT_ROOT="${OUT_ROOT:-/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_active_$(date +%Y%m%d_%H%M%S)}"

TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-2048}"
EVAL_LIMIT_PER_KEY="${EVAL_LIMIT_PER_KEY:-512}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

cd "$PROJECT_DIR"
mkdir -p "$OUT_ROOT"

echo "Prelaunch GPU status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

launch_one() {
  local run_id="$1"
  local gpu_id="$2"
  local screen_name="relclip_${run_id}"
  local screen_log="$OUT_ROOT/${run_id}.screen.log"

  screen -dmS "$screen_name" bash -lc "\
    cd '$PROJECT_DIR' && \
    PROJECT_DIR='$PROJECT_DIR' \
    BENCH_ROOT='$BENCH_ROOT' \
    OUT_ROOT='$OUT_ROOT' \
    PYTHON_BIN='$PYTHON_BIN' \
    GPU_ID='$gpu_id' \
    TRAIN_EXAMPLES='$TRAIN_EXAMPLES' \
    EVAL_LIMIT_PER_KEY='$EVAL_LIMIT_PER_KEY' \
    EPOCHS='$EPOCHS' \
    BATCH_SIZE='$BATCH_SIZE' \
    LR='$LR' \
    bash scripts/run_rrb_batch.sh '$run_id' 2>&1 | tee '$screen_log'"

  echo "Launched $run_id on GPU $gpu_id in screen $screen_name"
}

launch_one R027 0
launch_one R028 1

echo "OUT_ROOT=$OUT_ROOT"
screen -ls
