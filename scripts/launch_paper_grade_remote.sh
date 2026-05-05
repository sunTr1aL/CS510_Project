#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/shared/nas/data/m1/zixuans8/CS510_Project}"
BENCH_ROOT="${BENCH_ROOT:-/shared/nas/data/m1/zixuans8/relbottleneck/benchmarks/public_hf}"
ENV_NAME="${ENV_NAME:-relclip}"
PYTHON_BIN="${PYTHON_BIN:-/shared/nas/data/m1/zixuans8/miniconda3/envs/$ENV_NAME/bin/python}"
GPU_ID="${GPU_ID:-2}"
OUT_ROOT="${OUT_ROOT:-/shared/nas/data/m1/zixuans8/relbottleneck/outputs/paper_grade_$(date +%Y%m%d_%H%M%S)}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

screen -dmS relclip_paper_grade bash -lc "PROJECT_DIR='$PROJECT_DIR' BENCH_ROOT='$BENCH_ROOT' OUT_ROOT='$OUT_ROOT' ENV_NAME='$ENV_NAME' PYTHON_BIN='$PYTHON_BIN' GPU_ID='$GPU_ID' TRAIN_EXAMPLES='${TRAIN_EXAMPLES:-2048}' EVAL_LIMIT_PER_KEY='${EVAL_LIMIT_PER_KEY:-512}' EPOCHS='${EPOCHS:-5}' BATCH_SIZE='${BATCH_SIZE:-64}' LR='${LR:-1e-4}' bash '$PROJECT_DIR/scripts/run_paper_grade_batch.sh' 2>&1 | tee '$OUT_ROOT/paper_grade_batch.log'"

echo "Launched relclip_paper_grade"
echo "Output root: $OUT_ROOT"
screen -ls
