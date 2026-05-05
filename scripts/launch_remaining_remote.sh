#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/shared/nas/data/m1/zixuans8/CS510_Project}"
BENCH_ROOT="${BENCH_ROOT:-/shared/nas/data/m1/zixuans8/relbottleneck/benchmarks/public_hf}"
OUT_ROOT="${OUT_ROOT:-/shared/nas/data/m1/zixuans8/relbottleneck/outputs/runs}"
ENV_NAME="${ENV_NAME:-relclip}"
PYTHON_BIN="${PYTHON_BIN:-/shared/nas/data/m1/zixuans8/miniconda3/envs/$ENV_NAME/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

screen -dmS relclip_remaining bash -lc "PROJECT_DIR='$PROJECT_DIR' BENCH_ROOT='$BENCH_ROOT' OUT_ROOT='$OUT_ROOT' ENV_NAME='$ENV_NAME' PYTHON_BIN='$PYTHON_BIN' GPU_ID='${GPU_ID:-2}' bash '$PROJECT_DIR/scripts/run_remaining_batch.sh' 2>&1 | tee '$OUT_ROOT/remaining_batch.log'"

screen -ls
