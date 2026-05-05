#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/shared/nas/data/m1/zixuans8/CS510_Project}"
BENCH_ROOT="${BENCH_ROOT:-/shared/nas/data/m1/zixuans8/relbottleneck/benchmarks/public_hf}"
OUT_ROOT="${OUT_ROOT:-/shared/nas/data/m1/zixuans8/relbottleneck/outputs/runs}"
ENV_NAME="${ENV_NAME:-relclip}"
PYTHON_BIN="${PYTHON_BIN:-/shared/nas/data/m1/zixuans8/miniconda3/envs/$ENV_NAME/bin/python}"
GPU_ID="${GPU_ID:-2}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

cd "$PROJECT_DIR"
mkdir -p "$OUT_ROOT"

run_train() {
  local rid="$1"
  local extra_args="${2:-}"
  mkdir -p "$OUT_ROOT/$rid"
  {
    echo "==== $rid ===="
    date
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m experiments.run_experiment \
      --run-id "$rid" \
      --dataset-root "$BENCH_ROOT" \
      --benchmark-root "$BENCH_ROOT" \
      --output-root "$OUT_ROOT" \
      --device cuda \
      --max-examples-per-key 64 \
      --epochs 2 \
      --batch-size 16 \
      $extra_args
  } 2>&1 | tee "$OUT_ROOT/$rid/launch.log"
}

for rid in R005 R006 R007 R008 R009 R010 R011 R012 R013 R014 R015 R016 R017; do
  run_train "$rid"
done

mkdir -p "$OUT_ROOT/R018"
{
  echo "==== R018 ===="
  date
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m experiments.run_experiment \
    --run-id R018 \
    --benchmark-root "$BENCH_ROOT" \
    --checkpoint "$OUT_ROOT/R011/adapter.pt" \
    --output-root "$OUT_ROOT" \
    --device cuda
} 2>&1 | tee "$OUT_ROOT/R018/launch.log"

date
echo "remaining batch complete"

