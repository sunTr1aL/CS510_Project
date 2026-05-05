#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/shared/nas/data/m1/zixuans8/CS510_Project}"
BENCH_ROOT="${BENCH_ROOT:-/shared/nas/data/m1/zixuans8/relbottleneck/benchmarks/public_hf}"
OUT_ROOT="${OUT_ROOT:-/shared/nas/data/m1/zixuans8/relbottleneck/outputs/winoground_r034_$(date +%Y%m%d_%H%M%S)}"
ENV_NAME="${ENV_NAME:-relclip}"
PYTHON_BIN="${PYTHON_BIN:-/shared/nas/data/m1/zixuans8/miniconda3/envs/$ENV_NAME/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
R028_CKPT="${R028_CKPT:-/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_active_20260429_155721/R028/adapter.pt}"
R005_CKPT="${R005_CKPT:-/shared/nas/data/m1/zixuans8/relbottleneck/outputs/updated_end2end_20260428_115746/R005/adapter.pt}"

cd "$PROJECT_DIR"
mkdir -p "$OUT_ROOT/R034"

run_eval() {
  local label="$1"
  shift
  echo "==== R034 Winoground $label on GPU $GPU_ID ===="
  date
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m experiments.run_experiment \
    --run-id R034 \
    --benchmark-root "$BENCH_ROOT" \
    --output-root "$OUT_ROOT" \
    --device "$DEVICE" \
    --variant-mode "$label" \
    --write-records \
    "$@"
}

run_eval R003_openclip
run_eval R028_rrb --checkpoint "$R028_CKPT"
run_eval R005_plain --checkpoint "$R005_CKPT"

"$PYTHON_BIN" scripts/summarize_winoground_r034.py \
  --root "$OUT_ROOT" \
  --output "$OUT_ROOT/analysis/winoground_summary.md"

date
echo "R034 Winoground complete: $OUT_ROOT"
