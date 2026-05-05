#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/shared/nas/data/m1/zixuans8/CS510_Project}"
BENCH_ROOT="${BENCH_ROOT:-/shared/nas/data/m1/zixuans8/relbottleneck/benchmarks/public_hf}"
OUT_ROOT="${OUT_ROOT:-/shared/nas/data/m1/zixuans8/relbottleneck/outputs/paper_grade_$(date +%Y%m%d_%H%M%S)}"
ENV_NAME="${ENV_NAME:-relclip}"
PYTHON_BIN="${PYTHON_BIN:-/shared/nas/data/m1/zixuans8/miniconda3/envs/$ENV_NAME/bin/python}"
GPU_ID="${GPU_ID:-2}"
TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-2048}"
EVAL_LIMIT_PER_KEY="${EVAL_LIMIT_PER_KEY:-512}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
LAMBDA_DISTILL="${LAMBDA_DISTILL:-0.0}"
LAMBDA_ORIGINAL="${LAMBDA_ORIGINAL:-0.0}"
LAMBDA_TARGETED="${LAMBDA_TARGETED:-0.0}"
BACKEND="${BACKEND:-adapter}"
GUARDRAIL_RETRIEVAL_IMAGES="${GUARDRAIL_RETRIEVAL_IMAGES:-1000}"
GUARDRAIL_IMAGENET_EXAMPLES="${GUARDRAIL_IMAGENET_EXAMPLES:-5000}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

cd "$PROJECT_DIR"
mkdir -p "$OUT_ROOT"

run_eval() {
  local rid="$1"
  local checkpoint="${2:-}"
  mkdir -p "$OUT_ROOT/$rid"
  {
    echo "==== $rid eval ===="
    date
    if [ -n "$checkpoint" ]; then
      CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m experiments.run_experiment \
        --run-id "$rid" \
        --checkpoint "$checkpoint" \
        --benchmark-root "$BENCH_ROOT" \
        --output-root "$OUT_ROOT" \
        --device cuda \
        --eval-limit-per-key "$EVAL_LIMIT_PER_KEY" \
        --write-records
    else
      CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m experiments.run_experiment \
        --run-id "$rid" \
        --benchmark-root "$BENCH_ROOT" \
        --output-root "$OUT_ROOT" \
        --device cuda \
        --eval-limit-per-key "$EVAL_LIMIT_PER_KEY" \
        --write-records
    fi
  } 2>&1 | tee "$OUT_ROOT/$rid/launch.log"
}

run_train() {
  local rid="$1"
  mkdir -p "$OUT_ROOT/$rid"
  {
    echo "==== $rid train+eval ===="
    date
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m experiments.run_experiment \
      --run-id "$rid" \
      --dataset-root "$BENCH_ROOT" \
      --benchmark-root "$BENCH_ROOT" \
      --output-root "$OUT_ROOT" \
      --device cuda \
      --max-examples-per-key "$TRAIN_EXAMPLES" \
      --eval-limit-per-key "$EVAL_LIMIT_PER_KEY" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --lr "$LR" \
      --lambda-distill "$LAMBDA_DISTILL" \
      --lambda-original "$LAMBDA_ORIGINAL" \
      --lambda-targeted "$LAMBDA_TARGETED" \
      --backend "$BACKEND" \
      --full-eval-after-train \
      --write-records
  } 2>&1 | tee "$OUT_ROOT/$rid/launch.log"
}

run_eval R003

mkdir -p "$OUT_ROOT/guardrails/R003"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/evaluate_guardrails.py \
  --benchmark-root "$BENCH_ROOT" \
  --output-dir "$OUT_ROOT/guardrails/R003" \
  --device cuda \
  --max-retrieval-images "$GUARDRAIL_RETRIEVAL_IMAGES" \
  --max-imagenet-examples "$GUARDRAIL_IMAGENET_EXAMPLES" \
  --batch-size "$BATCH_SIZE" \
  2>&1 | tee "$OUT_ROOT/guardrails/R003/launch.log"

for rid in R005 R006 R007 R008 R009 R010 R011 R012 R013 R014 R015 R016 R017; do
  run_train "$rid"
  mkdir -p "$OUT_ROOT/guardrails/$rid"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/evaluate_guardrails.py \
    --benchmark-root "$BENCH_ROOT" \
    --checkpoint "$OUT_ROOT/$rid/adapter.pt" \
    --output-dir "$OUT_ROOT/guardrails/$rid" \
    --device cuda \
    --max-retrieval-images "$GUARDRAIL_RETRIEVAL_IMAGES" \
    --max-imagenet-examples "$GUARDRAIL_IMAGENET_EXAMPLES" \
    --batch-size "$BATCH_SIZE" \
    2>&1 | tee "$OUT_ROOT/guardrails/$rid/launch.log"
done

mkdir -p "$OUT_ROOT/analysis"
"$PYTHON_BIN" scripts/summarize_paper_grade_results.py \
  --root "$OUT_ROOT" \
  --output "$OUT_ROOT/analysis/summary.md"

date
echo "paper-grade batch complete: $OUT_ROOT"
