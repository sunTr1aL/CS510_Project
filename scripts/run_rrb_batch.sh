#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/shared/nas/data/m1/zixuans8/CS510_Project}"
BENCH_ROOT="${BENCH_ROOT:-/shared/nas/data/m1/zixuans8/relbottleneck/benchmarks/public_hf}"
OUT_ROOT="${OUT_ROOT:-/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_$(date +%Y%m%d_%H%M%S)}"
ENV_NAME="${ENV_NAME:-relclip}"
PYTHON_BIN="${PYTHON_BIN:-/shared/nas/data/m1/zixuans8/miniconda3/envs/$ENV_NAME/bin/python}"
GPU_ID="${GPU_ID:-0}"
TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-2048}"
EVAL_LIMIT_PER_KEY="${EVAL_LIMIT_PER_KEY:-512}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
GUARDRAIL_RETRIEVAL_IMAGES="${GUARDRAIL_RETRIEVAL_IMAGES:-1000}"
GUARDRAIL_IMAGENET_EXAMPLES="${GUARDRAIL_IMAGENET_EXAMPLES:-5000}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

cd "$PROJECT_DIR"
mkdir -p "$OUT_ROOT"

rrb_args_for_run() {
  case "$1" in
    R019)
      echo "--residual-alpha 0.05 --residual-gate fixed --lambda-anchor 5 --lambda-delta 0.01 --lambda-targeted 1 --anchor-ratio 0.5"
      ;;
    R020)
      echo "--residual-alpha 0.10 --residual-gate fixed --lambda-anchor 5 --lambda-delta 0.01 --lambda-targeted 1 --anchor-ratio 0.5"
      ;;
    R021)
      echo "--residual-alpha 0.10 --residual-gate fixed --lambda-anchor 10 --lambda-delta 0.01 --lambda-targeted 1 --anchor-ratio 0.75"
      ;;
    R022)
      echo "--residual-alpha 0.10 --residual-gate vector --lambda-anchor 5 --lambda-delta 0.01 --lambda-targeted 1 --anchor-ratio 0.5"
      ;;
    R023)
      echo "--residual-alpha 0.05 --residual-gate fixed --lambda-anchor 5 --lambda-delta 0.01 --anchor-ratio 0.5"
      ;;
    R024)
      echo "--residual-alpha 0.05 --residual-gate fixed --lambda-anchor 5 --lambda-delta 0.01 --lambda-targeted 1 --anchor-ratio 0.5"
      ;;
    R025)
      echo "--residual-alpha 0.05 --residual-gate fixed --lambda-anchor 5 --lambda-delta 0.01 --lambda-targeted 1 --anchor-ratio 0.5"
      ;;
    R027)
      echo "--residual-alpha 0.05 --residual-gate fixed --lambda-anchor 5 --lambda-delta 0.01 --lambda-targeted 1 --anchor-ratio 0.5"
      ;;
    R028)
      echo "--residual-alpha 0.05 --residual-gate fixed --lambda-anchor 10 --lambda-delta 0.01 --anchor-ratio 0.75"
      ;;
    R029)
      echo "--residual-alpha 0.05 --residual-gate fixed --lambda-anchor 10 --lambda-delta 0.01 --lambda-targeted 1 --anchor-ratio 0.75"
      ;;
    R031|R032)
      echo "--residual-alpha 0.05 --residual-gate fixed --lambda-anchor 10 --lambda-delta 0.01 --anchor-ratio 0.75"
      ;;
    R033)
      echo "--residual-alpha 0.05 --residual-gate fixed --lambda-anchor 10 --lambda-delta 0.01 --anchor-ratio 0.75"
      ;;
    R035|R036|R037|R038|R039)
      echo "--residual-alpha 0.05 --residual-gate fixed --lambda-anchor 20 --lambda-delta 0.01 --anchor-ratio 0.75"
      ;;
    R040)
      echo "--residual-alpha 0.05 --residual-gate fixed --lambda-anchor 20 --lambda-delta 0.1 --anchor-ratio 0.75"
      ;;
    *)
      echo "Unsupported RRB run id: $1" >&2
      return 2
      ;;
  esac
}

run_one() {
  local rid="$1"
  local extra_args
  extra_args="$(rrb_args_for_run "$rid")"
  mkdir -p "$OUT_ROOT/$rid"
  {
    echo "==== $rid RRB train+eval on GPU $GPU_ID ===="
    date
    # shellcheck disable=SC2086
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
      --backend rrb \
      --full-eval-after-train \
      --write-records \
      $extra_args
  } 2>&1 | tee "$OUT_ROOT/$rid/launch.log"

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
}

if [ "$#" -eq 0 ]; then
  set -- R019 R020 R021 R022
fi

for rid in "$@"; do
  run_one "$rid"
done

mkdir -p "$OUT_ROOT/analysis"
"$PYTHON_BIN" scripts/summarize_paper_grade_results.py \
  --root "$OUT_ROOT" \
  --output "$OUT_ROOT/analysis/summary.md"

date
echo "RRB batch complete: $OUT_ROOT"
