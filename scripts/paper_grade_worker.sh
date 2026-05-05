#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/shared/nas/data/m1/zixuans8/CS510_Project}"
BENCH_ROOT="${BENCH_ROOT:-/shared/nas/data/m1/zixuans8/relbottleneck/benchmarks/public_hf}"
OUT_ROOT="${OUT_ROOT:-/shared/nas/data/m1/zixuans8/relbottleneck/outputs/paper_grade_20260424_195455}"
ENV_NAME="${ENV_NAME:-relclip}"
PYTHON_BIN="${PYTHON_BIN:-/shared/nas/data/m1/zixuans8/miniconda3/envs/$ENV_NAME/bin/python}"
GPU_ID="${GPU_ID:-0}"
TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-2048}"
EVAL_LIMIT_PER_KEY="${EVAL_LIMIT_PER_KEY:-512}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
LAMBDA_DISTILL="${LAMBDA_DISTILL:-0.0}"
LAMBDA_ORIGINAL="${LAMBDA_ORIGINAL:-0.0}"
LAMBDA_TARGETED="${LAMBDA_TARGETED:-0.0}"
BACKEND="${BACKEND:-adapter}"
GUARDRAIL_RETRIEVAL_IMAGES="${GUARDRAIL_RETRIEVAL_IMAGES:-1000}"
GUARDRAIL_IMAGENET_EXAMPLES="${GUARDRAIL_IMAGENET_EXAMPLES:-5000}"

cd "$PROJECT_DIR"
mkdir -p "$OUT_ROOT"

checkpoint_eval() {
  local rid="$1"
  RID="$rid" OUT_ROOT="$OUT_ROOT" BENCH_ROOT="$BENCH_ROOT" EVAL_LIMIT_PER_KEY="$EVAL_LIMIT_PER_KEY" \
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

from experiments.clip_backend import evaluate_adapter_checkpoint

rid = os.environ["RID"]
out_root = Path(os.environ["OUT_ROOT"])
summary = evaluate_adapter_checkpoint(
    checkpoint_path=out_root / rid / "adapter.pt",
    dataset_root=Path(os.environ["BENCH_ROOT"]),
    output_path=out_root / rid / "checkpoint_eval_summary.json",
    max_examples_per_key=int(os.environ["EVAL_LIMIT_PER_KEY"]),
    device="cuda",
    write_records=True,
)
print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
PY
}

train_eval() {
  local rid="$1"
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
}

guardrails() {
  local rid="$1"
  mkdir -p "$OUT_ROOT/guardrails/$rid"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/evaluate_guardrails.py \
    --benchmark-root "$BENCH_ROOT" \
    --checkpoint "$OUT_ROOT/$rid/adapter.pt" \
    --output-dir "$OUT_ROOT/guardrails/$rid" \
    --device cuda \
    --max-retrieval-images "$GUARDRAIL_RETRIEVAL_IMAGES" \
    --max-imagenet-examples "$GUARDRAIL_IMAGENET_EXAMPLES" \
    --batch-size "$BATCH_SIZE"
}

for rid in "$@"; do
  mkdir -p "$OUT_ROOT/$rid"
  echo "==== $rid worker on GPU $GPU_ID ===="
  date
  if [ ! -f "$OUT_ROOT/$rid/checkpoint_eval_summary.json" ]; then
    if [ -f "$OUT_ROOT/$rid/adapter.pt" ]; then
      checkpoint_eval "$rid" 2>&1 | tee "$OUT_ROOT/$rid/worker_eval.log"
    else
      train_eval "$rid" 2>&1 | tee "$OUT_ROOT/$rid/worker_train.log"
    fi
  else
    echo "$rid checkpoint eval already exists; skipping train/eval"
  fi
  if [ ! -f "$OUT_ROOT/guardrails/$rid/guardrail_summary.json" ]; then
    mkdir -p "$OUT_ROOT/guardrails/$rid"
    guardrails "$rid" 2>&1 | tee "$OUT_ROOT/guardrails/$rid/worker_guardrails.log"
  else
    echo "$rid guardrails already exist; skipping guardrails"
  fi
done
