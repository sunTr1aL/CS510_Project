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
GUARDRAIL_RETRIEVAL_IMAGES="${GUARDRAIL_RETRIEVAL_IMAGES:-1000}"
GUARDRAIL_IMAGENET_EXAMPLES="${GUARDRAIL_IMAGENET_EXAMPLES:-5000}"

cd "$PROJECT_DIR"
mkdir -p "$OUT_ROOT"

run_checkpoint_eval() {
  local rid="$1"
  mkdir -p "$OUT_ROOT/$rid"
  {
    echo "==== $rid checkpoint eval ===="
    date
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
  } 2>&1 | tee "$OUT_ROOT/$rid/resume_eval.log"
}

run_guardrails() {
  local rid="$1"
  mkdir -p "$OUT_ROOT/guardrails/$rid"
  {
    echo "==== $rid guardrails ===="
    date
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/evaluate_guardrails.py \
      --benchmark-root "$BENCH_ROOT" \
      --checkpoint "$OUT_ROOT/$rid/adapter.pt" \
      --output-dir "$OUT_ROOT/guardrails/$rid" \
      --device cuda \
      --max-retrieval-images "$GUARDRAIL_RETRIEVAL_IMAGES" \
      --max-imagenet-examples "$GUARDRAIL_IMAGENET_EXAMPLES" \
      --batch-size "$BATCH_SIZE"
  } 2>&1 | tee "$OUT_ROOT/guardrails/$rid/resume_guardrails.log"
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
      --full-eval-after-train \
      --write-records
  } 2>&1 | tee "$OUT_ROOT/$rid/resume_train.log"
}

if [ ! -f "$OUT_ROOT/R006/checkpoint_eval_summary.json" ]; then
  run_checkpoint_eval R006
fi
if [ ! -f "$OUT_ROOT/guardrails/R006/guardrail_summary.json" ]; then
  run_guardrails R006
fi

for rid in R007 R008 R009 R010 R011 R012 R013 R014 R015 R016 R017; do
  if [ ! -f "$OUT_ROOT/$rid/checkpoint_eval_summary.json" ]; then
    run_train "$rid"
  fi
  if [ ! -f "$OUT_ROOT/guardrails/$rid/guardrail_summary.json" ]; then
    run_guardrails "$rid"
  fi
done

mkdir -p "$OUT_ROOT/analysis"
"$PYTHON_BIN" scripts/summarize_paper_grade_results.py \
  --root "$OUT_ROOT" \
  --output "$OUT_ROOT/analysis/summary.md"

date
echo "resume complete: $OUT_ROOT"
