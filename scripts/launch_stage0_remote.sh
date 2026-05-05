#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/shared/nas/data/m1/zixuans8/CS510_Project}"
DATA_ROOT="${DATA_ROOT:-/shared/nas/data/m1/zixuans8/relbottleneck/data}"
BENCH_ROOT="${BENCH_ROOT:-/shared/nas/data/m1/zixuans8/relbottleneck/benchmarks}"
OUT_ROOT="${OUT_ROOT:-/shared/nas/data/m1/zixuans8/relbottleneck/outputs/runs}"
AUDIT_JSONL="${AUDIT_JSONL:-$DATA_ROOT/audit/structured_edits_200.jsonl}"
ENV_NAME="${ENV_NAME:-relclip}"
PYTHON_BIN="${PYTHON_BIN:-/shared/nas/data/m1/zixuans8/miniconda3/envs/$ENV_NAME/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
cd "$PROJECT_DIR"
mkdir -p "$OUT_ROOT"/R001 "$OUT_ROOT"/R002 "$OUT_ROOT"/R003
mkdir -p "$OUT_ROOT"/R004

if [ -f "$AUDIT_JSONL" ]; then
  R001_ARGS="--audit-jsonl '$AUDIT_JSONL'"
  R002_ARGS="--audit-jsonl '$AUDIT_JSONL'"
else
  echo "Audit JSONL not found at $AUDIT_JSONL; launching R001/R002 smoke checks."
  R001_ARGS="--smoke"
  R002_ARGS="--smoke"
fi

screen -dmS relclip_R001 bash -lc "cd '$PROJECT_DIR' && '$PYTHON_BIN' -m experiments.run_experiment --run-id R001 $R001_ARGS --output-root '$OUT_ROOT' 2>&1 | tee '$OUT_ROOT/R001/launch.log'"
screen -dmS relclip_R002 bash -lc "cd '$PROJECT_DIR' && '$PYTHON_BIN' -m experiments.run_experiment --run-id R002 $R002_ARGS --output-root '$OUT_ROOT' 2>&1 | tee '$OUT_ROOT/R002/launch.log'"
screen -dmS relclip_R003 bash -lc "cd '$PROJECT_DIR' && CUDA_VISIBLE_DEVICES=0 '$PYTHON_BIN' -m experiments.run_experiment --run-id R003 --benchmark-root '$BENCH_ROOT' --output-root '$OUT_ROOT' --device cuda --max-examples-per-key 32 2>&1 | tee '$OUT_ROOT/R003/launch.log'"
screen -dmS relclip_R004 bash -lc "cd '$PROJECT_DIR' && CUDA_VISIBLE_DEVICES=1 '$PYTHON_BIN' -m experiments.run_experiment --run-id R004 --dataset-root '$BENCH_ROOT' --benchmark-root '$BENCH_ROOT' --output-root '$OUT_ROOT' --device cuda --max-examples-per-key 64 --epochs 2 --batch-size 16 2>&1 | tee '$OUT_ROOT/R004/launch.log'"

screen -ls
