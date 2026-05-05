#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/shared/nas/data/m1/zixuans8/CS510_Project}"
BENCH_ROOT="${BENCH_ROOT:-/shared/nas/data/m1/zixuans8/relbottleneck/benchmarks/public_hf}"
OUT_ROOT="${OUT_ROOT:-/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_topvenue_$(date +%Y%m%d_%H%M%S)}"
ENV_NAME="${ENV_NAME:-relclip}"
PYTHON_BIN="${PYTHON_BIN:-/shared/nas/data/m1/zixuans8/miniconda3/envs/$ENV_NAME/bin/python}"
GPU_ID="${GPU_ID:-0}"
TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-2048}"
EVAL_LIMIT_PER_KEY="${EVAL_LIMIT_PER_KEY:-512}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
PROTECTED_RANK="${PROTECTED_RANK:-64}"
ROP_LAMBDA="${ROP_LAMBDA:-0.1}"
FLICKR_LIMIT="${FLICKR_LIMIT:-1000}"
GUARDRAIL_RETRIEVAL_IMAGES="${GUARDRAIL_RETRIEVAL_IMAGES:-1000}"
GUARDRAIL_IMAGENET_EXAMPLES="${GUARDRAIL_IMAGENET_EXAMPLES:-5000}"

R005_CKPT="${R005_CKPT:-/shared/nas/data/m1/zixuans8/relbottleneck/outputs/updated_end2end_20260428_115746/R005/adapter.pt}"
R028_CKPT="${R028_CKPT:-/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_active_20260429_155721/R028/adapter.pt}"
R035_CKPT="${R035_CKPT:-/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_m8_gpu0_20260429_211743/R035/adapter.pt}"
R038_CKPT="${R038_CKPT:-/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_m8_gpu1_20260429_211743/R038/adapter.pt}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

cd "$PROJECT_DIR"
mkdir -p "$OUT_ROOT"

run_slot_probe() {
  local label="$1"
  local checkpoint="$2"
  if [ ! -f "$checkpoint" ]; then
    echo "Skipping R041 $label; missing checkpoint: $checkpoint"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m experiments.run_experiment \
    --run-id R041 \
    --checkpoint "$checkpoint" \
    --benchmark-root "$BENCH_ROOT" \
    --output-root "$OUT_ROOT" \
    --device cuda \
    --eval-limit-per-key "$EVAL_LIMIT_PER_KEY" \
    --variant-mode "$label" \
    2>&1 | tee "$OUT_ROOT/R041/${label}_launch.log"
}

run_rrb_rop() {
  local output_name="$1"
  local seed="$2"
  mkdir -p "$OUT_ROOT/$output_name"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m experiments.run_experiment \
    --run-id R044 \
    --output-name "$output_name" \
    --seed "$seed" \
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
    --residual-alpha 0.05 \
    --residual-gate fixed \
    --lambda-anchor 20 \
    --lambda-delta 0.01 \
    --anchor-ratio 0.75 \
    --protected-basis "$OUT_ROOT/R042/protected_basis.pt" \
    --lambda-rop "$ROP_LAMBDA" \
    --rop-mode penalty_project \
    --full-eval-after-train \
    --write-records \
    2>&1 | tee "$OUT_ROOT/$output_name/launch.log"
}

run_guardrails() {
  local label="$1"
  local checkpoint="$2"
  mkdir -p "$OUT_ROOT/guardrails/$label"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/evaluate_guardrails.py \
    --benchmark-root "$BENCH_ROOT" \
    --checkpoint "$checkpoint" \
    --output-dir "$OUT_ROOT/guardrails/$label" \
    --device cuda \
    --max-retrieval-images "$GUARDRAIL_RETRIEVAL_IMAGES" \
    --max-imagenet-examples "$GUARDRAIL_IMAGENET_EXAMPLES" \
    --batch-size "$BATCH_SIZE" \
    2>&1 | tee "$OUT_ROOT/guardrails/$label/launch.log"
}

run_flickr() {
  local label="$1"
  local checkpoint="${2:-}"
  local args=()
  if [ -n "$checkpoint" ]; then
    if [ ! -f "$checkpoint" ]; then
      echo "Skipping R045 $label; missing checkpoint: $checkpoint"
      return 0
    fi
    args+=(--checkpoint "$checkpoint")
  fi
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m experiments.run_experiment \
    --run-id R045 \
    "${args[@]}" \
    --benchmark-root "$BENCH_ROOT" \
    --output-root "$OUT_ROOT" \
    --device cuda \
    --eval-limit-per-key "$FLICKR_LIMIT" \
    --batch-size "$BATCH_SIZE" \
    --variant-mode "$label" \
    2>&1 | tee "$OUT_ROOT/R045/${label}_launch.log"
}

echo "R041 slot responsibility probes"
mkdir -p "$OUT_ROOT/R041"
run_slot_probe R028 "$R028_CKPT"
run_slot_probe R035 "$R035_CKPT"
run_slot_probe R038 "$R038_CKPT"

echo "R042 ROP basis"
mkdir -p "$OUT_ROOT/R042"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m experiments.run_experiment \
  --run-id R042 \
  --benchmark-root "$BENCH_ROOT" \
  --output-root "$OUT_ROOT" \
  --device cuda \
  --max-examples-per-key "$TRAIN_EXAMPLES" \
  --protected-rank "$PROTECTED_RANK" \
  --batch-size "$BATCH_SIZE" \
  2>&1 | tee "$OUT_ROOT/R042/launch.log"

echo "R043 ROP pilot"
mkdir -p "$OUT_ROOT/R043"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m experiments.run_experiment \
  --run-id R043 \
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
  --residual-alpha 0.05 \
  --residual-gate fixed \
  --lambda-anchor 20 \
  --lambda-delta 0.01 \
  --anchor-ratio 0.75 \
  --protected-basis "$OUT_ROOT/R042/protected_basis.pt" \
  --lambda-rop "$ROP_LAMBDA" \
  --rop-mode penalty_project \
  --full-eval-after-train \
  --write-records \
  2>&1 | tee "$OUT_ROOT/R043/launch.log"
run_guardrails R043 "$OUT_ROOT/R043/adapter.pt"

"$PYTHON_BIN" - "$OUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
eval_summary = json.loads((root / "R043" / "checkpoint_eval_summary.json").read_text())
guardrail = json.loads((root / "guardrails" / "R043" / "guardrail_summary.json").read_text())
coco = guardrail.get("coco_retrieval") or {}
imagenet = guardrail.get("imagenet_zeroshot") or {}
payload = {
    "relation_accuracy": eval_summary.get("accuracy"),
    "coco_i2t_r1": coco.get("image_to_text_r1"),
    "imagenet_top1": imagenet.get("top1"),
    "relation_threshold": 0.76,
    "coco_i2t_r1_threshold": 0.734,
    "imagenet_top1_threshold": 0.634,
}
payload["passes"] = (
    payload["relation_accuracy"] is not None
    and payload["coco_i2t_r1"] is not None
    and payload["imagenet_top1"] is not None
    and payload["relation_accuracy"] >= payload["relation_threshold"]
    and payload["coco_i2t_r1"] >= payload["coco_i2t_r1_threshold"]
    and payload["imagenet_top1"] >= payload["imagenet_top1_threshold"]
)
(root / "R043_gate.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

if "$PYTHON_BIN" - "$OUT_ROOT/R043_gate.json" <<'PY'
import json
import sys
from pathlib import Path
raise SystemExit(0 if json.loads(Path(sys.argv[1]).read_text()).get("passes") else 1)
PY
then
  echo "R043 passed gate; launching R044 seed 2 and seed 3"
  run_rrb_rop R044_seed2 2
  run_guardrails R044_seed2 "$OUT_ROOT/R044_seed2/adapter.pt"
  run_rrb_rop R044_seed3 3
  run_guardrails R044_seed3 "$OUT_ROOT/R044_seed3/adapter.pt"
else
  echo "R043 failed gate; R044 replication seeds are skipped"
fi

echo "R045 Flickr30k retrieval"
mkdir -p "$OUT_ROOT/R045"
run_flickr OpenCLIP
run_flickr R005 "$R005_CKPT"
run_flickr R028 "$R028_CKPT"
run_flickr R038 "$R038_CKPT"
if [ -f "$OUT_ROOT/R043/adapter.pt" ]; then
  run_flickr R043 "$OUT_ROOT/R043/adapter.pt"
fi

mkdir -p "$OUT_ROOT/analysis"
"$PYTHON_BIN" scripts/summarize_topvenue_results.py \
  --root "$OUT_ROOT" \
  --output "$OUT_ROOT/analysis/R041_R045_results.md"

date
echo "Top-venue batch complete: $OUT_ROOT"
