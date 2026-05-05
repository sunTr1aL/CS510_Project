"""Run COCO retrieval and ImageNet zero-shot guardrail evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.clip_backend import evaluate_coco_retrieval, evaluate_imagenet_zeroshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-name", default="ViT-B-16")
    parser.add_argument("--pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-retrieval-images", type=int, default=1000)
    parser.add_argument("--max-imagenet-examples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-imagenet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    if not args.skip_retrieval:
        retrieval = evaluate_coco_retrieval(
            dataset_root=args.benchmark_root,
            output_path=args.output_dir / "coco_retrieval_summary.json",
            checkpoint_path=args.checkpoint,
            max_images=args.max_retrieval_images,
            batch_size=args.batch_size,
            model_name=args.model_name,
            pretrained=args.pretrained,
            device=args.device,
        )
        outputs["coco_retrieval"] = retrieval.__dict__
    if not args.skip_imagenet:
        imagenet = evaluate_imagenet_zeroshot(
            dataset_root=args.benchmark_root,
            output_path=args.output_dir / "imagenet_zeroshot_summary.json",
            checkpoint_path=args.checkpoint,
            max_examples=args.max_imagenet_examples,
            batch_size=args.batch_size,
            model_name=args.model_name,
            pretrained=args.pretrained,
            device=args.device,
        )
        outputs["imagenet_zeroshot"] = imagenet.__dict__
    (args.output_dir / "guardrail_summary.json").write_text(
        json.dumps(outputs, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
