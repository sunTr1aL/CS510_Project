"""CLI launcher for RelBottleneck-CLIP experiment runs."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Dict, List

from .audit_edits import load_jsonl, smoke_records, summarize, write_summary
from .configs import get_run, iter_runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List registered runs")
    parser.add_argument("--show", metavar="RUN_ID", help="Print one run spec as JSON")
    parser.add_argument("--run-id", help="Run id, e.g. R001")
    parser.add_argument("--dry-run", action="store_true", help="Only write/print the manifest")
    parser.add_argument("--smoke", action="store_true", help="Use built-in tiny inputs for a local smoke check")
    parser.add_argument("--audit-jsonl", type=Path, help="JSONL structured edits for audit/filter runs")
    parser.add_argument("--dataset-root", type=Path, help="Training dataset root")
    parser.add_argument("--benchmark-root", type=Path, help="Evaluation benchmark root")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint path for eval/diagnostics")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--output-name", help="Override output subdirectory name for repeated seeds")
    parser.add_argument("--slot-count", type=int, help="Override registered slot count")
    parser.add_argument("--seed", type=int, help="Override registered seed")
    parser.add_argument("--device", default="cuda", help="cuda, mps, or cpu")
    parser.add_argument("--model-name", default="ViT-B-16")
    parser.add_argument("--pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--max-examples-per-key", type=int, default=256)
    parser.add_argument("--eval-limit-per-key", type=int, help="Override evaluation examples per source")
    parser.add_argument("--eval-keys", help="Comma-separated evaluation source keys")
    parser.add_argument("--write-records", action="store_true", help="Write per-example JSONL eval records")
    parser.add_argument(
        "--full-eval-after-train",
        action="store_true",
        help="Evaluate the saved adapter checkpoint on --benchmark-root after training",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lambda-distill",
        type=float,
        default=0.0,
        help="Weight for preserving base OpenCLIP image/text embeddings during adapter training",
    )
    parser.add_argument(
        "--lambda-original",
        type=float,
        default=0.0,
        help="Weight for clean in-batch image/text alignment mixed into margin-based training",
    )
    parser.add_argument(
        "--lambda-targeted",
        type=float,
        default=0.0,
        help="Weight for targeted slot loss in token_bottleneck training",
    )
    parser.add_argument(
        "--lambda-anchor",
        type=float,
        default=0.0,
        help="Weight for RRB feature mimicry against frozen OpenCLIP embeddings",
    )
    parser.add_argument(
        "--lambda-delta",
        type=float,
        default=0.0,
        help="Weight for RRB residual norm regularization",
    )
    parser.add_argument(
        "--residual-alpha",
        type=float,
        default=0.05,
        help="Maximum/fixed RRB residual gate value",
    )
    parser.add_argument(
        "--residual-gate",
        choices=["fixed", "scalar", "vector"],
        default="fixed",
        help="RRB residual gate type",
    )
    parser.add_argument(
        "--anchor-ratio",
        type=float,
        default=0.5,
        help="RRB anchor/intervention balance; 0.5 approximates a 50/50 batch mix",
    )
    parser.add_argument("--protected-basis", type=Path, help="ROP protected-basis .pt file from R042")
    parser.add_argument("--protected-rank", type=int, default=64, help="Number of PCA directions for R042")
    parser.add_argument("--lambda-rop", type=float, default=0.0, help="Weight for protected-subspace residual penalty")
    parser.add_argument(
        "--rop-mode",
        choices=["none", "penalty", "project", "penalty_project"],
        default="none",
        help="ROP preservation mode for R043/R044",
    )
    parser.add_argument("--flickr-split", default="test", help="Flickr30k split for R045 retrieval")
    parser.add_argument(
        "--backend",
        choices=["adapter", "token_bottleneck", "rrb"],
        default="adapter",
        help="Training backend: pooled-feature adapter, hard token bottleneck, or residual RRB",
    )
    parser.add_argument("--variant-mode", help="Override the registered variant name")
    return parser


def shell_command(args: argparse.Namespace) -> str:
    parts = [
        "python3",
        "-m",
        "experiments.run_experiment",
        "--run-id",
        args.run_id.upper(),
        "--output-root",
        str(args.output_root),
        "--device",
        args.device,
    ]
    if args.dry_run:
        parts.append("--dry-run")
    if args.smoke:
        parts.append("--smoke")
    if args.audit_jsonl:
        parts.extend(["--audit-jsonl", str(args.audit_jsonl)])
    if args.dataset_root:
        parts.extend(["--dataset-root", str(args.dataset_root)])
    if args.benchmark_root:
        parts.extend(["--benchmark-root", str(args.benchmark_root)])
    if args.checkpoint:
        parts.extend(["--checkpoint", str(args.checkpoint)])
    if args.output_name:
        parts.extend(["--output-name", args.output_name])
    if args.slot_count is not None:
        parts.extend(["--slot-count", str(args.slot_count)])
    if args.seed is not None:
        parts.extend(["--seed", str(args.seed)])
    parts.extend(["--model-name", args.model_name])
    parts.extend(["--pretrained", args.pretrained])
    parts.extend(["--max-examples-per-key", str(args.max_examples_per_key)])
    if args.eval_limit_per_key is not None:
        parts.extend(["--eval-limit-per-key", str(args.eval_limit_per_key)])
    if args.eval_keys:
        parts.extend(["--eval-keys", args.eval_keys])
    if args.write_records:
        parts.append("--write-records")
    if args.full_eval_after_train:
        parts.append("--full-eval-after-train")
    parts.extend(["--epochs", str(args.epochs)])
    parts.extend(["--batch-size", str(args.batch_size)])
    parts.extend(["--lr", str(args.lr)])
    if args.lambda_distill:
        parts.extend(["--lambda-distill", str(args.lambda_distill)])
    if args.lambda_original:
        parts.extend(["--lambda-original", str(args.lambda_original)])
    if args.lambda_targeted:
        parts.extend(["--lambda-targeted", str(args.lambda_targeted)])
    if args.lambda_anchor:
        parts.extend(["--lambda-anchor", str(args.lambda_anchor)])
    if args.lambda_delta:
        parts.extend(["--lambda-delta", str(args.lambda_delta)])
    if args.residual_alpha != 0.05:
        parts.extend(["--residual-alpha", str(args.residual_alpha)])
    if args.residual_gate != "fixed":
        parts.extend(["--residual-gate", args.residual_gate])
    if args.anchor_ratio != 0.5:
        parts.extend(["--anchor-ratio", str(args.anchor_ratio)])
    if args.protected_basis:
        parts.extend(["--protected-basis", str(args.protected_basis)])
    if args.protected_rank != 64:
        parts.extend(["--protected-rank", str(args.protected_rank)])
    if args.lambda_rop:
        parts.extend(["--lambda-rop", str(args.lambda_rop)])
    if args.rop_mode != "none":
        parts.extend(["--rop-mode", args.rop_mode])
    if args.flickr_split != "test":
        parts.extend(["--flickr-split", args.flickr_split])
    if args.backend != "adapter":
        parts.extend(["--backend", args.backend])
    if args.variant_mode:
        parts.extend(["--variant-mode", args.variant_mode])
    return " ".join(shlex.quote(part) for part in parts)


def run_manifest(args: argparse.Namespace) -> Dict[str, object]:
    spec = get_run(args.run_id)
    slot_count = args.slot_count if args.slot_count is not None else spec.slot_count
    seed = args.seed if args.seed is not None else spec.seed
    output_name = args.output_name or spec.run_id
    run_dir = args.output_root / output_name
    return {
        "run": spec.to_dict(),
        "resolved": {
            "slot_count": slot_count,
            "seed": seed,
            "device": args.device,
            "model_name": args.model_name,
            "pretrained": args.pretrained,
            "max_examples_per_key": args.max_examples_per_key,
            "eval_limit_per_key": args.eval_limit_per_key,
            "eval_keys": args.eval_keys,
            "write_records": args.write_records,
            "full_eval_after_train": args.full_eval_after_train,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lambda_distill": args.lambda_distill,
            "lambda_original": args.lambda_original,
            "lambda_targeted": args.lambda_targeted,
            "lambda_anchor": args.lambda_anchor,
            "lambda_delta": args.lambda_delta,
            "residual_alpha": args.residual_alpha,
            "residual_gate": args.residual_gate,
            "anchor_ratio": args.anchor_ratio,
            "protected_basis": str(args.protected_basis) if args.protected_basis else None,
            "protected_rank": args.protected_rank,
            "lambda_rop": args.lambda_rop,
            "rop_mode": args.rop_mode,
            "flickr_split": args.flickr_split,
            "backend": args.backend,
            "variant_mode": args.variant_mode,
            "dataset_root": str(args.dataset_root) if args.dataset_root else None,
            "benchmark_root": str(args.benchmark_root) if args.benchmark_root else None,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "audit_jsonl": str(args.audit_jsonl) if args.audit_jsonl else None,
            "smoke": args.smoke,
            "dry_run": args.dry_run,
            "output_name": output_name,
            "output_dir": str(run_dir),
        },
        "command": shell_command(args),
    }


def validate_real_inputs(args: argparse.Namespace, stage: str) -> None:
    if args.smoke or args.dry_run:
        return
    if stage in {"audit", "filter_audit"}:
        if not args.audit_jsonl:
            raise SystemExit("--audit-jsonl is required for audit/filter runs unless --smoke is set")
        if not args.audit_jsonl.exists():
            raise SystemExit(f"Audit JSONL does not exist: {args.audit_jsonl}")
    if stage == "train":
        if not args.dataset_root:
            raise SystemExit("--dataset-root is required for training runs unless --smoke or --dry-run is set")
        if not args.dataset_root.exists():
            raise SystemExit(f"Dataset root does not exist: {args.dataset_root}")
    if stage in {"eval", "diagnostics", "winoground_eval", "slot_probe", "rop_basis", "flickr_retrieval"}:
        if not args.benchmark_root:
            raise SystemExit("--benchmark-root is required for eval/diagnostics unless --smoke or --dry-run is set")
        if not args.benchmark_root.exists():
            raise SystemExit(f"Benchmark root does not exist: {args.benchmark_root}")
    if stage == "slot_probe" and not args.checkpoint:
        raise SystemExit("--checkpoint is required for slot_probe runs unless --smoke or --dry-run is set")
    if args.protected_basis and not args.protected_basis.exists():
        raise SystemExit(f"Protected basis does not exist: {args.protected_basis}")
    if stage in {"eval", "winoground_eval", "slot_probe", "flickr_retrieval"} and args.checkpoint and not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint does not exist: {args.checkpoint}")


def write_manifest(manifest: Dict[str, object], run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def execute(args: argparse.Namespace) -> int:
    spec = get_run(args.run_id)
    validate_real_inputs(args, spec.stage)

    manifest = run_manifest(args)
    run_dir = args.output_root / (args.output_name or spec.run_id)
    manifest_path = write_manifest(manifest, run_dir)

    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        print(f"Wrote manifest: {manifest_path}")
        return 0

    if spec.stage in {"audit", "filter_audit"}:
        records = smoke_records() if args.smoke else load_jsonl(args.audit_jsonl)
        summary = summarize(records)
        summary_path = run_dir / "audit_summary.json"
        write_summary(summary, summary_path)
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        print(f"Wrote audit summary: {summary_path}")
        return 0

    if args.smoke:
        print(f"Smoke manifest written for {spec.run_id}: {manifest_path}")
        print("Training/evaluation backends are intentionally gated on real dataset paths.")
        return 0

    if spec.stage == "eval":
        from .clip_backend import evaluate_adapter_checkpoint, evaluate_openclip, split_keys

        eval_limit = args.eval_limit_per_key
        if eval_limit is None:
            eval_limit = args.max_examples_per_key
        eval_keys = split_keys(args.eval_keys)
        if args.checkpoint:
            summary = evaluate_adapter_checkpoint(
                checkpoint_path=args.checkpoint,
                dataset_root=args.benchmark_root,
                output_path=run_dir / "eval_summary.json",
                max_examples_per_key=eval_limit,
                model_name=args.model_name,
                pretrained=args.pretrained,
                device=args.device,
                keys=eval_keys,
                write_records=args.write_records,
            )
        else:
            summary = evaluate_openclip(
                dataset_root=args.benchmark_root,
                output_path=run_dir / "eval_summary.json",
                max_examples_per_key=eval_limit,
                model_name=args.model_name,
                pretrained=args.pretrained,
                device=args.device,
                keys=eval_keys,
                write_records=args.write_records,
            )
        print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
        return 0

    if spec.stage == "winoground_eval":
        from .clip_backend import evaluate_winoground

        label = args.variant_mode or ("checkpoint" if args.checkpoint else "openclip")
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)
        summary = evaluate_winoground(
            checkpoint_path=args.checkpoint,
            dataset_root=args.benchmark_root,
            output_path=run_dir / f"{safe_label}_winoground_summary.json",
            max_examples=args.eval_limit_per_key,
            model_name=args.model_name,
            pretrained=args.pretrained,
            device=args.device,
            write_records=args.write_records,
        )
        print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
        return 0

    if spec.stage == "train":
        from .clip_backend import train_rrb, train_tiny_adapter, train_token_bottleneck

        seed = manifest["resolved"]["seed"] or 0
        variant = args.variant_mode or spec.variant
        if args.backend == "rrb":
            train_fn = train_rrb
        elif args.backend == "token_bottleneck":
            train_fn = train_token_bottleneck
        else:
            train_fn = train_tiny_adapter
        summary = train_fn(
            dataset_root=args.dataset_root,
            output_dir=run_dir,
            train_examples=args.max_examples_per_key,
            eval_examples=max(16, min(128, args.max_examples_per_key // 2)),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            model_name=args.model_name,
            pretrained=args.pretrained,
            device=args.device,
            seed=int(seed),
            variant=variant,
            slot_count=manifest["resolved"]["slot_count"],
            lambda_distill=args.lambda_distill,
            lambda_original=args.lambda_original,
            lambda_targeted=args.lambda_targeted,
            **(
                {
                    "lambda_anchor": args.lambda_anchor,
                    "lambda_delta": args.lambda_delta,
                    "residual_alpha": args.residual_alpha,
                    "residual_gate": args.residual_gate,
                    "anchor_ratio": args.anchor_ratio,
                    "protected_basis_path": args.protected_basis,
                    "lambda_rop": args.lambda_rop,
                    "rop_mode": args.rop_mode,
                }
                if args.backend == "rrb"
                else {}
            ),
        )
        if args.full_eval_after_train:
            if not args.benchmark_root:
                raise SystemExit("--benchmark-root is required with --full-eval-after-train")
            from .clip_backend import evaluate_adapter_checkpoint, split_keys

            eval_limit = args.eval_limit_per_key
            if eval_limit is None:
                eval_limit = args.max_examples_per_key
            eval_summary = evaluate_adapter_checkpoint(
                checkpoint_path=run_dir / "adapter.pt",
                dataset_root=args.benchmark_root,
                output_path=run_dir / "checkpoint_eval_summary.json",
                max_examples_per_key=eval_limit,
                model_name=args.model_name,
                pretrained=args.pretrained,
                device=args.device,
                keys=split_keys(args.eval_keys),
                write_records=args.write_records,
            )
            print(json.dumps(eval_summary.__dict__, indent=2, sort_keys=True))
        print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
        return 0

    if spec.stage == "diagnostics":
        from .clip_backend import write_diagnostics

        summary = write_diagnostics(args.checkpoint, run_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if spec.stage == "slot_probe":
        from .clip_backend import slot_responsibility_probe, split_keys

        label = args.variant_mode or args.checkpoint.stem
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)
        summary = slot_responsibility_probe(
            checkpoint_path=args.checkpoint,
            dataset_root=args.benchmark_root,
            output_path=run_dir / f"{safe_label}_slot_probe_summary.json",
            max_examples_per_key=args.eval_limit_per_key or args.max_examples_per_key,
            model_name=args.model_name,
            pretrained=args.pretrained,
            device=args.device,
            keys=split_keys(args.eval_keys),
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if spec.stage == "rop_basis":
        from .clip_backend import write_rop_basis

        summary = write_rop_basis(
            dataset_root=args.benchmark_root,
            output_dir=run_dir,
            max_images=args.max_examples_per_key,
            protected_rank=args.protected_rank,
            batch_size=args.batch_size,
            model_name=args.model_name,
            pretrained=args.pretrained,
            device=args.device,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if spec.stage == "flickr_retrieval":
        from .clip_backend import evaluate_flickr30k_retrieval

        label = args.variant_mode or ("checkpoint" if args.checkpoint else "openclip")
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)
        summary = evaluate_flickr30k_retrieval(
            dataset_root=args.benchmark_root,
            output_path=run_dir / f"{safe_label}_flickr30k_summary.json",
            checkpoint_path=args.checkpoint,
            split=args.flickr_split,
            max_images=args.eval_limit_per_key or args.max_examples_per_key,
            batch_size=args.batch_size,
            model_name=args.model_name,
            pretrained=args.pretrained,
            device=args.device,
        )
        print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
        return 0

    raise SystemExit(f"No backend is implemented for stage {spec.stage!r} in {spec.run_id}")


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        for run in iter_runs():
            print(f"{run.run_id}\t{run.milestone}\t{run.priority}\t{run.stage}\t{run.variant}")
        return 0

    if args.show:
        print(json.dumps(get_run(args.show).to_dict(), indent=2, sort_keys=True))
        return 0

    if not args.run_id:
        parser.error("--run-id is required unless --list or --show is used")

    args.run_id = args.run_id.upper()
    return execute(args)


if __name__ == "__main__":
    sys.exit(main())
