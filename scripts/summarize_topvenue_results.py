"""Summarize R041-R045 top-venue robustness runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, (float, int)):
        return f"{value:.{digits}f}"
    if value is None:
        return "-"
    return str(value)


def write_slot_probe(lines: List[str], root: Path, aggregate: dict) -> None:
    rows = []
    for path in sorted((root / "R041").glob("*_slot_probe_summary.json")):
        item = load_json(path) or {}
        rows.append((path.stem.replace("_slot_probe_summary", ""), item))
    aggregate["R041"] = {label: item for label, item in rows}
    lines.extend(
        [
            "## R041 Slot Responsibility",
            "",
            "| Checkpoint | Examples | Entropy | Top-1 conc. | Edited overlap | Slot-delta lift | Verdict |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    if not rows:
        lines.append("| - | - | - | - | - | - | missing |")
    for label, item in rows:
        lines.append(
            "| {label} | {examples} | {entropy} | {top1} | {overlap} | {lift} | {verdict} |".format(
                label=label,
                examples=item.get("examples"),
                entropy=fmt(item.get("mean_slot_entropy")),
                top1=fmt(item.get("mean_top1_slot_concentration")),
                overlap=fmt(item.get("mean_edited_span_overlap")),
                lift=fmt(item.get("mean_slot_delta_lift")),
                verdict=item.get("verdict", "-"),
            )
        )
    lines.append("")


def write_rop(lines: List[str], root: Path, aggregate: dict) -> None:
    basis = load_json(root / "R042" / "rop_basis_summary.json") or {}
    gate = load_json(root / "R043_gate.json") or {}
    aggregate["R042"] = basis
    aggregate["R043_gate"] = gate
    lines.extend(
        [
            "## R042-R044 ROP Ablation",
            "",
            f"- R042 protected rank: {basis.get('protected_rank', '-')}; cumulative variance: {fmt(basis.get('cumulative_variance_explained'))}",
            f"- R043 gate: {'PASS' if gate.get('passes') else 'FAIL'}; relation {fmt(gate.get('relation_accuracy'))}, COCO I2T R@1 {fmt(gate.get('coco_i2t_r1'))}, ImageNet top-1 {fmt(gate.get('imagenet_top1'))}",
            "",
            "| Run | Relation acc. | COCO I2T R@1 | ImageNet top-1 | Residual mag. | Protected energy | ROP loss |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    rop_runs = {}
    for rid in ["R043", "R044_seed2", "R044_seed3"]:
        train = load_json(root / rid / "train_summary.json") or {}
        eval_summary = load_json(root / rid / "checkpoint_eval_summary.json") or {}
        guardrail = load_json(root / "guardrails" / rid / "guardrail_summary.json") or {}
        retrieval = guardrail.get("coco_retrieval") or {}
        imagenet = guardrail.get("imagenet_zeroshot") or {}
        if not train and not eval_summary and not guardrail:
            continue
        rop_runs[rid] = {"train": train, "eval": eval_summary, "guardrail": guardrail}
        lines.append(
            "| {rid} | {rel} | {coco} | {imagenet} | {resid} | {energy} | {rop_loss} |".format(
                rid=rid,
                rel=fmt(eval_summary.get("accuracy")),
                coco=fmt(retrieval.get("image_to_text_r1")),
                imagenet=fmt(imagenet.get("top1")),
                resid=fmt(train.get("residual_relative_magnitude")),
                energy=fmt(train.get("protected_residual_energy")),
                rop_loss=fmt(train.get("final_rop_loss")),
            )
        )
    if not rop_runs:
        lines.append("| - | - | - | - | - | - | - |")
    aggregate["R043_R044"] = rop_runs
    lines.append("")


def write_flickr(lines: List[str], root: Path, aggregate: dict) -> None:
    rows = []
    for path in sorted((root / "R045").glob("*_flickr30k_summary.json")):
        item = load_json(path) or {}
        rows.append((path.stem.replace("_flickr30k_summary", ""), item))
    aggregate["R045"] = {label: item for label, item in rows}
    lines.extend(
        [
            "## R045 Flickr30k Retrieval",
            "",
            "| System | Images | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if not rows:
        lines.append("| - | - | - | - | - | - | - | - |")
    for label, item in rows:
        lines.append(
            "| {label} | {images} | {i1} | {i5} | {i10} | {t1} | {t5} | {t10} |".format(
                label=label,
                images=item.get("images"),
                i1=fmt(item.get("image_to_text_r1")),
                i5=fmt(item.get("image_to_text_r5")),
                i10=fmt(item.get("image_to_text_r10")),
                t1=fmt(item.get("text_to_image_r1")),
                t5=fmt(item.get("text_to_image_r5")),
                t10=fmt(item.get("text_to_image_r10")),
            )
        )
    lines.append("")


def main() -> int:
    args = parse_args()
    aggregate = {"root": str(args.root)}
    lines: List[str] = [
        "# R041-R045 Initial Results",
        "",
        f"Output root: `{args.root}`",
        "",
    ]
    write_slot_probe(lines, args.root, aggregate)
    write_rop(lines, args.root, aggregate)
    write_flickr(lines, args.root, aggregate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    aggregate_path = args.output.with_suffix(".json")
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
