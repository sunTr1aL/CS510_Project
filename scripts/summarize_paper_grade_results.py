"""Summarize expanded RelBottleneck public-suite evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional


RUN_LABELS = {
    "R003": "pretrained OpenCLIP",
    "R005": "plain fine-tune",
    "R006": "bottleneck-only",
    "R007": "intervention-only",
    "R008": "full K=4",
    "R009": "full K=8",
    "R010": "full K=16",
    "R011": "full seed 1",
    "R012": "full seed 2",
    "R013": "full seed 3",
    "R014": "no targeted loss",
    "R015": "generic hard negatives",
    "R016": "unfiltered edits",
    "R017": "overbuilt variant",
    "R019": "RRB alpha=0.05 anchor=5",
    "R020": "RRB alpha=0.10 anchor=5",
    "R021": "RRB alpha=0.10 anchor=10",
    "R022": "RRB learned gate",
    "R023": "RRB no targeted",
    "R024": "RRB generic hard negatives",
    "R025": "RRB replicate",
    "R026": "RRB consolidation",
    "R027": "RRB full seed 3",
    "R028": "RRB no targeted stronger anchor",
    "R029": "RRB full stronger anchor",
    "R030": "RRB final diagnostics",
    "R031": "RRB no targeted stronger anchor seed 2",
    "R032": "RRB no targeted stronger anchor seed 3",
    "R033": "No-slot residual adapter control",
    "R034": "Winoground external validity",
    "R035": "RRB no targeted anchor=20 seed 1",
    "R036": "RRB no targeted anchor=20 seed 2",
    "R037": "RRB no targeted anchor=20 seed 3",
    "R038": "RRB no targeted anchor=20 seed 4",
    "R039": "RRB no targeted anchor=20 seed 5",
    "R040": "Constrained no-slot residual control",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_eval_path(run_dir: Path) -> Optional[Path]:
    for name in ["checkpoint_eval_summary.json", "eval_summary.json"]:
        path = run_dir / name
        if path.exists():
            return path
    return None


def load_results(root: Path) -> Dict[str, dict]:
    results = {}
    for rid in sorted(RUN_LABELS):
        run_dir = root / rid
        eval_path = run_eval_path(run_dir)
        train = load_json(run_dir / "train_summary.json")
        manifest = load_json(run_dir / "manifest.json")
        if not eval_path and not train:
            continue
        eval_summary = load_json(eval_path) if eval_path else None
        guardrail = load_json(root / "guardrails" / rid / "guardrail_summary.json")
        results[rid] = {
            "label": RUN_LABELS[rid],
            "train": train,
            "eval": eval_summary,
            "guardrail": guardrail,
            "manifest": manifest,
            "eval_path": str(eval_path) if eval_path else "",
        }
    return results


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, (float, int)):
        return f"{value:.{digits}f}"
    if value is None:
        return "-"
    return str(value)


def write_main_table(lines: List[str], results: Dict[str, dict]) -> None:
    baseline = results.get("R003", {}).get("eval", {})
    base_acc = baseline.get("accuracy")
    lines.extend(
        [
            "## Main Public-Suite Results",
            "",
            "| Run | System | Eval total | Accuracy | Delta vs R003 | Mean margin | Median margin | Train loss | Train eval |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rid, item in results.items():
        eval_summary = item.get("eval") or {}
        train = item.get("train") or {}
        acc = eval_summary.get("accuracy")
        delta = acc - base_acc if isinstance(acc, (float, int)) and isinstance(base_acc, (float, int)) else None
        lines.append(
            "| {rid} | {label} | {total} | {acc} | {delta} | {mean_margin} | {median_margin} | {loss} | {train_eval} |".format(
                rid=rid,
                label=item["label"],
                total=eval_summary.get("total", "-"),
                acc=fmt(acc),
                delta=fmt(delta),
                mean_margin=fmt(eval_summary.get("mean_margin")),
                median_margin=fmt(eval_summary.get("median_margin")),
                loss=fmt(train.get("final_loss")),
                train_eval=fmt(train.get("eval_accuracy")),
            )
        )
    lines.append("")


def write_source_table(lines: List[str], results: Dict[str, dict]) -> None:
    sources = sorted(
        {
            source
            for item in results.values()
            for source in ((item.get("eval") or {}).get("by_source") or {})
        }
    )
    if not sources:
        return
    selected = [
        rid
        for rid in [
            "R003",
            "R005",
            "R011",
            "R014",
            "R019",
            "R020",
            "R021",
            "R022",
            "R023",
            "R024",
            "R025",
            "R027",
            "R028",
            "R029",
            "R031",
            "R032",
            "R033",
            "R034",
            "R035",
            "R036",
            "R037",
            "R038",
            "R039",
            "R040",
        ]
        if rid in results
    ]
    lines.extend(["## Per-Source Accuracy", ""])
    header = "| Source | " + " | ".join(selected) + " |"
    lines.append(header)
    lines.append("|---|" + "|".join(["---:"] * len(selected)) + "|")
    for source in sources:
        row = [source]
        for rid in selected:
            by_source = (results[rid].get("eval") or {}).get("by_source") or {}
            row.append(fmt((by_source.get(source) or {}).get("accuracy")))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")


def write_seed_summary(lines: List[str], results: Dict[str, dict]) -> None:
    vals = []
    for rid in ["R011", "R012", "R013"]:
        acc = (results.get(rid, {}).get("eval") or {}).get("accuracy")
        if isinstance(acc, (float, int)):
            vals.append(float(acc))
    if not vals:
        return
    lines.extend(
        [
            "## Three-Seed Full-Model Summary",
            "",
            f"- Mean accuracy: {mean(vals):.4f}",
            f"- Population std: {pstdev(vals):.4f}" if len(vals) > 1 else "- Population std: -",
            f"- Min / max: {min(vals):.4f} / {max(vals):.4f}",
            "",
        ]
    )


def write_guardrail_table(lines: List[str], results: Dict[str, dict]) -> None:
    if not any(item.get("guardrail") for item in results.values()):
        return
    lines.extend(
        [
            "## Guardrail Metrics",
            "",
            "| Run | COCO I2T R@1 | COCO I2T R@5 | COCO T2I R@1 | COCO T2I R@5 | ImageNet top-1 | ImageNet top-5 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rid, item in results.items():
        guardrail = item.get("guardrail") or {}
        retrieval = guardrail.get("coco_retrieval") or {}
        imagenet = guardrail.get("imagenet_zeroshot") or {}
        lines.append(
            "| {rid} | {i2t1} | {i2t5} | {t2i1} | {t2i5} | {top1} | {top5} |".format(
                rid=rid,
                i2t1=fmt(retrieval.get("image_to_text_r1")),
                i2t5=fmt(retrieval.get("image_to_text_r5")),
                t2i1=fmt(retrieval.get("text_to_image_r1")),
                t2i5=fmt(retrieval.get("text_to_image_r5")),
                top1=fmt(imagenet.get("top1")),
                top5=fmt(imagenet.get("top5")),
            )
        )
    lines.append("")


def write_rrb_diagnostics(lines: List[str], results: Dict[str, dict]) -> None:
    rrb_rows = []
    for rid, item in results.items():
        train = item.get("train") or {}
        if train.get("backend") != "rrb":
            continue
        rrb_rows.append((rid, item["label"], train))
    if not rrb_rows:
        return
    lines.extend(
        [
            "## RRB Residual Diagnostics",
            "",
            "| Run | System | alpha | gate | anchor ratio | lambda anchor | residual rel. mag. | z0 cosine mean | z0 cosine min | final anchor loss | final delta loss |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rid, label, train in rrb_rows:
        lines.append(
            "| {rid} | {label} | {alpha} | {gate} | {ratio} | {l_anchor} | {relmag} | {cos_mean} | {cos_min} | {anchor_loss} | {delta_loss} |".format(
                rid=rid,
                label=label,
                alpha=fmt(train.get("residual_alpha")),
                gate=train.get("residual_gate", "-"),
                ratio=fmt(train.get("anchor_ratio")),
                l_anchor=fmt(train.get("lambda_anchor")),
                relmag=fmt(train.get("residual_relative_magnitude")),
                cos_mean=fmt(train.get("z0_cosine_mean")),
                cos_min=fmt(train.get("z0_cosine_min")),
                anchor_loss=fmt(train.get("final_anchor_loss")),
                delta_loss=fmt(train.get("final_delta_loss")),
            )
        )
    lines.append("")


def records_by_index(path: Path) -> Dict[int, dict]:
    records = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[int(record["index"])] = record
    return records


def write_pairwise(lines: List[str], root: Path, pairs: Iterable[tuple[str, str]]) -> None:
    lines.extend(["## Pairwise Win/Loss/Tie", ""])
    lines.append("| Pair | Wins | Losses | Ties | Mean margin delta |")
    lines.append("|---|---:|---:|---:|---:|")
    for left, right in pairs:
        left_records = records_by_index(root / left / "eval_records.jsonl")
        right_records = records_by_index(root / right / "eval_records.jsonl")
        keys = sorted(set(left_records) & set(right_records))
        if not keys:
            lines.append(f"| {left} vs {right} | - | - | - | - |")
            continue
        wins = losses = ties = 0
        deltas = []
        for key in keys:
            delta = float(left_records[key]["margin"]) - float(right_records[key]["margin"])
            deltas.append(delta)
            if delta > 1e-6:
                wins += 1
            elif delta < -1e-6:
                losses += 1
            else:
                ties += 1
        lines.append(f"| {left} vs {right} | {wins} | {losses} | {ties} | {mean(deltas):.4f} |")
    lines.append("")


def main() -> int:
    args = parse_args()
    results = load_results(args.root)
    lines: List[str] = [
        "# Expanded RelBottleneck Public-Suite Results",
        "",
        f"Output root: `{args.root}`",
        "",
        "These results evaluate staged public ARO/SugarCrepe suites with COCO retrieval and ImageNet zero-shot guardrails. RRB runs additionally report residual magnitude and z0-to-final drift diagnostics. They are still not a substitute for Winoground access or Flickr30k retrieval.",
        "",
    ]
    write_main_table(lines, results)
    write_source_table(lines, results)
    write_seed_summary(lines, results)
    write_guardrail_table(lines, results)
    write_rrb_diagnostics(lines, results)
    write_pairwise(
        lines,
        args.root,
        [
            ("R011", "R005"),
            ("R019", "R005"),
            ("R020", "R005"),
            ("R021", "R005"),
            ("R022", "R005"),
            ("R023", "R019"),
            ("R024", "R019"),
            ("R027", "R019"),
            ("R028", "R023"),
            ("R029", "R019"),
            ("R035", "R028"),
            ("R036", "R031"),
            ("R037", "R032"),
            ("R040", "R033"),
        ],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
