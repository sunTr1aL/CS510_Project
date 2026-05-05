"""Summarize R034 Winoground outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_summaries(root: Path) -> Dict[str, dict]:
    run_dir = root / "R034"
    summaries = {}
    for path in sorted(run_dir.glob("*_winoground_summary.json")):
        label = path.name.removesuffix("_winoground_summary.json")
        summaries[label] = json.loads(path.read_text(encoding="utf-8"))
    return summaries


def fmt(value: object) -> str:
    if isinstance(value, (float, int)):
        return f"{value:.4f}"
    return "-" if value is None else str(value)


def main() -> int:
    args = parse_args()
    summaries = load_summaries(args.root)
    lines = [
        "# R034 Winoground Results",
        "",
        f"Output root: `{args.root}`",
        "",
        "| System | Scorer | Total | Text score | Image score | Group score |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label, summary in summaries.items():
        lines.append(
            "| {label} | {scorer} | {total} | {text} | {image} | {group} |".format(
                label=label,
                scorer=summary.get("scorer", "-"),
                total=summary.get("total", "-"),
                text=fmt(summary.get("text_score")),
                image=fmt(summary.get("image_score")),
                group=fmt(summary.get("group_score")),
            )
        )
    lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
