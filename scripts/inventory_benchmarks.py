"""Inventory staged benchmark directories for experiment readiness."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List


EXPECTED = {
    "relation": [
        "aro_coco_order",
        "aro_visual_attribution",
        "aro_visual_relation",
        "sugarcrepe_replace_att",
        "sugarcrepe_replace_obj",
        "sugarcrepe_replace_rel",
        "sugarcrepe_swap_att",
        "sugarcrepe_swap_obj",
    ],
    "compositional": ["winoground"],
    "guardrail": ["flickr30k", "coco_captions_validation", "imagenet1k_validation"],
    "reasoning": ["folio"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def inspect_key(root: Path, key: str, group: str) -> Dict[str, object]:
    path = root / key
    parquet_files = sorted(glob.glob(str(path / "**" / "*.parquet"), recursive=True))
    json_files = sorted(glob.glob(str(path / "**" / "*.json"), recursive=True))
    jsonl_files = sorted(glob.glob(str(path / "**" / "*.jsonl"), recursive=True))
    arrow_files = sorted(glob.glob(str(path / "**" / "*.arrow"), recursive=True))
    csv_files = sorted(glob.glob(str(path / "**" / "*.csv"), recursive=True))
    readmes = sorted(glob.glob(str(path / "**" / "README.md"), recursive=True))
    data_files = [*parquet_files, *json_files, *jsonl_files, *arrow_files, *csv_files]
    return {
        "key": key,
        "group": group,
        "path": str(path),
        "exists": path.exists(),
        "parquet_files": len(parquet_files),
        "json_files": len(json_files),
        "jsonl_files": len(jsonl_files),
        "arrow_files": len(arrow_files),
        "csv_files": len(csv_files),
        "readme_files": len(readmes),
        "ready": path.exists() and bool(data_files),
        "sample_files": [str(item) for item in [*data_files[:4], *readmes[:1]]],
    }


def main() -> int:
    args = parse_args()
    rows: List[Dict[str, object]] = []
    for group, keys in EXPECTED.items():
        for key in keys:
            rows.append(inspect_key(args.root, key, group))
    payload = {"root": str(args.root), "datasets": rows}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
