"""Build a structured-edit audit JSONL from staged counterfactual datasets."""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.audit_edits import edit_locality, lexical_overlap
from experiments.clip_backend import iter_pair_examples


def infer_edit_spans(caption: str, counterfactual: str) -> tuple[str, str]:
    caption_words = caption.split()
    cf_words = counterfactual.split()
    matcher = difflib.SequenceMatcher(a=caption_words, b=cf_words, autojunk=False)
    edited: list[str] = []
    keep_blocks: list[list[str]] = []
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            block = caption_words[i1:i2]
            if block:
                keep_blocks.append(block)
        else:
            edited.extend(caption_words[i1:i2])
    keep = max(keep_blocks, key=len) if keep_blocks else []
    return " ".join(edited), " ".join(keep)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--limit-per-key", type=int, default=250)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    keys = [
        "sugarcrepe_replace_rel",
        "sugarcrepe_swap_obj",
        "sugarcrepe_swap_att",
        "aro_visual_relation",
        "aro_coco_order",
    ]
    with args.output.open("w", encoding="utf-8") as handle:
        for example in iter_pair_examples(args.benchmark_root, keys=keys, limit_per_key=args.limit_per_key):
            counterfactual = example.negatives[0]
            edited_span, keep_span = infer_edit_spans(example.positive, counterfactual)
            record = {
                "id": f"{example.source}_{count:06d}",
                "source": example.source,
                "caption": example.positive,
                "counterfactual": counterfactual,
                "edit_type": example.edit_type,
                "edited_span": edited_span,
                "keep_span": keep_span,
                "edited_span_indices": [],
                "kept_entity_span_indices": [],
                "lexical_overlap": lexical_overlap(example.positive, counterfactual),
                "edit_locality": edit_locality(example.positive, counterfactual),
                "filter_passed": True,
                "filter_score_margin": 0.0,
                "filter_note": "constructed from benchmark positive/counterfactual caption pairs",
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
            if count >= args.max_records:
                break
    print(f"Wrote {count} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
