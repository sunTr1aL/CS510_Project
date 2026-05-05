"""Heuristic checks for structured counterfactual caption edits."""

from __future__ import annotations

import difflib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class EditAuditSummary:
    count: int
    mean_lexical_overlap: float
    mean_edit_locality: float
    filter_keep_rate: float
    margin_min: float
    margin_mean: float
    margin_max: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL record") from exc
    return records


def lexical_overlap(a: str, b: str) -> float:
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def edit_locality(a: str, b: str) -> float:
    matcher = difflib.SequenceMatcher(a=a.split(), b=b.split(), autojunk=False)
    changed = 0
    total = max(len(a.split()), len(b.split()), 1)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            changed += max(i2 - i1, j2 - j1)
    return 1.0 - min(changed / total, 1.0)


def summarize(records: Sequence[Dict[str, object]]) -> EditAuditSummary:
    if not records:
        raise ValueError("No records to audit")

    overlaps: List[float] = []
    localities: List[float] = []
    margins: List[float] = []
    kept = 0

    for idx, record in enumerate(records):
        caption = str(record.get("caption", ""))
        counterfactual = str(record.get("counterfactual", ""))
        if not caption or not counterfactual:
            raise ValueError(f"Record {idx} must contain caption and counterfactual")
        overlaps.append(lexical_overlap(caption, counterfactual))
        localities.append(edit_locality(caption, counterfactual))

        if bool(record.get("filter_passed", False)):
            kept += 1
        if "filter_score_margin" in record:
            margins.append(float(record["filter_score_margin"]))

    if not margins:
        margins = [0.0]

    return EditAuditSummary(
        count=len(records),
        mean_lexical_overlap=sum(overlaps) / len(overlaps),
        mean_edit_locality=sum(localities) / len(localities),
        filter_keep_rate=kept / len(records),
        margin_min=min(margins),
        margin_mean=sum(margins) / len(margins),
        margin_max=max(margins),
    )


def smoke_records() -> List[Dict[str, object]]:
    return [
        {
            "caption": "a red cup sits to the left of a blue bowl",
            "counterfactual": "a red cup sits to the right of a blue bowl",
            "filter_passed": True,
            "filter_score_margin": 0.31,
        },
        {
            "caption": "a dog chases a child through the park",
            "counterfactual": "a child chases a dog through the park",
            "filter_passed": True,
            "filter_score_margin": 0.27,
        },
    ]


def write_summary(summary: EditAuditSummary, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")

