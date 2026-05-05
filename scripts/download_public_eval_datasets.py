"""Download public evaluation datasets used by the RelBottleneck plan."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

PUBLIC_RELATION_DATASETS: Dict[str, str] = {
    "aro_visual_relation": "mteb/ARO-Visual-Relation",
    "aro_visual_attribution": "mteb/ARO-Visual-Attribution",
    "aro_coco_order": "mteb/ARO-COCO-order",
    "sugarcrepe_replace_obj": "HuggingFaceM4/SugarCrepe_replace_obj",
    "sugarcrepe_replace_att": "HuggingFaceM4/SugarCrepe_replace_att",
    "sugarcrepe_replace_rel": "HuggingFaceM4/SugarCrepe_replace_rel",
    "sugarcrepe_swap_obj": "HuggingFaceM4/SugarCrepe_swap_obj",
    "sugarcrepe_swap_att": "HuggingFaceM4/SugarCrepe_swap_att",
}

GUARDRAIL_DATASETS: Dict[str, str] = {
    "flickr30k": "nlphuji/flickr30k",
    "coco_captions_validation": "Multimodal-Fatima/COCO_captions_validation",
    "imagenet1k_validation": "Tsomaros/Imagenet-1k_validation",
}

REASONING_DATASETS: Dict[str, str] = {
    "folio": "tasksource/folio",
}

GATED_OR_AUTH_DATASETS: Dict[str, str] = {
    "winoground": "facebook/winoground",
}


def all_dataset_items(
    include_guardrails: bool,
    include_reasoning: bool,
    include_gated: bool,
) -> List[Tuple[str, str, str]]:
    items: List[Tuple[str, str, str]] = [
        (key, repo_id, "relation") for key, repo_id in sorted(PUBLIC_RELATION_DATASETS.items())
    ]
    if include_guardrails:
        items.extend((key, repo_id, "guardrail") for key, repo_id in sorted(GUARDRAIL_DATASETS.items()))
    if include_reasoning:
        items.extend((key, repo_id, "reasoning") for key, repo_id in sorted(REASONING_DATASETS.items()))
    if include_gated:
        items.extend((key, repo_id, "gated_or_auth") for key, repo_id in sorted(GATED_OR_AUTH_DATASETS.items()))
    return items


def has_materialized_data(path: Path) -> bool:
    patterns = ["**/*.parquet", "**/*.json", "**/*.jsonl", "**/*.arrow"]
    return any(glob.glob(str(path / pattern), recursive=True) for pattern in patterns)


def materialize_with_datasets(repo_id: str, target: Path) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for script-based dataset materialization") from exc

    dataset = load_dataset(
        repo_id,
        cache_dir=str(target / ".datasets_cache"),
        token=os.environ.get("HF_TOKEN"),
        trust_remote_code=True,
    )
    dataset.save_to_disk(str(target / "dataset_cache"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Destination root")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-guardrails", action="store_true", help="Download Flickr/COCO/ImageNet guardrail datasets")
    parser.add_argument("--include-reasoning", action="store_true", help="Download FOLIO text-reasoning benchmark")
    parser.add_argument("--include-gated", action="store_true", help="Attempt gated/authenticated datasets such as Winoground")
    parser.add_argument(
        "--only",
        nargs="*",
        help="Subset of dataset keys to download",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    items = all_dataset_items(
        include_guardrails=args.include_guardrails,
        include_reasoning=args.include_reasoning,
        include_gated=args.include_gated,
    )
    valid = {key for key, _repo_id, _group in items}
    if args.only:
        unknown = sorted(set(args.only) - valid)
        if unknown:
            raise SystemExit(f"Unknown dataset keys for current include flags: {', '.join(unknown)}")
        items = [item for item in items if item[0] in set(args.only)]
    manifest = []

    for key, repo_id, group in items:
        target = args.root / key
        manifest.append({"key": key, "repo_id": repo_id, "group": group, "path": str(target)})
        if args.dry_run:
            continue
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise SystemExit(
                "huggingface_hub is required for downloads. "
                "Install requirements-experiment.txt first."
            ) from exc
        target.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(target),
                local_dir_use_symlinks=False,
                token=os.environ.get("HF_TOKEN"),
            )
        except Exception as exc:
            if group != "gated_or_auth":
                raise
            manifest[-1]["status"] = "failed"
            manifest[-1]["error"] = str(exc)
            continue
        if not has_materialized_data(target):
            try:
                materialize_with_datasets(repo_id, target)
                manifest[-1]["materialized_with_datasets"] = True
            except Exception as exc:
                manifest[-1]["materialized_with_datasets"] = False
                manifest[-1]["materialization_error"] = str(exc)
        manifest[-1]["status"] = "downloaded"

    args.root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.root / "eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gated_path = args.root / "gated_or_auth_manifest.json"
    gated = [
        {"key": key, "repo_id": repo_id, "reason": "requires authenticated access"}
        for key, repo_id in sorted(GATED_OR_AUTH_DATASETS.items())
        if not any(item["key"] == key and item.get("status") == "downloaded" for item in manifest)
    ]
    gated_path.write_text(json.dumps(gated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote gated manifest: {gated_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
