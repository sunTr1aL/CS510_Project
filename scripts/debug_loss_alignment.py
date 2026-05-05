"""Inspect CLIP-token span masks and targeted slot-loss wiring.

This diagnostic does not prove that slots are meaningful. It checks the cheaper
failure mode from the review: whether edited/kept spans map to tokenizer
positions cleanly and whether `targeted_slot_loss` has nonzero gradients.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPECIAL_TOKEN_IDS = {0, 49406, 49407}


def parse_indices(value: Optional[str]) -> Optional[List[int]]:
    if not value:
        return None
    indices = []
    for item in value.split(","):
        item = item.strip()
        if item:
            indices.append(int(item))
    return indices


def span_bounds(text: str, span: str) -> Tuple[int, int]:
    start = text.lower().find(span.lower())
    if start < 0:
        raise ValueError(f"Span {span!r} was not found in decoded text {text!r}")
    return start, start + len(span)


def overlaps(left: Tuple[int, int], right: Tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def token_mask_from_ranges(
    ranges: Dict[int, Tuple[int, int, str]],
    token_count: int,
    span: Optional[str] = None,
    indices: Optional[Sequence[int]] = None,
    decoded_text: Optional[str] = None,
) -> List[bool]:
    mask = [False] * token_count
    if indices is not None:
        for index in indices:
            if index < 0 or index >= token_count:
                raise ValueError(f"Token index {index} is outside token count {token_count}")
            mask[index] = True
        return mask
    if not span:
        return mask
    if decoded_text is None:
        raise ValueError("decoded_text is required when using a span")
    target = span_bounds(decoded_text, span)
    for index, char_range in ranges.items():
        if overlaps((char_range[0], char_range[1]), target):
            mask[index] = True
    return mask


def decode_token_ids(tokenizer, token_ids: Sequence[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        return str(decode(list(token_ids)))
    inner = getattr(tokenizer, "tokenizer", None)
    decode = getattr(inner, "decode", None)
    if callable(decode):
        return str(decode(list(token_ids)))
    raise RuntimeError("Tokenizer does not expose decode(); pass explicit token indices instead")


def clean_decoded(text: str) -> str:
    for token in ["<start_of_text>", "<end_of_text>", "<|startoftext|>", "<|endoftext|>"]:
        text = text.replace(token, "")
    return " ".join(text.replace("</w>", " ").split())


def token_char_ranges(tokenizer, token_ids: Sequence[int]) -> Tuple[str, Dict[int, Tuple[int, int, str]]]:
    active: List[Tuple[int, int]] = [
        (index, int(token_id))
        for index, token_id in enumerate(token_ids)
        if int(token_id) not in SPECIAL_TOKEN_IDS
    ]
    decoded_full = clean_decoded(decode_token_ids(tokenizer, [token_id for _index, token_id in active]))
    ranges: Dict[int, Tuple[int, int, str]] = {}
    previous = ""
    for offset, (index, _token_id) in enumerate(active, start=1):
        current = clean_decoded(decode_token_ids(tokenizer, [token_id for _idx, token_id in active[:offset]]))
        token_text = current[len(previous) :].strip()
        start = len(previous)
        end = len(current)
        ranges[index] = (start, end, token_text)
        previous = current
    return decoded_full, ranges


def tokenize(tokenizer, text: str) -> List[int]:
    tokens = tokenizer([text])
    if hasattr(tokens, "detach"):
        tokens = tokens.detach().cpu().tolist()
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    return [int(token) for token in tokens]


def build_rows(
    token_ids: Sequence[int],
    ranges: Dict[int, Tuple[int, int, str]],
    edited_mask: Sequence[bool],
    keep_mask: Sequence[bool],
) -> List[dict]:
    rows = []
    for index, token_id in enumerate(token_ids):
        start, end, text = ranges.get(index, (None, None, ""))
        rows.append(
            {
                "index": index,
                "token_id": int(token_id),
                "text": text,
                "char_start": start,
                "char_end": end,
                "edited": bool(edited_mask[index]),
                "keep": bool(keep_mask[index]),
            }
        )
    return rows


def debug_loss(
    token_count: int,
    edited_mask: Sequence[bool],
    keep_mask: Sequence[bool],
    token_dim: int,
    slot_count: int,
    seed: int,
) -> dict:
    import torch

    from experiments.relbottleneck import RelBottleneckFusion, targeted_slot_loss

    torch.manual_seed(seed)
    model = RelBottleneckFusion(token_dim=token_dim, clip_embed_dim=token_dim, slot_count=slot_count)
    text_tokens = torch.randn(1, token_count, token_dim, requires_grad=True)
    cf_tokens = torch.randn(1, token_count, token_dim, requires_grad=True)
    out_text = model(text_tokens)
    out_cf = model(cf_tokens)
    edited = torch.tensor([edited_mask], dtype=torch.bool)
    keep = torch.tensor([keep_mask], dtype=torch.bool)
    loss = targeted_slot_loss(
        out_text["slots"],
        out_cf["slots"],
        out_text["token_responsibility"],
        edited,
        keep,
    )
    loss.backward()
    responsibility = out_text["token_responsibility"].detach().squeeze(0)
    top_slots = responsibility.argmax(dim=0).cpu().tolist()
    return {
        "loss": float(loss.detach().cpu()),
        "text_token_grad_norm": float(text_tokens.grad.norm().detach().cpu()),
        "cf_token_grad_norm": float(cf_tokens.grad.norm().detach().cpu()),
        "edited_token_count": int(sum(edited_mask)),
        "keep_token_count": int(sum(keep_mask)),
        "top_slot_by_token": [int(item) for item in top_slots],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--counterfactual", required=True)
    parser.add_argument("--edited-span")
    parser.add_argument("--keep-span")
    parser.add_argument("--edited-token-indices")
    parser.add_argument("--keep-token-indices")
    parser.add_argument("--model-name", default="ViT-B-16")
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--slot-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import open_clip

    tokenizer = open_clip.get_tokenizer(args.model_name)
    token_ids = tokenize(tokenizer, args.caption)
    decoded_text, ranges = token_char_ranges(tokenizer, token_ids)
    edited_mask = token_mask_from_ranges(
        ranges,
        len(token_ids),
        span=args.edited_span,
        indices=parse_indices(args.edited_token_indices),
        decoded_text=decoded_text,
    )
    keep_mask = token_mask_from_ranges(
        ranges,
        len(token_ids),
        span=args.keep_span,
        indices=parse_indices(args.keep_token_indices),
        decoded_text=decoded_text,
    )
    result = {
        "caption": args.caption,
        "counterfactual": args.counterfactual,
        "decoded_caption": decoded_text,
        "tokens": build_rows(token_ids, ranges, edited_mask, keep_mask),
        "loss_debug": debug_loss(
            len(token_ids),
            edited_mask,
            keep_mask,
            token_dim=args.token_dim,
            slot_count=args.slot_count,
            seed=args.seed,
        ),
        "note": "Random token features test loss plumbing only; inspect real checkpoints before making slot semantics claims.",
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
