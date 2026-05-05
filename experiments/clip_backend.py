"""OpenCLIP evaluation and tiny training backend for staged parquet datasets."""

from __future__ import annotations

import ast
import csv
import glob
import difflib
import json
import math
import random
import zipfile
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from statistics import median
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

try:
    from torch import nn as _torch_nn
except ImportError:  # pragma: no cover - training paths require torch.
    _torch_nn = None


SUGARCREPE_KEYS = [
    "sugarcrepe_replace_rel",
    "sugarcrepe_replace_att",
    "sugarcrepe_replace_obj",
    "sugarcrepe_swap_att",
    "sugarcrepe_swap_obj",
]

ARO_KEYS = [
    "aro_visual_relation",
    "aro_visual_attribution",
    "aro_coco_order",
]


@dataclass
class PairExample:
    source: str
    image: object
    positive: str
    negatives: List[str]
    edit_type: str


@dataclass
class EvalSummary:
    model_name: str
    pretrained: str
    device: str
    scorer: str
    total: int
    correct: int
    accuracy: float
    mean_margin: float
    median_margin: float
    by_source: Dict[str, Dict[str, float]]
    records_path: Optional[str] = None


@dataclass
class RetrievalSummary:
    scorer: str
    model_name: str
    pretrained: str
    device: str
    images: int
    captions: int
    image_to_text_r1: float
    image_to_text_r5: float
    image_to_text_r10: float
    text_to_image_r1: float
    text_to_image_r5: float
    text_to_image_r10: float


@dataclass
class ZeroShotSummary:
    scorer: str
    model_name: str
    pretrained: str
    device: str
    total: int
    top1: float
    top5: float


@dataclass
class WinogroundSummary:
    scorer: str
    model_name: str
    pretrained: str
    device: str
    total: int
    text_correct: int
    image_correct: int
    group_correct: int
    text_score: float
    image_score: float
    group_score: float
    by_tag: Dict[str, Dict[str, float]]
    by_collapsed_tag: Dict[str, Dict[str, float]]
    records_path: Optional[str] = None


@dataclass
class TrainSummary:
    backend: str
    variant: str
    model_name: str
    pretrained: str
    device: str
    train_examples: int
    epochs: int
    final_loss: float
    eval_accuracy: float
    eval_total: int
    eval_correct: int
    eval_mean_margin: float
    checkpoint: str
    loss_mode: str
    lambda_distill: float = 0.0
    lambda_original: float = 0.0
    lambda_targeted: float = 0.0
    lambda_anchor: float = 0.0
    lambda_delta: float = 0.0
    residual_alpha: float = 0.0
    residual_gate: str = ""
    anchor_ratio: float = 0.0
    final_task_loss: float = 0.0
    final_distill_loss: float = 0.0
    final_original_loss: float = 0.0
    final_targeted_loss: float = 0.0
    final_anchor_loss: float = 0.0
    final_delta_loss: float = 0.0
    residual_relative_magnitude: float = 0.0
    z0_cosine_mean: float = 0.0
    z0_cosine_min: float = 0.0
    lambda_rop: float = 0.0
    rop_mode: str = "none"
    protected_rank: int = 0
    final_rop_loss: float = 0.0
    protected_residual_energy: float = 0.0


def _require_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required to read staged parquet datasets") from exc
    return pq


def _require_pil():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for image loading") from exc
    return Image


def load_torch_checkpoint(torch_module, checkpoint_path: Path, map_location):
    try:
        return torch_module.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch.
        return torch_module.load(checkpoint_path, map_location=map_location)


def _decode_token_ids(tokenizer, token_ids: Sequence[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        return str(decode(list(token_ids)))
    inner = getattr(tokenizer, "tokenizer", None)
    decode = getattr(inner, "decode", None)
    if callable(decode):
        return str(decode(list(token_ids)))
    raise RuntimeError("Tokenizer does not expose decode(); explicit token masks are required")


def _clean_decoded(text: str) -> str:
    for token in ["<start_of_text>", "<end_of_text>", "<|startoftext|>", "<|endoftext|>"]:
        text = text.replace(token, "")
    return " ".join(text.replace("</w>", " ").split())


def _infer_spans(caption: str, counterfactual: str) -> Tuple[str, str]:
    caption_words = caption.split()
    cf_words = counterfactual.split()
    matcher = difflib.SequenceMatcher(a=caption_words, b=cf_words, autojunk=False)
    edited: List[str] = []
    keep_blocks: List[List[str]] = []
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            block = caption_words[i1:i2]
            if block:
                keep_blocks.append(block)
        else:
            edited.extend(caption_words[i1:i2])
    keep = max(keep_blocks, key=len) if keep_blocks else []
    return " ".join(edited), " ".join(keep)


def _span_token_mask(tokenizer, token_ids: Sequence[int], span: str):
    import torch

    mask = torch.zeros(len(token_ids), dtype=torch.bool)
    if not span:
        return mask
    active = [
        (index, int(token_id))
        for index, token_id in enumerate(token_ids)
        if int(token_id) not in {0, 49406, 49407}
    ]
    decoded_full = _clean_decoded(_decode_token_ids(tokenizer, [token_id for _index, token_id in active]))
    start = decoded_full.lower().find(span.lower())
    if start < 0:
        return mask
    target = (start, start + len(span))
    previous = ""
    for offset, (index, _token_id) in enumerate(active, start=1):
        current = _clean_decoded(_decode_token_ids(tokenizer, [token_id for _idx, token_id in active[:offset]]))
        bounds = (len(previous), len(current))
        if bounds[0] < target[1] and target[0] < bounds[1]:
            mask[index] = True
        previous = current
    return mask


def parquet_rows(dataset_root: Path, key: str, limit: Optional[int] = None) -> Iterator[dict]:
    pq = _require_parquet()
    files = sorted(glob.glob(str(dataset_root / key / "data" / "*.parquet")))
    seen = 0
    for file_path in files:
        table = pq.read_table(file_path)
        for row in table.to_pylist():
            yield row
            seen += 1
            if limit is not None and seen >= limit:
                return


def image_from_struct(value: object):
    Image = _require_pil()
    if not isinstance(value, dict):
        raise ValueError("Row does not contain an image/images struct")
    if value.get("bytes") is not None:
        return Image.open(BytesIO(value["bytes"])).convert("RGB")
    if value.get("path"):
        return Image.open(value["path"]).convert("RGB")
    raise ValueError("Image struct does not contain bytes or path")


def row_image(row: dict):
    return image_from_struct(row.get("image", row.get("images")))


def iter_winoground_rows(
    dataset_root: Path,
    limit: Optional[int] = None,
) -> Iterator[dict]:
    root = dataset_root / "winoground" if (dataset_root / "winoground").exists() else dataset_root
    files = sorted(glob.glob(str(root / "data" / "*.parquet")))
    if not files:
        raise ValueError(f"No Winoground parquet files found under {root}")
    seen = 0
    pq = _require_parquet()
    for file_path in files:
        table = pq.read_table(file_path)
        for row in table.to_pylist():
            yield row
            seen += 1
            if limit is not None and seen >= limit:
                return


def iter_coco_retrieval_rows(
    dataset_root: Path,
    limit: Optional[int] = None,
    key: str = "coco_captions_validation",
) -> Iterator[Tuple[object, List[str]]]:
    for row in parquet_rows(dataset_root, key, limit=limit):
        captions = [str(item) for item in row.get("sentences_raw", []) if str(item).strip()]
        if captions:
            yield row_image(row), captions


def iter_flickr30k_retrieval_rows(
    dataset_root: Path,
    limit: Optional[int] = None,
    split: str = "test",
) -> Iterator[Tuple[object, List[str]]]:
    Image = _require_pil()
    root = dataset_root / "flickr30k" if (dataset_root / "flickr30k").exists() else dataset_root
    csv_path = root / "flickr_annotations_30k.csv"
    zip_path = root / "flickr30k-images.zip"
    if not csv_path.exists():
        raise ValueError(f"No Flickr30k annotation CSV found under {root}")
    if not zip_path.exists():
        raise ValueError(f"No Flickr30k image zip found under {root}")

    seen = 0
    with zipfile.ZipFile(zip_path) as images_zip, csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_split = str(row.get("split") or "").strip().lower()
            if split and row_split != split.lower():
                continue
            filename = str(row.get("filename") or "").strip()
            if not filename:
                continue
            captions = [str(item) for item in ast.literal_eval(row["raw"]) if str(item).strip()]
            if not captions:
                continue
            member = f"flickr30k-images/{filename}"
            try:
                with images_zip.open(member) as image_handle:
                    image = Image.open(BytesIO(image_handle.read())).convert("RGB")
            except KeyError:
                continue
            yield image, captions
            seen += 1
            if limit is not None and seen >= limit:
                return


def iter_imagenet_rows(
    dataset_root: Path,
    limit: Optional[int] = None,
    key: str = "imagenet1k_validation",
) -> Iterator[Tuple[object, int]]:
    for row in parquet_rows(dataset_root, key, limit=limit):
        yield row_image(row), int(row["label"])


def iter_pair_examples(
    dataset_root: Path,
    keys: Optional[Sequence[str]] = None,
    limit_per_key: Optional[int] = None,
) -> Iterator[PairExample]:
    keys = list(keys or [*SUGARCREPE_KEYS, *ARO_KEYS])
    for key in keys:
        if key.startswith("sugarcrepe_"):
            for row in parquet_rows(dataset_root, key, limit=limit_per_key):
                labels = list(row["tested_labels"])
                if len(labels) < 2:
                    continue
                yield PairExample(
                    source=key,
                    image=row_image(row),
                    positive=labels[0],
                    negatives=[labels[1]],
                    edit_type=key.replace("sugarcrepe_", ""),
                )
        elif key in {"aro_visual_relation", "aro_visual_attribution"}:
            for row in parquet_rows(dataset_root, key, limit=limit_per_key):
                yield PairExample(
                    source=key,
                    image=row_image(row),
                    positive=row["true_caption"],
                    negatives=[row["false_caption"]],
                    edit_type=row.get("relation_name", key),
                )
        elif key == "aro_coco_order":
            for row in parquet_rows(dataset_root, key, limit=limit_per_key):
                negatives = [
                    row[name]
                    for name in ["hard_text_1", "hard_text_2", "hard_text_3", "hard_text_4"]
                    if row.get(name)
                ]
                yield PairExample(
                    source=key,
                    image=row_image(row),
                    positive=row["correct_caption"],
                    negatives=negatives,
                    edit_type="word_order",
                )


def split_keys(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    keys = [item.strip() for item in value.split(",") if item.strip()]
    return keys or None


def score_examples(
    examples: Sequence[PairExample],
    score_fn,
    records_path: Optional[Path] = None,
) -> Tuple[int, int, List[float], Dict[str, Dict[str, float]]]:
    total = 0
    correct = 0
    margins: List[float] = []
    source_counts: Dict[str, List[float]] = {}
    records_handle = None
    if records_path:
        records_path.parent.mkdir(parents=True, exist_ok=True)
        records_handle = records_path.open("w", encoding="utf-8")

    try:
        for index, example in enumerate(examples):
            captions = [example.positive, *example.negatives]
            scores = score_fn(example.image, captions)
            pred_idx = max(range(len(scores)), key=lambda idx: scores[idx])
            best_negative = max(scores[1:]) if len(scores) > 1 else -math.inf
            margin = float(scores[0] - best_negative)
            ok = int(pred_idx == 0)
            total += 1
            correct += ok
            margins.append(margin)
            source_counts.setdefault(example.source, [0.0, 0.0, 0.0])
            source_counts[example.source][0] += ok
            source_counts[example.source][1] += 1
            source_counts[example.source][2] += margin
            if records_handle:
                records_handle.write(
                    json.dumps(
                        {
                            "index": index,
                            "source": example.source,
                            "edit_type": example.edit_type,
                            "correct": bool(ok),
                            "predicted_index": pred_idx,
                            "positive_score": float(scores[0]),
                            "best_negative_score": float(best_negative),
                            "margin": margin,
                            "num_candidates": len(captions),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    finally:
        if records_handle:
            records_handle.close()

    by_source = {
        source: {
            "correct": vals[0],
            "total": vals[1],
            "accuracy": vals[0] / vals[1],
            "mean_margin": vals[2] / vals[1],
        }
        for source, vals in sorted(source_counts.items())
        if vals[1]
    }
    return total, correct, margins, by_source


def winoground_decisions(a: float, b: float, c: float, d: float) -> Tuple[bool, bool, bool]:
    """Return standard Winoground text, image, and group decisions.

    Scores follow the notation from the dataset card:
    a=s(caption_0, image_0), b=s(caption_1, image_0),
    c=s(caption_1, image_1), d=s(caption_0, image_1).
    """

    text_ok = a > b and c > d
    image_ok = a > d and c > b
    return text_ok, image_ok, text_ok and image_ok


def _add_winoground_bucket(
    buckets: Dict[str, List[int]],
    key: str,
    text_ok: bool,
    image_ok: bool,
    group_ok: bool,
) -> None:
    values = buckets.setdefault(key or "unknown", [0, 0, 0, 0])
    values[0] += 1
    values[1] += int(text_ok)
    values[2] += int(image_ok)
    values[3] += int(group_ok)


def _finalize_winoground_buckets(buckets: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
    final = {}
    for key, values in sorted(buckets.items()):
        total, text_correct, image_correct, group_correct = values
        final[key] = {
            "total": total,
            "text_score": text_correct / total if total else math.nan,
            "image_score": image_correct / total if total else math.nan,
            "group_score": group_correct / total if total else math.nan,
        }
    return final


def numeric_histogram(values: Sequence[float], bins: int = 20) -> Dict[str, object]:
    if not values:
        return {"count": 0, "bins": []}
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return {
            "count": len(values),
            "min": lo,
            "max": hi,
            "bins": [{"left": lo, "right": hi, "count": len(values)}],
        }
    width = (hi - lo) / bins
    counts = [0 for _ in range(bins)]
    for value in values:
        idx = min(bins - 1, int((value - lo) / width))
        counts[idx] += 1
    return {
        "count": len(values),
        "min": lo,
        "max": hi,
        "bins": [
            {"left": lo + idx * width, "right": lo + (idx + 1) * width, "count": count}
            for idx, count in enumerate(counts)
        ],
    }


class OpenCLIPScorer:
    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
        device: str = "cuda",
    ) -> None:
        import open_clip
        import torch

        self.torch = torch
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.model_name = model_name
        self.pretrained = pretrained

    def image_features(self, images: Sequence[object]):
        torch = self.torch
        batch = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        with torch.no_grad():
            return self.model.encode_image(batch, normalize=True)

    def image_token_features(self, images: Sequence[object]):
        import torch.nn.functional as F

        torch = self.torch
        visual = self.model.visual
        if not all(hasattr(visual, name) for name in ["_embeds", "transformer", "_pool"]):
            raise RuntimeError("Token extraction currently supports OpenCLIP VisionTransformer backbones")
        batch = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        with torch.no_grad():
            x = visual._embeds(batch)
            x = visual.transformer(x)
            pooled, tokens = visual._pool(x)
            embedding = pooled @ visual.proj if visual.proj is not None else pooled
            embedding = F.normalize(embedding, dim=-1)
        return tokens, pooled, embedding

    def text_features(self, texts: Sequence[str]):
        torch = self.torch
        tokens = self.tokenizer(list(texts)).to(self.device)
        with torch.no_grad():
            return self.model.encode_text(tokens, normalize=True)

    def text_token_features(self, texts: Sequence[str]):
        tokens, pooled, embedding, _token_ids = self.text_token_features_with_ids(texts)
        return tokens, pooled, embedding

    def text_token_features_with_ids(self, texts: Sequence[str]):
        import torch.nn.functional as F

        from open_clip.model import text_global_pool

        torch = self.torch
        token_ids = self.tokenizer(list(texts)).to(self.device)
        with torch.no_grad():
            cast_dtype = self.model.transformer.get_cast_dtype()
            x = self.model.token_embedding(token_ids).to(cast_dtype)
            x = x + self.model.positional_embedding.to(cast_dtype)
            x = self.model.transformer(x, attn_mask=self.model.attn_mask)
            tokens = self.model.ln_final(x)
            pooled = text_global_pool(
                tokens,
                token_ids,
                self.model.text_pool_type,
                eos_token_id=getattr(self.model, "text_eos_id", None),
            )
            projection = self.model.text_projection
            if projection is not None:
                if hasattr(projection, "weight"):
                    embedding = projection(pooled)
                else:
                    embedding = pooled @ projection
            else:
                embedding = pooled
            embedding = F.normalize(embedding, dim=-1)
        return tokens, pooled, embedding, token_ids

    def score_candidates(self, image, captions: Sequence[str]) -> List[float]:
        image_features = self.image_features([image])
        text_features = self.text_features(captions)
        scores = (image_features @ text_features.T).squeeze(0)
        return [float(x) for x in scores.detach().cpu()]


class AdapterCheckpointScorer:
    def __init__(
        self,
        checkpoint_path: Path,
        model_name: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
        device: str = "cuda",
    ) -> None:
        import torch

        self.base = OpenCLIPScorer(model_name=model_name, pretrained=pretrained, device=device)
        self.torch = torch
        payload = load_torch_checkpoint(torch, checkpoint_path, map_location=self.base.device)
        dim = int(payload.get("dim") or payload["adapter_state_dict"]["0.weight"].shape[1])
        variant = str(payload.get("variant", "full_relbottleneck"))
        slot_count = payload.get("slot_count")
        self.adapter = EmbeddingAdapter(dim, self.base.device, variant=variant, slot_count=slot_count)
        self.adapter.module.load_state_dict(payload["adapter_state_dict"])
        self.adapter.module.eval()
        self.checkpoint_path = checkpoint_path
        self.variant = variant
        self.slot_count = slot_count

    @property
    def device(self):
        return self.base.device

    def score_candidates(self, image, captions: Sequence[str]) -> List[float]:
        with self.torch.no_grad():
            image_features = self.adapter(self.base.image_features([image]))
            text_features = self.adapter(self.base.text_features(captions))
            scores = (image_features @ text_features.T).squeeze(0)
        return [float(x) for x in scores.detach().cpu()]

    def image_features(self, images: Sequence[object]):
        with self.torch.no_grad():
            return self.adapter(self.base.image_features(images))

    def text_features(self, texts: Sequence[str]):
        with self.torch.no_grad():
            return self.adapter(self.base.text_features(texts))


class TokenRelBottleneckCheckpointScorer:
    def __init__(
        self,
        checkpoint_path: Path,
        model_name: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
        device: str = "cuda",
    ) -> None:
        import torch

        self.base = OpenCLIPScorer(model_name=model_name, pretrained=pretrained, device=device)
        self.torch = torch
        payload = load_torch_checkpoint(torch, checkpoint_path, map_location=self.base.device)
        self.adapter = TokenRelBottleneckAdapter(
            image_token_dim=int(payload["image_token_dim"]),
            text_token_dim=int(payload["text_token_dim"]),
            embed_dim=int(payload["dim"]),
            device=self.base.device,
            slot_count=payload.get("slot_count"),
        )
        self.adapter.module.load_state_dict(payload["model_state_dict"])
        self.adapter.module.eval()
        self.checkpoint_path = checkpoint_path
        self.variant = str(payload.get("variant", "full_relbottleneck"))
        self.slot_count = payload.get("slot_count")

    @property
    def device(self):
        return self.base.device

    def score_candidates(self, image, captions: Sequence[str]) -> List[float]:
        with self.torch.no_grad():
            image_features = self.image_features([image])
            text_features = self.text_features(captions)
            scores = (image_features @ text_features.T).squeeze(0)
        return [float(x) for x in scores.detach().cpu()]

    def image_features(self, images: Sequence[object]):
        with self.torch.no_grad():
            tokens, global_token, _base = self.base.image_token_features(images)
            return self.adapter.image_features(tokens, global_token)

    def text_features(self, texts: Sequence[str]):
        with self.torch.no_grad():
            tokens, global_token, _base = self.base.text_token_features(texts)
            return self.adapter.text_features(tokens, global_token)


class RelResidualBottleneckCheckpointScorer:
    def __init__(
        self,
        checkpoint_path: Path,
        model_name: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
        device: str = "cuda",
    ) -> None:
        import torch

        self.base = OpenCLIPScorer(model_name=model_name, pretrained=pretrained, device=device)
        self.torch = torch
        payload = load_torch_checkpoint(torch, checkpoint_path, map_location=self.base.device)
        variant = str(payload.get("variant", "rrb_full"))
        adapter_cls = NoSlotResidualAdapter if variant == "no_slot_residual_adapter" else RelResidualBottleneckAdapter
        adapter_kwargs = {
            "image_token_dim": int(payload["image_token_dim"]),
            "text_token_dim": int(payload["text_token_dim"]),
            "embed_dim": int(payload["dim"]),
            "device": self.base.device,
            "residual_alpha": float(payload.get("residual_alpha", 0.05)),
            "residual_gate": str(payload.get("residual_gate", "fixed")),
            "protected_basis": payload.get("protected_basis"),
            "rop_mode": str(payload.get("rop_mode", "none")),
        }
        if adapter_cls is RelResidualBottleneckAdapter:
            adapter_kwargs["slot_count"] = payload.get("slot_count")
        self.adapter = adapter_cls(**adapter_kwargs)
        self.adapter.module.load_state_dict(payload["model_state_dict"])
        self.adapter.module.eval()
        self.checkpoint_path = checkpoint_path
        self.variant = variant
        self.slot_count = payload.get("slot_count")

    @property
    def device(self):
        return self.base.device

    def score_candidates(self, image, captions: Sequence[str]) -> List[float]:
        with self.torch.no_grad():
            image_features = self.image_features([image])
            text_features = self.text_features(captions)
            scores = (image_features @ text_features.T).squeeze(0)
        return [float(x) for x in scores.detach().cpu()]

    def image_features(self, images: Sequence[object]):
        with self.torch.no_grad():
            tokens, _global_token, base = self.base.image_token_features(images)
            return self.adapter.image_features(tokens, base)

    def text_features(self, texts: Sequence[str]):
        with self.torch.no_grad():
            tokens, _global_token, base = self.base.text_token_features(texts)
            return self.adapter.text_features(tokens, base)


def build_checkpoint_scorer(
    checkpoint_path: Path,
    model_name: str,
    pretrained: str,
    device: str,
):
    import torch

    payload = load_torch_checkpoint(torch, checkpoint_path, map_location="cpu")
    if payload.get("backend") == "rrb":
        return RelResidualBottleneckCheckpointScorer(
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            pretrained=pretrained,
            device=device,
        )
    if payload.get("backend") == "token_bottleneck":
        return TokenRelBottleneckCheckpointScorer(
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            pretrained=pretrained,
            device=device,
        )
    return AdapterCheckpointScorer(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )


def build_scorer(
    checkpoint_path: Optional[Path],
    model_name: str,
    pretrained: str,
    device: str,
):
    if checkpoint_path:
        return build_checkpoint_scorer(
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            pretrained=pretrained,
            device=device,
        )
    return OpenCLIPScorer(model_name=model_name, pretrained=pretrained, device=device)


def scorer_name(scorer) -> str:
    if isinstance(scorer, AdapterCheckpointScorer):
        return f"adapter:{scorer.variant}"
    if isinstance(scorer, TokenRelBottleneckCheckpointScorer):
        return f"token_bottleneck:{scorer.variant}"
    if isinstance(scorer, RelResidualBottleneckCheckpointScorer):
        return f"rrb:{scorer.variant}"
    return "openclip"


def batched(seq: Sequence[object], batch_size: int) -> Iterator[Sequence[object]]:
    for start in range(0, len(seq), batch_size):
        yield seq[start : start + batch_size]


def evaluate_openclip(
    dataset_root: Path,
    output_path: Path,
    max_examples_per_key: Optional[int] = 256,
    model_name: str = "ViT-B-16",
    pretrained: str = "laion2b_s34b_b88k",
    device: str = "cuda",
    keys: Optional[Sequence[str]] = None,
    write_records: bool = False,
) -> EvalSummary:
    scorer = OpenCLIPScorer(model_name=model_name, pretrained=pretrained, device=device)
    examples = list(iter_pair_examples(dataset_root, keys=keys, limit_per_key=max_examples_per_key))
    records_path = output_path.with_name("eval_records.jsonl") if write_records else None
    total, correct, margins, by_source = score_examples(
        examples,
        scorer.score_candidates,
        records_path=records_path,
    )
    summary = EvalSummary(
        model_name=model_name,
        pretrained=pretrained,
        device=str(scorer.device),
        scorer="openclip",
        total=total,
        correct=correct,
        accuracy=correct / total if total else math.nan,
        mean_margin=sum(margins) / len(margins) if margins else math.nan,
        median_margin=median(margins) if margins else math.nan,
        by_source=by_source,
        records_path=str(records_path) if records_path else None,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def evaluate_adapter_checkpoint(
    checkpoint_path: Path,
    dataset_root: Path,
    output_path: Path,
    max_examples_per_key: Optional[int] = 256,
    model_name: str = "ViT-B-16",
    pretrained: str = "laion2b_s34b_b88k",
    device: str = "cuda",
    keys: Optional[Sequence[str]] = None,
    write_records: bool = True,
) -> EvalSummary:
    scorer = build_checkpoint_scorer(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )
    examples = list(iter_pair_examples(dataset_root, keys=keys, limit_per_key=max_examples_per_key))
    records_path = output_path.with_name("eval_records.jsonl") if write_records else None
    total, correct, margins, by_source = score_examples(
        examples,
        scorer.score_candidates,
        records_path=records_path,
    )
    summary = EvalSummary(
        model_name=model_name,
        pretrained=pretrained,
        device=str(scorer.device),
        scorer=scorer_name(scorer),
        total=total,
        correct=correct,
        accuracy=correct / total if total else math.nan,
        mean_margin=sum(margins) / len(margins) if margins else math.nan,
        median_margin=median(margins) if margins else math.nan,
        by_source=by_source,
        records_path=str(records_path) if records_path else None,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def evaluate_winoground(
    dataset_root: Path,
    output_path: Path,
    checkpoint_path: Optional[Path] = None,
    max_examples: Optional[int] = None,
    model_name: str = "ViT-B-16",
    pretrained: str = "laion2b_s34b_b88k",
    device: str = "cuda",
    write_records: bool = True,
) -> WinogroundSummary:
    scorer = build_scorer(checkpoint_path, model_name, pretrained, device)
    rows = list(iter_winoground_rows(dataset_root, limit=max_examples))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records_path = output_path.with_name(output_path.stem + "_records.jsonl") if write_records else None
    records_handle = records_path.open("w", encoding="utf-8") if records_path else None

    total = 0
    text_correct = 0
    image_correct = 0
    group_correct = 0
    by_tag: Dict[str, List[int]] = {}
    by_collapsed_tag: Dict[str, List[int]] = {}

    try:
        for index, row in enumerate(rows):
            images = [image_from_struct(row["image_0"]), image_from_struct(row["image_1"])]
            captions = [str(row["caption_0"]), str(row["caption_1"])]
            image_features = scorer.image_features(images)
            text_features = scorer.text_features(captions)
            scores = image_features @ text_features.T
            a = float(scores[0, 0].detach().cpu())
            b = float(scores[0, 1].detach().cpu())
            d = float(scores[1, 0].detach().cpu())
            c = float(scores[1, 1].detach().cpu())
            text_ok, image_ok, group_ok = winoground_decisions(a, b, c, d)
            total += 1
            text_correct += int(text_ok)
            image_correct += int(image_ok)
            group_correct += int(group_ok)
            _add_winoground_bucket(by_tag, str(row.get("tag") or "unknown"), text_ok, image_ok, group_ok)
            _add_winoground_bucket(
                by_collapsed_tag,
                str(row.get("collapsed_tag") or "unknown"),
                text_ok,
                image_ok,
                group_ok,
            )
            if records_handle:
                records_handle.write(
                    json.dumps(
                        {
                            "index": index,
                            "id": row.get("id"),
                            "tag": row.get("tag"),
                            "secondary_tag": row.get("secondary_tag"),
                            "collapsed_tag": row.get("collapsed_tag"),
                            "caption_0": captions[0],
                            "caption_1": captions[1],
                            "scores": {
                                "c0_i0": a,
                                "c1_i0": b,
                                "c1_i1": c,
                                "c0_i1": d,
                            },
                            "text_correct": text_ok,
                            "image_correct": image_ok,
                            "group_correct": group_ok,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    finally:
        if records_handle:
            records_handle.close()

    summary = WinogroundSummary(
        scorer=scorer_name(scorer),
        model_name=model_name,
        pretrained=pretrained,
        device=str(scorer.device),
        total=total,
        text_correct=text_correct,
        image_correct=image_correct,
        group_correct=group_correct,
        text_score=text_correct / total if total else math.nan,
        image_score=image_correct / total if total else math.nan,
        group_score=group_correct / total if total else math.nan,
        by_tag=_finalize_winoground_buckets(by_tag),
        by_collapsed_tag=_finalize_winoground_buckets(by_collapsed_tag),
        records_path=str(records_path) if records_path else None,
    )
    output_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def evaluate_coco_retrieval(
    dataset_root: Path,
    output_path: Path,
    checkpoint_path: Optional[Path] = None,
    max_images: int = 1000,
    batch_size: int = 64,
    model_name: str = "ViT-B-16",
    pretrained: str = "laion2b_s34b_b88k",
    device: str = "cuda",
) -> RetrievalSummary:
    import torch

    scorer = build_scorer(checkpoint_path, model_name, pretrained, device)
    rows = list(iter_coco_retrieval_rows(dataset_root, limit=max_images))
    if not rows:
        raise ValueError(f"No COCO retrieval rows found under {dataset_root}")
    images = [row[0] for row in rows]
    captions: List[str] = []
    caption_to_image: List[int] = []
    image_to_caption_indices: List[List[int]] = []
    for image_idx, (_image, row_captions) in enumerate(rows):
        inds = []
        for caption in row_captions:
            inds.append(len(captions))
            captions.append(caption)
            caption_to_image.append(image_idx)
        image_to_caption_indices.append(inds)

    image_features = []
    text_features = []
    for image_batch in batched(images, batch_size):
        image_features.append(scorer.image_features(image_batch).detach().cpu())
    for text_batch in batched(captions, batch_size):
        text_features.append(scorer.text_features(text_batch).detach().cpu())
    image_matrix = torch.cat(image_features, dim=0)
    text_matrix = torch.cat(text_features, dim=0)
    sim = image_matrix @ text_matrix.T

    i2t_r1 = i2t_r5 = i2t_r10 = 0
    for image_idx, gold_caption_indices in enumerate(image_to_caption_indices):
        ranking = torch.argsort(sim[image_idx], descending=True)
        top1 = set(int(x) for x in ranking[:1])
        top5 = set(int(x) for x in ranking[:5])
        top10 = set(int(x) for x in ranking[:10])
        gold = set(gold_caption_indices)
        i2t_r1 += int(bool(gold & top1))
        i2t_r5 += int(bool(gold & top5))
        i2t_r10 += int(bool(gold & top10))

    t2i_r1 = t2i_r5 = t2i_r10 = 0
    sim_t = sim.T
    for caption_idx, gold_image_idx in enumerate(caption_to_image):
        ranking = torch.argsort(sim_t[caption_idx], descending=True)
        top10 = [int(x) for x in ranking[:10]]
        t2i_r1 += int(top10[0] == gold_image_idx)
        t2i_r5 += int(gold_image_idx in top10[:5])
        t2i_r10 += int(gold_image_idx in top10)

    summary = RetrievalSummary(
        scorer=scorer_name(scorer),
        model_name=model_name,
        pretrained=pretrained,
        device=str(scorer.device),
        images=len(images),
        captions=len(captions),
        image_to_text_r1=i2t_r1 / len(images),
        image_to_text_r5=i2t_r5 / len(images),
        image_to_text_r10=i2t_r10 / len(images),
        text_to_image_r1=t2i_r1 / len(captions),
        text_to_image_r5=t2i_r5 / len(captions),
        text_to_image_r10=t2i_r10 / len(captions),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def evaluate_flickr30k_retrieval(
    dataset_root: Path,
    output_path: Path,
    checkpoint_path: Optional[Path] = None,
    split: str = "test",
    max_images: int = 1000,
    batch_size: int = 64,
    model_name: str = "ViT-B-16",
    pretrained: str = "laion2b_s34b_b88k",
    device: str = "cuda",
) -> RetrievalSummary:
    import torch

    scorer = build_scorer(checkpoint_path, model_name, pretrained, device)
    rows = list(iter_flickr30k_retrieval_rows(dataset_root, limit=max_images, split=split))
    if not rows:
        raise ValueError(f"No Flickr30k retrieval rows found under {dataset_root} split={split}")
    images = [row[0] for row in rows]
    captions: List[str] = []
    caption_to_image: List[int] = []
    image_to_caption_indices: List[List[int]] = []
    for image_idx, (_image, row_captions) in enumerate(rows):
        inds = []
        for caption in row_captions:
            inds.append(len(captions))
            captions.append(caption)
            caption_to_image.append(image_idx)
        image_to_caption_indices.append(inds)

    image_features = []
    text_features = []
    for image_batch in batched(images, batch_size):
        image_features.append(scorer.image_features(image_batch).detach().cpu())
    for text_batch in batched(captions, batch_size):
        text_features.append(scorer.text_features(text_batch).detach().cpu())
    image_matrix = torch.cat(image_features, dim=0)
    text_matrix = torch.cat(text_features, dim=0)
    sim = image_matrix @ text_matrix.T

    i2t_r1 = i2t_r5 = i2t_r10 = 0
    for image_idx, gold_caption_indices in enumerate(image_to_caption_indices):
        ranking = torch.argsort(sim[image_idx], descending=True)
        gold = set(gold_caption_indices)
        i2t_r1 += int(bool(gold & set(int(x) for x in ranking[:1])))
        i2t_r5 += int(bool(gold & set(int(x) for x in ranking[:5])))
        i2t_r10 += int(bool(gold & set(int(x) for x in ranking[:10])))

    t2i_r1 = t2i_r5 = t2i_r10 = 0
    sim_t = sim.T
    for caption_idx, gold_image_idx in enumerate(caption_to_image):
        ranking = torch.argsort(sim_t[caption_idx], descending=True)
        top10 = [int(x) for x in ranking[:10]]
        t2i_r1 += int(top10[0] == gold_image_idx)
        t2i_r5 += int(gold_image_idx in top10[:5])
        t2i_r10 += int(gold_image_idx in top10)

    summary = RetrievalSummary(
        scorer=scorer_name(scorer),
        model_name=model_name,
        pretrained=pretrained,
        device=str(scorer.device),
        images=len(images),
        captions=len(captions),
        image_to_text_r1=i2t_r1 / len(images),
        image_to_text_r5=i2t_r5 / len(images),
        image_to_text_r10=i2t_r10 / len(images),
        text_to_image_r1=t2i_r1 / len(captions),
        text_to_image_r5=t2i_r5 / len(captions),
        text_to_image_r10=t2i_r10 / len(captions),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def imagenet_categories() -> List[str]:
    try:
        from torchvision.models import ResNet50_Weights

        return list(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torchvision ImageNet category metadata is required for zero-shot eval") from exc


def evaluate_imagenet_zeroshot(
    dataset_root: Path,
    output_path: Path,
    checkpoint_path: Optional[Path] = None,
    max_examples: int = 5000,
    batch_size: int = 64,
    model_name: str = "ViT-B-16",
    pretrained: str = "laion2b_s34b_b88k",
    device: str = "cuda",
) -> ZeroShotSummary:
    import torch

    scorer = build_scorer(checkpoint_path, model_name, pretrained, device)
    categories = imagenet_categories()
    prompts = [f"a photo of a {name}" for name in categories]
    text_features = []
    for text_batch in batched(prompts, batch_size):
        text_features.append(scorer.text_features(text_batch).detach())
    class_features = torch.cat(text_features, dim=0).T

    examples = list(iter_imagenet_rows(dataset_root, limit=max_examples))
    if not examples:
        raise ValueError(f"No ImageNet validation rows found under {dataset_root}")
    total = 0
    top1 = 0
    top5 = 0
    for batch in batched(examples, batch_size):
        images = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], device=scorer.device)
        image_features = scorer.image_features(images)
        logits = image_features @ class_features.to(scorer.device)
        preds = torch.topk(logits, k=5, dim=-1).indices
        top1 += int((preds[:, 0] == labels).sum().detach().cpu())
        top5 += int((preds == labels.unsqueeze(1)).any(dim=1).sum().detach().cpu())
        total += len(batch)

    summary = ZeroShotSummary(
        scorer=scorer_name(scorer),
        model_name=model_name,
        pretrained=pretrained,
        device=str(scorer.device),
        total=total,
        top1=top1 / total,
        top5=top5 / total,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def write_rop_basis(
    dataset_root: Path,
    output_dir: Path,
    max_images: int = 512,
    protected_rank: int = 64,
    batch_size: int = 64,
    model_name: str = "ViT-B-16",
    pretrained: str = "laion2b_s34b_b88k",
    device: str = "cuda",
) -> dict:
    import torch

    scorer = OpenCLIPScorer(model_name=model_name, pretrained=pretrained, device=device)
    try:
        rows = list(iter_coco_retrieval_rows(dataset_root, limit=max_images))
    except Exception:
        pair_examples = list(iter_pair_examples(dataset_root, limit_per_key=max(1, max_images // len([*SUGARCREPE_KEYS, *ARO_KEYS]))))
        rows = [(example.image, [example.positive]) for example in pair_examples[:max_images]]
    if not rows:
        raise ValueError(f"No anchor rows found under {dataset_root}")

    features = []
    images = [row[0] for row in rows]
    captions = [caption for _image, row_captions in rows for caption in row_captions[:1]]
    for image_batch in batched(images, batch_size):
        features.append(scorer.image_features(image_batch).detach().cpu())
    for text_batch in batched(captions, batch_size):
        features.append(scorer.text_features(text_batch).detach().cpu())
    matrix = torch.cat(features, dim=0).float()
    mean_vec = matrix.mean(dim=0, keepdim=True)
    centered = matrix - mean_vec
    rank = max(1, min(int(protected_rank), centered.shape[0] - 1, centered.shape[1]))
    _u, s, vh = torch.linalg.svd(centered, full_matrices=False)
    variance = s.pow(2)
    total_variance = float(variance.sum().item())
    basis = vh[:rank].T.contiguous()
    explained = variance[:rank] / variance.sum().clamp_min(1e-12)
    output_dir.mkdir(parents=True, exist_ok=True)
    basis_path = output_dir / "protected_basis.pt"
    torch.save(
        {
            "basis": basis,
            "mean": mean_vec.squeeze(0),
            "rank": rank,
            "model_name": model_name,
            "pretrained": pretrained,
            "source": "coco_captions_validation_or_pair_fallback",
        },
        basis_path,
    )
    summary = {
        "basis_path": str(basis_path),
        "model_name": model_name,
        "pretrained": pretrained,
        "device": str(scorer.device),
        "anchor_images": len(images),
        "anchor_captions": len(captions),
        "embedding_count": int(matrix.shape[0]),
        "embedding_dim": int(matrix.shape[1]),
        "protected_rank": rank,
        "total_variance": total_variance,
        "variance_explained": [float(x) for x in explained[: min(rank, 20)].tolist()],
        "cumulative_variance_explained": float(explained.sum().item()),
        "basis_shape": [int(item) for item in basis.shape],
    }
    (output_dir / "rop_basis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def slot_responsibility_probe(
    checkpoint_path: Path,
    dataset_root: Path,
    output_path: Path,
    max_examples_per_key: Optional[int] = 128,
    model_name: str = "ViT-B-16",
    pretrained: str = "laion2b_s34b_b88k",
    device: str = "cuda",
    keys: Optional[Sequence[str]] = None,
) -> dict:
    import torch
    import torch.nn.functional as F

    scorer = build_checkpoint_scorer(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )
    if not isinstance(scorer, RelResidualBottleneckCheckpointScorer):
        raise ValueError(f"Slot probe expects an RRB checkpoint, got {scorer_name(scorer)}")
    examples = list(iter_pair_examples(dataset_root, keys=keys, limit_per_key=max_examples_per_key))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records_path = output_path.with_name(output_path.stem + "_records.jsonl")
    rows = []
    with records_path.open("w", encoding="utf-8") as handle, torch.no_grad():
        for index, example in enumerate(examples):
            if not example.negatives:
                continue
            texts = [example.positive, example.negatives[0]]
            text_tokens, _text_globals, text_base, token_ids = scorer.base.text_token_features_with_ids(texts)
            pos_out = scorer.adapter.text_output(text_tokens[:1], text_base[:1])
            neg_out = scorer.adapter.text_output(text_tokens[1:], text_base[1:])
            edited_span, keep_span = _infer_spans(example.positive, example.negatives[0])
            ids = [int(item) for item in token_ids[0].detach().cpu().tolist()]
            edited_mask = _span_token_mask(scorer.base.tokenizer, ids, edited_span).to(scorer.device)
            keep_mask = _span_token_mask(scorer.base.tokenizer, ids, keep_span).to(scorer.device)
            active_mask = torch.tensor([item not in {0, 49406, 49407} for item in ids], device=scorer.device)
            if edited_mask.sum().item() == 0 or active_mask.sum().item() == 0:
                continue
            resp = pos_out["token_responsibility"][0]
            edited_weights = (resp * edited_mask.float().unsqueeze(0)).sum(dim=-1)
            edited_weights = edited_weights / edited_weights.sum().clamp_min(1e-8)
            keep_weights = (resp * keep_mask.float().unsqueeze(0)).sum(dim=-1)
            keep_weights = keep_weights / keep_weights.sum().clamp_min(1e-8)
            entropy = float((-(edited_weights * edited_weights.clamp_min(1e-8).log()).sum() / math.log(len(edited_weights))).detach().cpu())
            top_values, top_indices = torch.topk(edited_weights, k=min(2, len(edited_weights)))
            top_slot = int(top_indices[0].detach().cpu())
            top_resp = resp[top_slot]
            edited_overlap = float(
                ((top_resp * edited_mask.float()).sum() / (top_resp * active_mask.float()).sum().clamp_min(1e-8))
                .detach()
                .cpu()
            )
            slot_delta = 1.0 - F.cosine_similarity(pos_out["slots"][0], neg_out["slots"][0], dim=-1)
            weighted_delta = float((edited_weights * slot_delta).sum().detach().cpu())
            mean_delta = float(slot_delta.mean().detach().cpu())
            record = {
                "index": index,
                "source": example.source,
                "edit_type": example.edit_type,
                "edited_span": edited_span,
                "keep_span": keep_span,
                "slot_entropy": entropy,
                "top1_slot_concentration": float(top_values[0].detach().cpu()),
                "top2_slot_concentration": float(top_values.sum().detach().cpu()),
                "keep_top1_slot_concentration": float(keep_weights.max().detach().cpu()),
                "edited_span_overlap": edited_overlap,
                "weighted_attribution_delta": weighted_delta,
                "top_slot_delta": float(slot_delta[top_slot].detach().cpu()),
                "mean_slot_delta": mean_delta,
                "slot_delta_lift": float(slot_delta[top_slot].detach().cpu()) - mean_delta,
                "top_slot": top_slot,
            }
            rows.append(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def avg(name: str) -> float:
        return sum(float(row[name]) for row in rows) / len(rows) if rows else math.nan

    summary = {
        "checkpoint": str(checkpoint_path),
        "scorer": scorer_name(scorer),
        "slot_count": scorer.slot_count,
        "examples": len(rows),
        "records_path": str(records_path),
        "mean_slot_entropy": avg("slot_entropy"),
        "mean_top1_slot_concentration": avg("top1_slot_concentration"),
        "mean_top2_slot_concentration": avg("top2_slot_concentration"),
        "mean_keep_top1_slot_concentration": avg("keep_top1_slot_concentration"),
        "mean_edited_span_overlap": avg("edited_span_overlap"),
        "mean_weighted_attribution_delta": avg("weighted_attribution_delta"),
        "mean_slot_delta_lift": avg("slot_delta_lift"),
    }
    concentrated = (
        summary["examples"] > 0
        and summary["mean_top1_slot_concentration"] >= 0.45
        and summary["mean_slot_entropy"] <= 0.65
        and summary["mean_slot_delta_lift"] > 0.0
    )
    summary["verdict"] = "concentrated" if concentrated else "diffuse_or_unproven"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


class EmbeddingAdapter:
    def __init__(self, dim: int, device, variant: str = "full_relbottleneck", slot_count: Optional[int] = None) -> None:
        import torch
        from torch import nn

        slot_count = slot_count or 8
        if variant in {"bottleneck_only", "full_relbottleneck", "best_full_model"}:
            hidden = max(16, min(dim, slot_count * 32))
        elif variant == "overbuilt_targeting_or_relation_head":
            hidden = dim * 2
        else:
            hidden = dim

        self.module = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)).to(device)

    def parameters(self):
        return self.module.parameters()

    def __call__(self, features):
        import torch.nn.functional as F

        return F.normalize(features + 0.1 * self.module(features), dim=-1)


class TokenRelBottleneckAdapter:
    def __init__(
        self,
        image_token_dim: int,
        text_token_dim: int,
        embed_dim: int,
        device,
        slot_count: Optional[int] = None,
    ) -> None:
        from torch import nn

        from .relbottleneck import RelBottleneckFusion

        slot_count = slot_count or 8
        self.module = nn.ModuleDict(
            {
                "image": RelBottleneckFusion(
                    token_dim=image_token_dim,
                    clip_embed_dim=embed_dim,
                    slot_count=slot_count,
                ),
                "text": RelBottleneckFusion(
                    token_dim=text_token_dim,
                    clip_embed_dim=embed_dim,
                    slot_count=slot_count,
                ),
            }
        ).to(device)

    def parameters(self):
        return self.module.parameters()

    def image_features(self, tokens, global_token):
        return self.image_output(tokens, global_token)["embedding"]

    def text_features(self, tokens, global_token):
        return self.text_output(tokens, global_token)["embedding"]

    def image_output(self, tokens, global_token):
        return self.module["image"](tokens, global_token=global_token)

    def text_output(self, tokens, global_token):
        return self.module["text"](tokens, global_token=global_token)


def _prepare_protected_basis(protected_basis, device):
    if protected_basis is None:
        return None
    import torch.nn.functional as F

    basis = protected_basis.to(device=device)
    if basis.ndim != 2 or basis.shape[1] == 0:
        return None
    return F.normalize(basis.float(), dim=0)


def _apply_rop_output(output: dict, protected_basis, rop_mode: str) -> dict:
    import torch
    import torch.nn.functional as F

    if protected_basis is None:
        output["protected_residual_energy"] = torch.zeros_like(output["residual_relative_magnitude"])
        return output
    residual = output["residual"]
    coeff = residual @ protected_basis
    protected = coeff @ protected_basis.T
    output["protected_residual_energy"] = protected.pow(2).sum(dim=-1)
    if rop_mode in {"project", "penalty_project"}:
        residual = residual - protected
        embedding = F.normalize(output["base_embedding"] + residual, dim=-1)
        output["residual"] = residual
        output["embedding"] = embedding
        output["residual_relative_magnitude"] = residual.norm(dim=-1) / output["base_embedding"].norm(dim=-1).clamp_min(1e-6)
        output["z0_cosine"] = F.cosine_similarity(output["base_embedding"], embedding, dim=-1)
    return output


class RelResidualBottleneckAdapter:
    def __init__(
        self,
        image_token_dim: int,
        text_token_dim: int,
        embed_dim: int,
        device,
        slot_count: Optional[int] = None,
        residual_alpha: float = 0.05,
        residual_gate: str = "fixed",
        protected_basis=None,
        rop_mode: str = "none",
    ) -> None:
        from torch import nn

        from .relbottleneck import RelResidualBottleneck

        slot_count = slot_count or 8
        self.module = nn.ModuleDict(
            {
                "image": RelResidualBottleneck(
                    token_dim=image_token_dim,
                    clip_embed_dim=embed_dim,
                    slot_count=slot_count,
                    residual_alpha=residual_alpha,
                    gate_type=residual_gate,
                ),
                "text": RelResidualBottleneck(
                    token_dim=text_token_dim,
                    clip_embed_dim=embed_dim,
                    slot_count=slot_count,
                    residual_alpha=residual_alpha,
                    gate_type=residual_gate,
                ),
            }
        ).to(device)
        self.protected_basis = _prepare_protected_basis(protected_basis, device)
        self.rop_mode = rop_mode

    def parameters(self):
        return self.module.parameters()

    def image_features(self, tokens, base_embedding):
        return self.image_output(tokens, base_embedding)["embedding"]

    def text_features(self, tokens, base_embedding):
        return self.text_output(tokens, base_embedding)["embedding"]

    def image_output(self, tokens, base_embedding):
        return _apply_rop_output(self.module["image"](tokens, base_embedding), self.protected_basis, self.rop_mode)

    def text_output(self, tokens, base_embedding):
        return _apply_rop_output(self.module["text"](tokens, base_embedding), self.protected_basis, self.rop_mode)


class NoSlotResidualAdapter:
    def __init__(
        self,
        image_token_dim: int,
        text_token_dim: int,
        embed_dim: int,
        device,
        residual_alpha: float = 0.05,
        residual_gate: str = "fixed",
        protected_basis=None,
        rop_mode: str = "none",
    ) -> None:
        from torch import nn

        self.module = nn.ModuleDict(
            {
                "image": NoSlotResidualBranch(
                    token_dim=image_token_dim,
                    embed_dim=embed_dim,
                    residual_alpha=residual_alpha,
                    gate_type=residual_gate,
                ),
                "text": NoSlotResidualBranch(
                    token_dim=text_token_dim,
                    embed_dim=embed_dim,
                    residual_alpha=residual_alpha,
                    gate_type=residual_gate,
                ),
            }
        ).to(device)
        self.protected_basis = _prepare_protected_basis(protected_basis, device)
        self.rop_mode = rop_mode

    def parameters(self):
        return self.module.parameters()

    def image_features(self, tokens, base_embedding):
        return self.image_output(tokens, base_embedding)["embedding"]

    def text_features(self, tokens, base_embedding):
        return self.text_output(tokens, base_embedding)["embedding"]

    def image_output(self, tokens, base_embedding):
        return _apply_rop_output(self.module["image"](tokens, base_embedding), self.protected_basis, self.rop_mode)

    def text_output(self, tokens, base_embedding):
        return _apply_rop_output(self.module["text"](tokens, base_embedding), self.protected_basis, self.rop_mode)


class NoSlotResidualBranch(_torch_nn.Module if _torch_nn is not None else object):
    def __init__(
        self,
        token_dim: int,
        embed_dim: int,
        residual_alpha: float = 0.05,
        gate_type: str = "fixed",
    ) -> None:
        if _torch_nn is None:
            raise RuntimeError("PyTorch is required for no-slot residual training")
        super().__init__()
        nn = _torch_nn
        if gate_type not in {"fixed", "scalar", "vector"}:
            raise ValueError(f"Unsupported gate_type: {gate_type}")
        self.norm = nn.LayerNorm(token_dim)
        self.delta = nn.Sequential(
            nn.Linear(token_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        nn.init.normal_(self.delta[-1].weight, std=1e-3)
        nn.init.zeros_(self.delta[-1].bias)
        self.gate_type = gate_type
        self.max_alpha = float(residual_alpha)
        if gate_type == "fixed":
            self.register_buffer("fixed_gate", self.norm.weight.new_tensor(float(residual_alpha)))
        else:
            shape = (embed_dim,) if gate_type == "vector" else (1,)
            self.gate_logits = nn.Parameter(self.norm.weight.new_zeros(shape))

    def forward(self, tokens, base_embedding):
        import torch
        import torch.nn.functional as F

        pooled = self.norm(tokens).mean(dim=1)
        delta_rel = self.delta(pooled)
        if self.gate_type == "fixed":
            gate = self.fixed_gate
        else:
            gate = torch.sigmoid(self.gate_logits) * self.max_alpha
        residual = gate * delta_rel
        embedding = F.normalize(base_embedding + residual, dim=-1)
        residual_relative_magnitude = residual.norm(dim=-1) / base_embedding.norm(dim=-1).clamp_min(1e-6)
        z0_cosine = F.cosine_similarity(base_embedding, embedding, dim=-1)
        return {
            "embedding": embedding,
            "base_embedding": base_embedding,
            "delta_rel": delta_rel,
            "residual": residual,
            "gate": gate,
            "residual_relative_magnitude": residual_relative_magnitude,
            "z0_cosine": z0_cosine,
            "slots": pooled.unsqueeze(1),
            "token_responsibility": tokens.new_zeros(tokens.shape[0], 1, tokens.shape[1]),
            "slot_affinity": tokens.new_ones(tokens.shape[0], 1, 1),
        }


def loss_mode_for_variant(variant: str) -> str:
    if variant in {"plain_finetune", "bottleneck_only"}:
        return "in_batch_clip"
    if variant == "no_targeted_loss":
        return "weak_margin"
    if variant == "generic_hard_negatives":
        return "shuffled_negative_margin"
    if variant == "unfiltered_structured_edits":
        return "low_weight_margin"
    if variant == "overbuilt_targeting_or_relation_head":
        return "strong_margin"
    return "structured_margin"


def margin_for_loss_mode(loss_mode: str) -> float:
    return {
        "weak_margin": 0.03,
        "low_weight_margin": 0.05,
        "strong_margin": 0.15,
    }.get(loss_mode, 0.1)


def train_tiny_adapter(
    dataset_root: Path,
    output_dir: Path,
    train_examples: int = 256,
    eval_examples: int = 128,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 1e-4,
    model_name: str = "ViT-B-16",
    pretrained: str = "laion2b_s34b_b88k",
    device: str = "cuda",
    seed: int = 0,
    variant: str = "full_relbottleneck",
    slot_count: Optional[int] = None,
    lambda_distill: float = 0.0,
    lambda_original: float = 0.0,
    lambda_targeted: float = 0.0,
) -> TrainSummary:
    import torch
    import torch.nn.functional as F

    random.seed(seed)
    torch.manual_seed(seed)
    scorer = OpenCLIPScorer(model_name=model_name, pretrained=pretrained, device=device)
    examples = list(iter_pair_examples(dataset_root, limit_per_key=train_examples))
    if not examples:
        raise ValueError(f"No training examples found under {dataset_root}")
    random.shuffle(examples)
    train_set = examples[:train_examples]
    eval_set = examples[train_examples : train_examples + eval_examples]
    if not eval_set:
        eval_set = examples[: min(eval_examples, len(examples))]

    image_features = []
    pos_features = []
    neg_features = []
    for example in train_set:
        image_features.append(scorer.image_features([example.image]).squeeze(0))
        text_features = scorer.text_features([example.positive, example.negatives[0]])
        pos_features.append(text_features[0])
        neg_features.append(text_features[1])

    dim = int(image_features[0].shape[-1])
    adapter = EmbeddingAdapter(dim, scorer.device, variant=variant, slot_count=slot_count)
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    final_loss = 0.0
    final_task_loss = 0.0
    final_distill_loss = 0.0
    final_original_loss = 0.0
    indices = list(range(len(train_set)))

    loss_mode = loss_mode_for_variant(variant)

    for _epoch in range(epochs):
        random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            img = torch.stack([image_features[idx] for idx in batch_idx]).to(scorer.device)
            pos = torch.stack([pos_features[idx] for idx in batch_idx]).to(scorer.device)
            neg = torch.stack([neg_features[idx] for idx in batch_idx]).to(scorer.device)
            img_a = adapter(img)
            pos_a = adapter(pos)
            neg_a = adapter(neg)
            if loss_mode == "in_batch_clip":
                logits = img_a @ pos_a.T / 0.07
                labels = torch.arange(logits.shape[0], device=scorer.device)
                task_loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            else:
                if loss_mode == "shuffled_negative_margin" and neg_a.shape[0] > 1:
                    neg_a = neg_a[torch.randperm(neg_a.shape[0], device=scorer.device)]
                pos_score = (img_a * pos_a).sum(dim=-1)
                neg_score = (img_a * neg_a).sum(dim=-1)
                margin = margin_for_loss_mode(loss_mode)
                task_loss = F.relu(margin - pos_score + neg_score).mean()
            if lambda_distill:
                distill_loss = (
                    (1.0 - (img_a * img.detach()).sum(dim=-1)).mean()
                    + (1.0 - (pos_a * pos.detach()).sum(dim=-1)).mean()
                    + (1.0 - (neg_a * neg.detach()).sum(dim=-1)).mean()
                ) / 3.0
            else:
                distill_loss = torch.zeros((), device=scorer.device)
            if lambda_original and loss_mode != "in_batch_clip":
                original_logits = img_a @ pos_a.T / 0.07
                labels = torch.arange(original_logits.shape[0], device=scorer.device)
                original_loss = (
                    F.cross_entropy(original_logits, labels)
                    + F.cross_entropy(original_logits.T, labels)
                ) / 2
            else:
                original_loss = torch.zeros((), device=scorer.device)
            loss = task_loss + lambda_distill * distill_loss + lambda_original * original_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            final_loss = float(loss.detach().cpu())
            final_task_loss = float(task_loss.detach().cpu())
            final_distill_loss = float(distill_loss.detach().cpu())
            final_original_loss = float(original_loss.detach().cpu())

    eval_total = 0
    eval_correct = 0
    eval_margins: List[float] = []
    with torch.no_grad():
        for example in eval_set:
            img = adapter(scorer.image_features([example.image]))
            texts = adapter(scorer.text_features([example.positive, example.negatives[0]]))
            scores = (img @ texts.T).squeeze(0)
            ok = int(int(torch.argmax(scores).detach().cpu()) == 0)
            eval_correct += ok
            eval_total += 1
            eval_margins.append(float((scores[0] - scores[1]).detach().cpu()))

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "adapter.pt"
    torch.save(
        {
            "backend": "adapter",
            "adapter_state_dict": adapter.module.state_dict(),
            "model_name": model_name,
            "pretrained": pretrained,
            "seed": seed,
            "variant": variant,
            "slot_count": slot_count,
            "lambda_distill": lambda_distill,
            "lambda_original": lambda_original,
            "lambda_targeted": lambda_targeted,
            "dim": dim,
        },
        checkpoint,
    )
    summary = TrainSummary(
        backend="adapter",
        variant=variant,
        model_name=model_name,
        pretrained=pretrained,
        device=str(scorer.device),
        train_examples=len(train_set),
        epochs=epochs,
        final_loss=final_loss,
        eval_accuracy=eval_correct / eval_total if eval_total else math.nan,
        eval_total=eval_total,
        eval_correct=eval_correct,
        eval_mean_margin=sum(eval_margins) / len(eval_margins) if eval_margins else math.nan,
        checkpoint=str(checkpoint),
        loss_mode=loss_mode,
        lambda_distill=lambda_distill,
        lambda_original=lambda_original,
        lambda_targeted=lambda_targeted,
        final_task_loss=final_task_loss,
        final_distill_loss=final_distill_loss,
        final_original_loss=final_original_loss,
        final_targeted_loss=0.0,
    )
    (output_dir / "train_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def train_token_bottleneck(
    dataset_root: Path,
    output_dir: Path,
    train_examples: int = 256,
    eval_examples: int = 128,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 1e-4,
    model_name: str = "ViT-B-16",
    pretrained: str = "laion2b_s34b_b88k",
    device: str = "cuda",
    seed: int = 0,
    variant: str = "full_relbottleneck",
    slot_count: Optional[int] = None,
    lambda_distill: float = 0.0,
    lambda_original: float = 0.0,
    lambda_targeted: float = 0.0,
) -> TrainSummary:
    import torch
    import torch.nn.functional as F

    random.seed(seed)
    torch.manual_seed(seed)
    scorer = OpenCLIPScorer(model_name=model_name, pretrained=pretrained, device=device)
    examples = list(iter_pair_examples(dataset_root, limit_per_key=train_examples))
    if not examples:
        raise ValueError(f"No training examples found under {dataset_root}")
    random.shuffle(examples)
    train_set = examples[:train_examples]
    eval_set = examples[train_examples : train_examples + eval_examples]
    if not eval_set:
        eval_set = examples[: min(eval_examples, len(examples))]

    image_tokens = []
    image_globals = []
    image_base = []
    pos_tokens = []
    pos_globals = []
    pos_base = []
    neg_tokens = []
    neg_globals = []
    neg_base = []
    edited_masks = []
    keep_masks = []
    for example in train_set:
        tokens, global_token, base_embedding = scorer.image_token_features([example.image])
        image_tokens.append(tokens.squeeze(0).detach().cpu())
        image_globals.append(global_token.squeeze(0).detach().cpu())
        image_base.append(base_embedding.squeeze(0).detach().cpu())
        text_tokens, text_globals, text_base, text_token_ids = scorer.text_token_features_with_ids(
            [example.positive, example.negatives[0]]
        )
        pos_tokens.append(text_tokens[0].detach().cpu())
        pos_globals.append(text_globals[0].detach().cpu())
        pos_base.append(text_base[0].detach().cpu())
        neg_tokens.append(text_tokens[1].detach().cpu())
        neg_globals.append(text_globals[1].detach().cpu())
        neg_base.append(text_base[1].detach().cpu())
        edited_span, keep_span = _infer_spans(example.positive, example.negatives[0])
        token_ids = [int(item) for item in text_token_ids[0].detach().cpu().tolist()]
        edited_masks.append(_span_token_mask(scorer.tokenizer, token_ids, edited_span))
        keep_masks.append(_span_token_mask(scorer.tokenizer, token_ids, keep_span))

    image_token_dim = int(image_tokens[0].shape[-1])
    text_token_dim = int(pos_tokens[0].shape[-1])
    embed_dim = int(image_base[0].shape[-1])
    adapter = TokenRelBottleneckAdapter(
        image_token_dim=image_token_dim,
        text_token_dim=text_token_dim,
        embed_dim=embed_dim,
        device=scorer.device,
        slot_count=slot_count,
    )
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    final_loss = 0.0
    final_task_loss = 0.0
    final_distill_loss = 0.0
    final_original_loss = 0.0
    final_targeted_loss = 0.0
    indices = list(range(len(train_set)))
    loss_mode = f"token_{loss_mode_for_variant(variant)}"

    for _epoch in range(epochs):
        random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            img_tokens = torch.stack([image_tokens[idx] for idx in batch_idx]).to(scorer.device)
            img_global = torch.stack([image_globals[idx] for idx in batch_idx]).to(scorer.device)
            img_teacher = torch.stack([image_base[idx] for idx in batch_idx]).to(scorer.device)
            pos_tok = torch.stack([pos_tokens[idx] for idx in batch_idx]).to(scorer.device)
            pos_global = torch.stack([pos_globals[idx] for idx in batch_idx]).to(scorer.device)
            pos_teacher = torch.stack([pos_base[idx] for idx in batch_idx]).to(scorer.device)
            neg_tok = torch.stack([neg_tokens[idx] for idx in batch_idx]).to(scorer.device)
            neg_global = torch.stack([neg_globals[idx] for idx in batch_idx]).to(scorer.device)
            neg_teacher = torch.stack([neg_base[idx] for idx in batch_idx]).to(scorer.device)
            edited_mask = torch.stack([edited_masks[idx] for idx in batch_idx]).to(scorer.device)
            keep_mask = torch.stack([keep_masks[idx] for idx in batch_idx]).to(scorer.device)
            img_a = adapter.image_features(img_tokens, img_global)
            pos_out = adapter.text_output(pos_tok, pos_global)
            neg_out = adapter.text_output(neg_tok, neg_global)
            pos_a = pos_out["embedding"]
            neg_a = neg_out["embedding"]
            base_loss_mode = loss_mode.replace("token_", "", 1)
            if base_loss_mode == "in_batch_clip":
                logits = img_a @ pos_a.T / 0.07
                labels = torch.arange(logits.shape[0], device=scorer.device)
                task_loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            else:
                if base_loss_mode == "shuffled_negative_margin" and neg_a.shape[0] > 1:
                    neg_a = neg_a[torch.randperm(neg_a.shape[0], device=scorer.device)]
                pos_score = (img_a * pos_a).sum(dim=-1)
                neg_score = (img_a * neg_a).sum(dim=-1)
                task_loss = F.relu(margin_for_loss_mode(base_loss_mode) - pos_score + neg_score).mean()
            if lambda_distill:
                distill_loss = (
                    (1.0 - (img_a * img_teacher).sum(dim=-1)).mean()
                    + (1.0 - (pos_a * pos_teacher).sum(dim=-1)).mean()
                    + (1.0 - (neg_a * neg_teacher).sum(dim=-1)).mean()
                ) / 3.0
            else:
                distill_loss = torch.zeros((), device=scorer.device)
            if lambda_original and base_loss_mode != "in_batch_clip":
                original_logits = img_a @ pos_a.T / 0.07
                labels = torch.arange(original_logits.shape[0], device=scorer.device)
                original_loss = (
                    F.cross_entropy(original_logits, labels)
                    + F.cross_entropy(original_logits.T, labels)
                ) / 2
            else:
                original_loss = torch.zeros((), device=scorer.device)
            if lambda_targeted and base_loss_mode not in {"in_batch_clip", "weak_margin", "shuffled_negative_margin"}:
                from .relbottleneck import targeted_slot_loss

                targeted_loss = targeted_slot_loss(
                    pos_out["slots"],
                    neg_out["slots"],
                    pos_out["token_responsibility"],
                    edited_mask,
                    keep_mask,
                )
            else:
                targeted_loss = torch.zeros((), device=scorer.device)
            loss = (
                task_loss
                + lambda_distill * distill_loss
                + lambda_original * original_loss
                + lambda_targeted * targeted_loss
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            final_loss = float(loss.detach().cpu())
            final_task_loss = float(task_loss.detach().cpu())
            final_distill_loss = float(distill_loss.detach().cpu())
            final_original_loss = float(original_loss.detach().cpu())
            final_targeted_loss = float(targeted_loss.detach().cpu())

    eval_total = 0
    eval_correct = 0
    eval_margins: List[float] = []
    with torch.no_grad():
        for example in eval_set:
            img_tokens, img_global, _img_base = scorer.image_token_features([example.image])
            text_tokens, text_globals, _text_base = scorer.text_token_features([example.positive, example.negatives[0]])
            img = adapter.image_features(img_tokens, img_global)
            pos = adapter.text_features(text_tokens[:1], text_globals[:1])
            neg = adapter.text_features(text_tokens[1:], text_globals[1:])
            scores = torch.stack([(img * pos).sum(dim=-1), (img * neg).sum(dim=-1)], dim=1).squeeze(0)
            ok = int(int(torch.argmax(scores).detach().cpu()) == 0)
            eval_correct += ok
            eval_total += 1
            eval_margins.append(float((scores[0] - scores[1]).detach().cpu()))

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "adapter.pt"
    torch.save(
        {
            "backend": "token_bottleneck",
            "model_state_dict": adapter.module.state_dict(),
            "model_name": model_name,
            "pretrained": pretrained,
            "seed": seed,
            "variant": variant,
            "slot_count": slot_count,
            "lambda_distill": lambda_distill,
            "lambda_original": lambda_original,
            "lambda_targeted": lambda_targeted,
            "image_token_dim": image_token_dim,
            "text_token_dim": text_token_dim,
            "dim": embed_dim,
        },
        checkpoint,
    )
    summary = TrainSummary(
        backend="token_bottleneck",
        variant=variant,
        model_name=model_name,
        pretrained=pretrained,
        device=str(scorer.device),
        train_examples=len(train_set),
        epochs=epochs,
        final_loss=final_loss,
        eval_accuracy=eval_correct / eval_total if eval_total else math.nan,
        eval_total=eval_total,
        eval_correct=eval_correct,
        eval_mean_margin=sum(eval_margins) / len(eval_margins) if eval_margins else math.nan,
        checkpoint=str(checkpoint),
        loss_mode=loss_mode,
        lambda_distill=lambda_distill,
        lambda_original=lambda_original,
        lambda_targeted=lambda_targeted,
        final_task_loss=final_task_loss,
        final_distill_loss=final_distill_loss,
        final_original_loss=final_original_loss,
        final_targeted_loss=final_targeted_loss,
    )
    (output_dir / "train_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _append_rrb_drift(metrics: Dict[str, List[float]], *outputs) -> None:
    for output in outputs:
        metrics["residual_relative_magnitude"].extend(
            float(x) for x in output["residual_relative_magnitude"].detach().cpu().tolist()
        )
        metrics["z0_cosine"].extend(float(x) for x in output["z0_cosine"].detach().cpu().tolist())


def _write_rrb_drift(output_dir: Path, metrics: Dict[str, List[float]]) -> Dict[str, float]:
    residual_values = metrics.get("residual_relative_magnitude", [])
    cosine_values = metrics.get("z0_cosine", [])
    summary = {
        "residual_relative_magnitude": {
            "mean": sum(residual_values) / len(residual_values) if residual_values else math.nan,
            "min": min(residual_values) if residual_values else math.nan,
            "max": max(residual_values) if residual_values else math.nan,
            "histogram": numeric_histogram(residual_values),
        },
        "z0_cosine": {
            "mean": sum(cosine_values) / len(cosine_values) if cosine_values else math.nan,
            "min": min(cosine_values) if cosine_values else math.nan,
            "max": max(cosine_values) if cosine_values else math.nan,
            "histogram": numeric_histogram(cosine_values),
        },
    }
    (output_dir / "rrb_drift_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "residual_relative_magnitude": float(summary["residual_relative_magnitude"]["mean"]),
        "z0_cosine_mean": float(summary["z0_cosine"]["mean"]),
        "z0_cosine_min": float(summary["z0_cosine"]["min"]),
    }


def load_protected_basis(path: Optional[Path], device=None):
    if not path:
        return None
    import torch

    payload = load_torch_checkpoint(torch, path, map_location=device or "cpu")
    if isinstance(payload, dict) and "basis" in payload:
        basis = payload["basis"]
    else:
        basis = payload
    return basis.to(device=device) if device is not None else basis


def train_rrb(
    dataset_root: Path,
    output_dir: Path,
    train_examples: int = 256,
    eval_examples: int = 128,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 1e-4,
    model_name: str = "ViT-B-16",
    pretrained: str = "laion2b_s34b_b88k",
    device: str = "cuda",
    seed: int = 0,
    variant: str = "rrb_full",
    slot_count: Optional[int] = None,
    lambda_distill: float = 0.0,
    lambda_original: float = 0.0,
    lambda_targeted: float = 0.0,
    lambda_anchor: float = 5.0,
    lambda_delta: float = 0.01,
    residual_alpha: float = 0.05,
    residual_gate: str = "fixed",
    anchor_ratio: float = 0.5,
    protected_basis_path: Optional[Path] = None,
    lambda_rop: float = 0.0,
    rop_mode: str = "none",
) -> TrainSummary:
    import torch
    import torch.nn.functional as F

    random.seed(seed)
    torch.manual_seed(seed)
    scorer = OpenCLIPScorer(model_name=model_name, pretrained=pretrained, device=device)
    examples = list(iter_pair_examples(dataset_root, limit_per_key=train_examples))
    if not examples:
        raise ValueError(f"No training examples found under {dataset_root}")
    random.shuffle(examples)
    train_set = examples[:train_examples]
    eval_set = examples[train_examples : train_examples + eval_examples]
    if not eval_set:
        eval_set = examples[: min(eval_examples, len(examples))]

    image_tokens = []
    image_base = []
    pos_tokens = []
    pos_base = []
    neg_tokens = []
    neg_base = []
    edited_masks = []
    keep_masks = []
    for example in train_set:
        tokens, _global_token, base_embedding = scorer.image_token_features([example.image])
        image_tokens.append(tokens.squeeze(0).detach().cpu())
        image_base.append(base_embedding.squeeze(0).detach().cpu())
        text_tokens, _text_globals, text_base, text_token_ids = scorer.text_token_features_with_ids(
            [example.positive, example.negatives[0]]
        )
        pos_tokens.append(text_tokens[0].detach().cpu())
        pos_base.append(text_base[0].detach().cpu())
        neg_tokens.append(text_tokens[1].detach().cpu())
        neg_base.append(text_base[1].detach().cpu())
        edited_span, keep_span = _infer_spans(example.positive, example.negatives[0])
        token_ids = [int(item) for item in text_token_ids[0].detach().cpu().tolist()]
        edited_masks.append(_span_token_mask(scorer.tokenizer, token_ids, edited_span))
        keep_masks.append(_span_token_mask(scorer.tokenizer, token_ids, keep_span))

    image_token_dim = int(image_tokens[0].shape[-1])
    text_token_dim = int(pos_tokens[0].shape[-1])
    embed_dim = int(image_base[0].shape[-1])
    protected_basis = load_protected_basis(protected_basis_path, scorer.device)
    if protected_basis is None:
        rop_mode = "none"
        lambda_rop = 0.0
    adapter_cls = NoSlotResidualAdapter if variant == "no_slot_residual_adapter" else RelResidualBottleneckAdapter
    adapter_kwargs = {
        "image_token_dim": image_token_dim,
        "text_token_dim": text_token_dim,
        "embed_dim": embed_dim,
        "device": scorer.device,
        "residual_alpha": residual_alpha,
        "residual_gate": residual_gate,
        "protected_basis": protected_basis,
        "rop_mode": rop_mode,
    }
    if adapter_cls is RelResidualBottleneckAdapter:
        adapter_kwargs["slot_count"] = slot_count
    adapter = adapter_cls(
        **adapter_kwargs,
    )
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    final_loss = 0.0
    final_task_loss = 0.0
    final_distill_loss = 0.0
    final_original_loss = 0.0
    final_targeted_loss = 0.0
    final_anchor_loss = 0.0
    final_delta_loss = 0.0
    final_rop_loss = 0.0
    indices = list(range(len(train_set)))
    base_loss_mode = loss_mode_for_variant(variant)
    loss_mode = f"rrb_{base_loss_mode}"
    anchor_ratio = max(0.0, min(1.0, float(anchor_ratio)))
    task_weight = max(0.0, 1.0 - anchor_ratio)

    for _epoch in range(epochs):
        random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            img_tokens = torch.stack([image_tokens[idx] for idx in batch_idx]).to(scorer.device)
            img_teacher = torch.stack([image_base[idx] for idx in batch_idx]).to(scorer.device)
            pos_tok = torch.stack([pos_tokens[idx] for idx in batch_idx]).to(scorer.device)
            pos_teacher = torch.stack([pos_base[idx] for idx in batch_idx]).to(scorer.device)
            neg_tok = torch.stack([neg_tokens[idx] for idx in batch_idx]).to(scorer.device)
            neg_teacher = torch.stack([neg_base[idx] for idx in batch_idx]).to(scorer.device)
            edited_mask = torch.stack([edited_masks[idx] for idx in batch_idx]).to(scorer.device)
            keep_mask = torch.stack([keep_masks[idx] for idx in batch_idx]).to(scorer.device)
            img_out = adapter.image_output(img_tokens, img_teacher)
            pos_out = adapter.text_output(pos_tok, pos_teacher)
            neg_out = adapter.text_output(neg_tok, neg_teacher)
            img_a = img_out["embedding"]
            pos_a = pos_out["embedding"]
            neg_a = neg_out["embedding"]
            if base_loss_mode == "in_batch_clip":
                logits = img_a @ pos_a.T / 0.07
                labels = torch.arange(logits.shape[0], device=scorer.device)
                task_loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            else:
                if base_loss_mode == "shuffled_negative_margin" and neg_a.shape[0] > 1:
                    neg_a = neg_a[torch.randperm(neg_a.shape[0], device=scorer.device)]
                pos_score = (img_a * pos_a).sum(dim=-1)
                neg_score = (img_a * neg_a).sum(dim=-1)
                task_loss = F.relu(margin_for_loss_mode(base_loss_mode) - pos_score + neg_score).mean()
            if lambda_distill:
                distill_loss = (
                    (1.0 - (img_a * img_teacher).sum(dim=-1)).mean()
                    + (1.0 - (pos_a * pos_teacher).sum(dim=-1)).mean()
                    + (1.0 - (neg_a * neg_teacher).sum(dim=-1)).mean()
                ) / 3.0
            else:
                distill_loss = torch.zeros((), device=scorer.device)
            if lambda_original and base_loss_mode != "in_batch_clip":
                original_logits = img_a @ pos_a.T / 0.07
                labels = torch.arange(original_logits.shape[0], device=scorer.device)
                original_loss = (
                    F.cross_entropy(original_logits, labels)
                    + F.cross_entropy(original_logits.T, labels)
                ) / 2
            else:
                original_loss = torch.zeros((), device=scorer.device)
            if lambda_targeted and base_loss_mode not in {"in_batch_clip", "weak_margin", "shuffled_negative_margin"}:
                from .relbottleneck import targeted_slot_loss

                targeted_loss = targeted_slot_loss(
                    pos_out["slots"],
                    neg_out["slots"],
                    pos_out["token_responsibility"],
                    edited_mask,
                    keep_mask,
                )
            else:
                targeted_loss = torch.zeros((), device=scorer.device)
            anchor_loss = (
                F.mse_loss(img_a, img_teacher)
                + F.mse_loss(pos_a, pos_teacher)
                + F.mse_loss(neg_a, neg_teacher)
            ) / 3.0
            delta_loss = (
                img_out["residual"].pow(2).sum(dim=-1).mean()
                + pos_out["residual"].pow(2).sum(dim=-1).mean()
                + neg_out["residual"].pow(2).sum(dim=-1).mean()
            ) / 3.0
            if lambda_rop and rop_mode in {"penalty", "penalty_project"}:
                rop_loss = (
                    img_out["protected_residual_energy"].mean()
                    + pos_out["protected_residual_energy"].mean()
                    + neg_out["protected_residual_energy"].mean()
                ) / 3.0
            else:
                rop_loss = torch.zeros((), device=scorer.device)
            loss = (
                task_weight * task_loss
                + lambda_distill * distill_loss
                + lambda_original * original_loss
                + lambda_targeted * targeted_loss
                + lambda_anchor * anchor_ratio * anchor_loss
                + lambda_delta * delta_loss
                + lambda_rop * rop_loss
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            final_loss = float(loss.detach().cpu())
            final_task_loss = float(task_loss.detach().cpu())
            final_distill_loss = float(distill_loss.detach().cpu())
            final_original_loss = float(original_loss.detach().cpu())
            final_targeted_loss = float(targeted_loss.detach().cpu())
            final_anchor_loss = float(anchor_loss.detach().cpu())
            final_delta_loss = float(delta_loss.detach().cpu())
            final_rop_loss = float(rop_loss.detach().cpu())

    eval_total = 0
    eval_correct = 0
    eval_margins: List[float] = []
    drift_metrics: Dict[str, List[float]] = {"residual_relative_magnitude": [], "z0_cosine": [], "protected_residual_energy": []}
    with torch.no_grad():
        for example in eval_set:
            img_tokens, _img_global, img_base_eval = scorer.image_token_features([example.image])
            text_tokens, _text_globals, text_base_eval = scorer.text_token_features(
                [example.positive, example.negatives[0]]
            )
            img_out = adapter.image_output(img_tokens, img_base_eval)
            pos_out = adapter.text_output(text_tokens[:1], text_base_eval[:1])
            neg_out = adapter.text_output(text_tokens[1:], text_base_eval[1:])
            img = img_out["embedding"]
            pos = pos_out["embedding"]
            neg = neg_out["embedding"]
            scores = torch.stack([(img * pos).sum(dim=-1), (img * neg).sum(dim=-1)], dim=1).squeeze(0)
            ok = int(int(torch.argmax(scores).detach().cpu()) == 0)
            eval_correct += ok
            eval_total += 1
            eval_margins.append(float((scores[0] - scores[1]).detach().cpu()))
            _append_rrb_drift(drift_metrics, img_out, pos_out, neg_out)
            for output in [img_out, pos_out, neg_out]:
                drift_metrics["protected_residual_energy"].extend(
                    float(x) for x in output["protected_residual_energy"].detach().cpu().tolist()
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    drift_summary = _write_rrb_drift(output_dir, drift_metrics)
    checkpoint = output_dir / "adapter.pt"
    torch.save(
        {
            "backend": "rrb",
            "model_state_dict": adapter.module.state_dict(),
            "model_name": model_name,
            "pretrained": pretrained,
            "seed": seed,
            "variant": variant,
            "slot_count": slot_count,
            "lambda_distill": lambda_distill,
            "lambda_original": lambda_original,
            "lambda_targeted": lambda_targeted,
            "lambda_anchor": lambda_anchor,
            "lambda_delta": lambda_delta,
            "anchor_ratio": anchor_ratio,
            "residual_alpha": residual_alpha,
            "residual_gate": residual_gate,
            "rop_mode": rop_mode,
            "lambda_rop": lambda_rop,
            "protected_basis": protected_basis.detach().cpu() if protected_basis is not None else None,
            "protected_basis_path": str(protected_basis_path) if protected_basis_path else None,
            "image_token_dim": image_token_dim,
            "text_token_dim": text_token_dim,
            "dim": embed_dim,
        },
        checkpoint,
    )
    summary = TrainSummary(
        backend="rrb",
        variant=variant,
        model_name=model_name,
        pretrained=pretrained,
        device=str(scorer.device),
        train_examples=len(train_set),
        epochs=epochs,
        final_loss=final_loss,
        eval_accuracy=eval_correct / eval_total if eval_total else math.nan,
        eval_total=eval_total,
        eval_correct=eval_correct,
        eval_mean_margin=sum(eval_margins) / len(eval_margins) if eval_margins else math.nan,
        checkpoint=str(checkpoint),
        loss_mode=loss_mode,
        lambda_distill=lambda_distill,
        lambda_original=lambda_original,
        lambda_targeted=lambda_targeted,
        lambda_anchor=lambda_anchor,
        lambda_delta=lambda_delta,
        residual_alpha=residual_alpha,
        residual_gate=residual_gate,
        anchor_ratio=anchor_ratio,
        final_task_loss=final_task_loss,
        final_distill_loss=final_distill_loss,
        final_original_loss=final_original_loss,
        final_targeted_loss=final_targeted_loss,
        final_anchor_loss=final_anchor_loss,
        final_delta_loss=final_delta_loss,
        residual_relative_magnitude=drift_summary["residual_relative_magnitude"],
        z0_cosine_mean=drift_summary["z0_cosine_mean"],
        z0_cosine_min=drift_summary["z0_cosine_min"],
        lambda_rop=lambda_rop,
        rop_mode=rop_mode,
        protected_rank=int(protected_basis.shape[1]) if protected_basis is not None else 0,
        final_rop_loss=final_rop_loss,
        protected_residual_energy=(
            sum(drift_metrics["protected_residual_energy"]) / len(drift_metrics["protected_residual_energy"])
            if drift_metrics["protected_residual_energy"]
            else 0.0
        ),
    )
    (output_dir / "train_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _records_from_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _compact_histogram_stats(payload: Optional[dict], key: str) -> Dict[str, object]:
    if not payload:
        return {}
    stats = payload.get(key) or {}
    histogram = stats.get("histogram") or {}
    return {
        "mean": stats.get("mean"),
        "min": stats.get("min"),
        "max": stats.get("max"),
        "count": histogram.get("count"),
    }


def _failure_taxonomy(records: Sequence[dict]) -> List[dict]:
    buckets: Dict[Tuple[str, str], Dict[str, float]] = {}
    for record in records:
        key = (str(record.get("source", "")), str(record.get("edit_type", "")))
        bucket = buckets.setdefault(
            key,
            {"total": 0.0, "failures": 0.0, "margin_sum": 0.0, "failure_margin_sum": 0.0},
        )
        margin = float(record.get("margin", 0.0))
        bucket["total"] += 1
        bucket["margin_sum"] += margin
        if not record.get("correct"):
            bucket["failures"] += 1
            bucket["failure_margin_sum"] += margin

    rows = []
    for (source, edit_type), bucket in sorted(buckets.items()):
        total = bucket["total"]
        failures = bucket["failures"]
        rows.append(
            {
                "source": source,
                "edit_type": edit_type,
                "total": int(total),
                "failures": int(failures),
                "failure_rate": failures / total if total else math.nan,
                "mean_margin": bucket["margin_sum"] / total if total else math.nan,
                "mean_failure_margin": bucket["failure_margin_sum"] / failures if failures else None,
            }
        )
    return sorted(rows, key=lambda row: (-row["failure_rate"], -row["failures"], row["source"]))


def _case_view(record: dict) -> dict:
    return {
        "index": record.get("index"),
        "source": record.get("source"),
        "edit_type": record.get("edit_type"),
        "correct": record.get("correct"),
        "margin": record.get("margin"),
        "positive_score": record.get("positive_score"),
        "best_negative_score": record.get("best_negative_score"),
        "predicted_index": record.get("predicted_index"),
    }


def _write_diagnostics_report(output_dir: Path, summary: dict) -> Path:
    eval_summary = summary.get("eval_summary") or {}
    guardrails = summary.get("guardrails") or {}
    retrieval = guardrails.get("coco_retrieval") or {}
    imagenet = guardrails.get("imagenet_zeroshot") or {}
    drift = summary.get("drift") or {}
    residual = drift.get("residual_relative_magnitude") or {}
    cosine = drift.get("z0_cosine") or {}

    lines = [
        "# R030 Diagnostics",
        "",
        f"Selected run: `{summary.get('selected_run_id')}`",
        f"Checkpoint: `{summary.get('checkpoint')}`",
        f"Status: `{summary.get('status')}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Relation accuracy | {eval_summary.get('accuracy', '-'):.4f} |"
        if isinstance(eval_summary.get("accuracy"), (float, int))
        else "| Relation accuracy | - |",
        f"| COCO I2T R@1 | {retrieval.get('image_to_text_r1', '-'):.4f} |"
        if isinstance(retrieval.get("image_to_text_r1"), (float, int))
        else "| COCO I2T R@1 | - |",
        f"| ImageNet top-1 | {imagenet.get('top1', '-'):.4f} |"
        if isinstance(imagenet.get("top1"), (float, int))
        else "| ImageNet top-1 | - |",
        f"| Residual relative magnitude mean | {residual.get('mean', '-'):.4f} |"
        if isinstance(residual.get("mean"), (float, int))
        else "| Residual relative magnitude mean | - |",
        f"| z0 cosine mean | {cosine.get('mean', '-'):.4f} |"
        if isinstance(cosine.get("mean"), (float, int))
        else "| z0 cosine mean | - |",
        "",
        "## Failure Taxonomy",
        "",
        "| Source | Edit type | Failures | Total | Failure rate | Mean margin |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.get("failure_taxonomy", []):
        lines.append(
            "| {source} | {edit_type} | {failures} | {total} | {failure_rate:.4f} | {mean_margin:.4f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Hardest Failures",
            "",
            "| Index | Source | Edit type | Margin | Positive | Best negative |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for record in summary.get("hardest_failures", [])[:10]:
        lines.append(
            "| {index} | {source} | {edit_type} | {margin:.4f} | {positive_score:.4f} | {best_negative_score:.4f} |".format(
                **record
            )
        )

    lines.extend(
        [
            "",
            "## Closest Correct Cases",
            "",
            "| Index | Source | Edit type | Margin | Positive | Best negative |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for record in summary.get("closest_correct", [])[:10]:
        lines.append(
            "| {index} | {source} | {edit_type} | {margin:.4f} | {positive_score:.4f} | {best_negative_score:.4f} |".format(
                **record
            )
        )

    report_path = output_dir / "diagnostics_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_diagnostics(checkpoint: Optional[Path], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = checkpoint.parent if checkpoint else output_dir
    run_root = checkpoint_dir.parent
    selected_run_id = checkpoint_dir.name if checkpoint else None
    eval_summary = _read_json(checkpoint_dir / "checkpoint_eval_summary.json")
    train_summary = _read_json(checkpoint_dir / "train_summary.json")
    drift_summary = _read_json(checkpoint_dir / "rrb_drift_summary.json")
    guardrail_summary = _read_json(run_root / "guardrails" / checkpoint_dir.name / "guardrail_summary.json")
    records = _records_from_jsonl(checkpoint_dir / "eval_records.jsonl")
    failures = sorted((r for r in records if not r.get("correct")), key=lambda r: float(r.get("margin", 0.0)))
    correct = sorted((r for r in records if r.get("correct")), key=lambda r: float(r.get("margin", 0.0)))

    summary = {
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_exists": bool(checkpoint and checkpoint.exists()),
        "selected_run_id": selected_run_id,
        "status": "done" if checkpoint and checkpoint.exists() else "missing_checkpoint",
        "eval_summary": eval_summary,
        "guardrails": guardrail_summary,
        "train_summary": train_summary,
        "drift": {
            "residual_relative_magnitude": _compact_histogram_stats(
                drift_summary, "residual_relative_magnitude"
            ),
            "z0_cosine": _compact_histogram_stats(drift_summary, "z0_cosine"),
        },
        "failure_taxonomy": _failure_taxonomy(records),
        "hardest_failures": [_case_view(record) for record in failures[:25]],
        "closest_correct": [_case_view(record) for record in correct[:25]],
        "records_analyzed": len(records),
    }
    report_path = _write_diagnostics_report(output_dir, summary)
    summary["report_path"] = str(report_path)
    failures_path = output_dir / "selected_failure_cases.jsonl"
    with failures_path.open("w", encoding="utf-8") as handle:
        for record in summary["hardest_failures"]:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary["selected_failure_cases_path"] = str(failures_path)
    (output_dir / "diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
