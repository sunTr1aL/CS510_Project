"""Latent relational bottleneck modules for CLIP token features.

This file is intentionally independent of OpenCLIP. It consumes token tensors
from any encoder with shape ``[batch, tokens, dim]`` and returns a normalized
embedding plus responsibility maps for targeted intervention losses.
"""

from __future__ import annotations

from typing import Optional

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - lets non-training utilities import.
    torch = None
    Tensor = object
    nn = object
    F = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for relbottleneck modules")


if torch is not None:

    class LatentRelationalBottleneck(nn.Module):
        """Slot bottleneck with one slot self-attention block."""

        def __init__(
            self,
            token_dim: int,
            embed_dim: int,
            slot_count: int = 8,
            num_heads: int = 4,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.slot_count = slot_count
            self.slot_queries = nn.Parameter(torch.randn(slot_count, token_dim) * 0.02)
            self.cross_attn = nn.MultiheadAttention(
                token_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.slot_attn = nn.MultiheadAttention(
                token_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm1 = nn.LayerNorm(token_dim)
            self.norm2 = nn.LayerNorm(token_dim)
            self.readout = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, embed_dim),
            )

        def forward(self, tokens: Tensor) -> dict:
            batch = tokens.shape[0]
            queries = self.slot_queries.unsqueeze(0).expand(batch, -1, -1)
            slots, token_resp = self.cross_attn(
                queries,
                tokens,
                tokens,
                need_weights=True,
                average_attn_weights=True,
            )
            slots = self.norm1(slots + queries)
            related, slot_affinity = self.slot_attn(
                slots,
                slots,
                slots,
                need_weights=True,
                average_attn_weights=True,
            )
            slots = self.norm2(slots + related)
            pooled = slots.mean(dim=1)
            embedding = F.normalize(self.readout(pooled), dim=-1)
            return {
                "embedding": embedding,
                "slots": slots,
                "token_responsibility": token_resp,
                "slot_affinity": slot_affinity,
            }


    class RelBottleneckFusion(nn.Module):
        """Fuse a frozen/global CLIP summary with the bottleneck readout."""

        def __init__(
            self,
            token_dim: int,
            clip_embed_dim: int,
            slot_count: int = 8,
            num_heads: int = 4,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.bottleneck = LatentRelationalBottleneck(
                token_dim=token_dim,
                embed_dim=clip_embed_dim,
                slot_count=slot_count,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.global_proj = nn.Linear(token_dim, clip_embed_dim)
            self.fuse = nn.Linear(clip_embed_dim * 2, clip_embed_dim)

        def forward(self, tokens: Tensor, global_token: Optional[Tensor] = None) -> dict:
            if global_token is None:
                global_token = tokens[:, 0]
            bottleneck_out = self.bottleneck(tokens)
            global_embedding = F.normalize(self.global_proj(global_token), dim=-1)
            fused = torch.cat([global_embedding, bottleneck_out["embedding"]], dim=-1)
            bottleneck_out["embedding"] = F.normalize(self.fuse(fused), dim=-1)
            bottleneck_out["global_embedding"] = global_embedding
            return bottleneck_out

    class RelResidualBottleneck(nn.Module):
        """Parser-free relational slots used as a bounded residual correction."""

        def __init__(
            self,
            token_dim: int,
            clip_embed_dim: int,
            slot_count: int = 8,
            num_heads: int = 4,
            dropout: float = 0.0,
            residual_alpha: float = 0.05,
            gate_type: str = "fixed",
        ) -> None:
            super().__init__()
            if gate_type not in {"fixed", "scalar", "vector"}:
                raise ValueError(f"Unsupported gate_type: {gate_type}")
            self.bottleneck = LatentRelationalBottleneck(
                token_dim=token_dim,
                embed_dim=clip_embed_dim,
                slot_count=slot_count,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.delta = nn.Linear(clip_embed_dim, clip_embed_dim)
            nn.init.normal_(self.delta.weight, std=1e-3)
            nn.init.zeros_(self.delta.bias)
            self.gate_type = gate_type
            self.max_alpha = float(residual_alpha)
            if gate_type == "fixed":
                self.register_buffer("fixed_gate", torch.tensor(float(residual_alpha)))
            else:
                shape = (clip_embed_dim,) if gate_type == "vector" else (1,)
                # sigmoid(0)=0.5, so learned gates start at half of max_alpha.
                self.gate_logits = nn.Parameter(torch.zeros(shape))

        def gate(self) -> Tensor:
            if self.gate_type == "fixed":
                return self.fixed_gate
            return torch.sigmoid(self.gate_logits) * self.max_alpha

        def forward(self, tokens: Tensor, base_embedding: Tensor) -> dict:
            bottleneck_out = self.bottleneck(tokens)
            delta_rel = self.delta(bottleneck_out["embedding"])
            gate = self.gate()
            residual = gate * delta_rel
            embedding = F.normalize(base_embedding + residual, dim=-1)
            residual_relative_magnitude = residual.norm(dim=-1) / base_embedding.norm(dim=-1).clamp_min(1e-6)
            z0_cosine = F.cosine_similarity(base_embedding, embedding, dim=-1)
            bottleneck_out.update(
                {
                    "embedding": embedding,
                    "base_embedding": base_embedding,
                    "delta_rel": delta_rel,
                    "residual": residual,
                    "gate": gate,
                    "residual_relative_magnitude": residual_relative_magnitude,
                    "z0_cosine": z0_cosine,
                }
            )
            return bottleneck_out


else:

    class LatentRelationalBottleneck:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()


    class RelBottleneckFusion:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()


    class RelResidualBottleneck:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()


def targeted_slot_loss(
    slots_text,
    slots_cf,
    token_responsibility,
    edited_token_mask,
    keep_token_mask,
    tau_sep: float = 0.4,
    lambda_sep: float = 1.0,
    lambda_keep: float = 1.0,
):
    """Compute the text-side targeted intervention loss.

    Masks are expected to have shape ``[batch, tokens]`` and responsibility has
    shape ``[batch, slots, tokens]``.
    """

    _require_torch()
    slot_cos = F.cosine_similarity(slots_text, slots_cf, dim=-1)
    edited_weights = torch.matmul(
        token_responsibility,
        edited_token_mask.float().unsqueeze(-1),
    ).squeeze(-1)
    keep_weights = torch.matmul(
        token_responsibility,
        keep_token_mask.float().unsqueeze(-1),
    ).squeeze(-1)

    edited_weights = edited_weights / edited_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    keep_weights = keep_weights / keep_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    sep = edited_weights * torch.relu(slot_cos - tau_sep)
    keep = keep_weights * (1.0 - slot_cos)
    return lambda_sep * sep.sum(dim=-1).mean() + lambda_keep * keep.sum(dim=-1).mean()
