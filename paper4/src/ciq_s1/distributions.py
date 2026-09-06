"""Memory-bounded gold-position distribution metrics for C01-S1."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FixedSupportDistribution:
    """Per-position top-k token probabilities plus one OTHER mass."""

    token_ids: list[list[int]]
    probabilities: list[list[float]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def build_fixed_support_distribution(
    logits: torch.Tensor, *, top_k: int
) -> FixedSupportDistribution:
    """Freeze the BF16 top-k support and aggregate all remaining probability mass."""
    if logits.ndim != 2:
        raise ValueError("gold-position logits must have shape [positions, vocabulary]")
    if not 0 < top_k < logits.shape[-1]:
        raise ValueError("top_k must be positive and smaller than the vocabulary")
    probabilities = torch.softmax(logits.float(), dim=-1)
    top_probabilities, top_ids = torch.topk(probabilities, k=top_k, dim=-1, sorted=True)
    other = (1.0 - top_probabilities.sum(dim=-1, keepdim=True)).clamp_min(0.0)
    compact = torch.cat([top_probabilities, other], dim=-1)
    compact = compact / compact.sum(dim=-1, keepdim=True)
    return FixedSupportDistribution(
        token_ids=top_ids.cpu().tolist(),
        probabilities=compact.cpu().tolist(),
    )


def project_logits_to_fixed_support(
    logits: torch.Tensor, reference: FixedSupportDistribution
) -> torch.Tensor:
    """Project condition logits onto BF16's fixed top-k support plus OTHER."""
    if logits.ndim != 2 or logits.shape[0] != len(reference.token_ids):
        raise ValueError("condition logits do not match reference answer positions")
    token_ids = torch.tensor(
        reference.token_ids, device=logits.device, dtype=torch.long
    )
    probabilities = torch.softmax(logits.float(), dim=-1)
    selected = probabilities.gather(dim=-1, index=token_ids)
    other = (1.0 - selected.sum(dim=-1, keepdim=True)).clamp_min(0.0)
    compact = torch.cat([selected, other], dim=-1)
    return compact / compact.sum(dim=-1, keepdim=True)


def jensen_shannon_divergence(
    first: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    """Compute row-wise natural-log Jensen-Shannon divergence."""
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("JS inputs must have identical [positions, support] shapes")
    if torch.any(first < 0) or torch.any(second < 0):
        raise ValueError("probabilities cannot be negative")
    first = first.float() / first.float().sum(dim=-1, keepdim=True)
    second = second.float() / second.float().sum(dim=-1, keepdim=True)
    midpoint = 0.5 * (first + second)

    def kl(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        terms = torch.where(
            left > 0,
            left
            * (
                torch.log(left)
                - torch.log(right.clamp_min(torch.finfo(right.dtype).tiny))
            ),
            torch.zeros_like(left),
        )
        return terms.sum(dim=-1)

    values = 0.5 * kl(first, midpoint) + 0.5 * kl(second, midpoint)
    if torch.any(~torch.isfinite(values)):
        raise RuntimeError("JS divergence is non-finite")
    return values


def compare_to_fixed_support(
    logits: torch.Tensor, reference: FixedSupportDistribution
) -> tuple[float, list[float]]:
    """Return mean and position-wise JS divergence against frozen BF16 support."""
    projected = project_logits_to_fixed_support(logits, reference)
    reference_values = torch.tensor(
        reference.probabilities, device=logits.device, dtype=torch.float32
    )
    divergences = jensen_shannon_divergence(reference_values, projected)
    mean_value = float(divergences.mean().item())
    if not math.isfinite(mean_value):
        raise RuntimeError("mean JS divergence is non-finite")
    return mean_value, divergences.cpu().tolist()
