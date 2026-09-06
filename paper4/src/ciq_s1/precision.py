"""S1-only W8 restoration while preserving the locked S0 W4 implementation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch

from ..ciq_s0.module_manifest import group_tensor_names
from ..ciq_s0.quantization import (
    QuantizerSpec,
    named_parameter_map,
    quantize_dequantize_tensor,
)


@dataclass(frozen=True)
class W8RestorationSpec:
    """Locked C01-S1 symmetric W8 diagnostic restoration definition."""

    bits: int = 8
    group_size: int = 128
    scale_dtype: str = "float32"
    symmetric: bool = True
    rounding: str = "round-to-nearest-even"
    clipping: str = "no percentile clipping; integer saturation only"
    qmin: int = -127
    qmax: int = 127

    def validate(self) -> None:
        """Reject configurations outside the locked S1 W8 definition."""
        expected = W8RestorationSpec()
        if self != expected:
            raise ValueError(f"S1 W8 spec must equal the locked definition: {expected}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def quantize_dequantize_w8(
    tensor: torch.Tensor, spec: W8RestorationSpec
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deterministic W8 reconstruction, integer codes, and FP32 scales."""
    spec.validate()
    if not tensor.is_floating_point():
        raise TypeError("W8 restoration requires a floating-point tensor")
    if tensor.numel() == 0:
        return (
            tensor.clone(),
            torch.empty_like(tensor, dtype=torch.int8),
            torch.empty(0, dtype=torch.float32, device=tensor.device),
        )
    original_shape = tensor.shape
    flat = tensor.detach().float().reshape(-1)
    padded_count = math.ceil(flat.numel() / spec.group_size) * spec.group_size
    if padded_count != flat.numel():
        flat = torch.nn.functional.pad(flat, (0, padded_count - flat.numel()))
    grouped = flat.reshape(-1, spec.group_size)
    scales = grouped.abs().amax(dim=1) / float(spec.qmax)
    safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    codes = torch.round(grouped / safe_scales[:, None]).clamp(spec.qmin, spec.qmax)
    reconstructed = (codes * safe_scales[:, None]).reshape(-1)[: tensor.numel()]
    return (
        reconstructed.reshape(original_shape).to(tensor.dtype),
        codes.to(torch.int8),
        scales.float(),
    )


def profile_logical_bytes(
    manifest: Mapping[str, Any], profile: Mapping[str, str]
) -> int:
    """Compute exact logical resident bytes for an all-W4 restoration profile."""
    total = 0
    for group in manifest["groups"]:
        precision = profile.get(str(group["group_id"]), "w4")
        if precision not in {"w4", "w8", "bf16"}:
            raise ValueError(f"unsupported precision {precision}")
        total += int(group[f"{precision}_bytes"])
    unknown = set(profile) - {str(group["group_id"]) for group in manifest["groups"]}
    if unknown:
        raise KeyError(f"profile contains unknown groups: {sorted(unknown)}")
    return total


@torch.no_grad()
def apply_restoration_profile(
    model: torch.nn.Module,
    originals: Mapping[str, torch.Tensor],
    manifest: dict[str, Any],
    profile: Mapping[str, str],
    *,
    w8_spec: W8RestorationSpec,
) -> None:
    """Restore selected all-W4 groups to W8 or exact BF16 originals."""
    parameters = named_parameter_map(model)
    for group_id, precision in profile.items():
        if precision == "w4":
            continue
        for name in group_tensor_names(manifest, group_id):
            original = originals[name]
            if precision == "bf16":
                reconstructed = original
            elif precision == "w8":
                reconstructed, _, _ = quantize_dequantize_w8(original, w8_spec)
            else:
                raise ValueError(f"unsupported restoration precision {precision}")
            parameters[name].data.copy_(reconstructed)


@torch.no_grad()
def reset_profile_to_w4(
    model: torch.nn.Module,
    originals: Mapping[str, torch.Tensor],
    manifest: dict[str, Any],
    profile: Mapping[str, str],
) -> None:
    """Return protected tensors to the exact locked S0 W4 reconstruction."""
    parameters = named_parameter_map(model)
    w4_spec = QuantizerSpec()
    for group_id, precision in profile.items():
        if precision == "w4":
            continue
        for name in group_tensor_names(manifest, group_id):
            reconstructed, _, _ = quantize_dequantize_tensor(originals[name], w4_spec)
            parameters[name].data.copy_(reconstructed)
