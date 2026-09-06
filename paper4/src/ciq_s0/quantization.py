"""Deterministic group-wise symmetric RTN diagnostic quantization."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class QuantizerSpec:
    """Locked C01-S0 diagnostic quantizer definition."""

    bits: int = 4
    group_size: int = 128
    scale_dtype: str = "float32"
    symmetric: bool = True
    rounding: str = "round-to-nearest-even"
    clipping: str = "no percentile clipping; integer saturation only"
    qmin: int = -7
    qmax: int = 7

    def validate(self) -> None:
        """Reject any configuration outside the preregistered S0 definition."""
        expected = QuantizerSpec()
        if self != expected:
            raise ValueError(
                f"S0 quantizer must equal the locked specification: {expected}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class TensorQuantizationAudit:
    """Audit record for one in-place diagnostic quantization."""

    tensor_name: str
    parameter_count: int
    before_sha256: str
    after_sha256: str
    changed: bool
    source_dtype: str
    reconstructed_dtype: str
    max_absolute_error: float
    mean_absolute_error: float
    nonzero_error_count: int
    logical_packed_bytes: int
    scale_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash exact tensor storage bytes together with shape and dtype."""
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def logical_quantized_bytes(
    parameter_count: int, *, bits: int, group_size: int, scale_bytes: int = 4
) -> int:
    """Compute packed code plus per-group scale bytes for a tensor."""
    code_bytes = math.ceil(parameter_count * bits / 8)
    scale_count = math.ceil(parameter_count / group_size)
    return code_bytes + scale_count * scale_bytes


def quantize_dequantize_tensor(
    tensor: torch.Tensor, spec: QuantizerSpec
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize flattened contiguous groups and return reconstruction, codes, scales."""
    spec.validate()
    if not tensor.is_floating_point():
        raise TypeError("diagnostic quantization requires a floating-point tensor")
    if tensor.numel() == 0:
        return (
            tensor.clone(),
            torch.empty_like(tensor, dtype=torch.int8),
            torch.empty(0, dtype=torch.float32, device=tensor.device),
        )

    original_shape = tensor.shape
    flat = tensor.detach().to(torch.float32).reshape(-1)
    padded_count = math.ceil(flat.numel() / spec.group_size) * spec.group_size
    if padded_count != flat.numel():
        flat = torch.nn.functional.pad(flat, (0, padded_count - flat.numel()))
    groups = flat.reshape(-1, spec.group_size)
    max_abs = groups.abs().amax(dim=1)
    scales = max_abs / float(spec.qmax)
    safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    codes = torch.round(groups / safe_scales[:, None]).clamp(spec.qmin, spec.qmax)
    reconstructed = (codes * safe_scales[:, None]).reshape(-1)[: tensor.numel()]
    reconstructed = reconstructed.reshape(original_shape).to(tensor.dtype)
    return reconstructed, codes.to(torch.int8), scales.to(torch.float32)


def named_parameter_map(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    """Return a unique name-to-parameter map."""
    parameters = dict(model.named_parameters())
    if len(parameters) != sum(1 for _ in model.named_parameters()):
        raise RuntimeError("model contains duplicate named-parameter keys")
    return parameters


def parameter_hashes(
    model: torch.nn.Module, tensor_names: Iterable[str]
) -> dict[str, str]:
    """Hash a named subset of model parameters."""
    parameters = named_parameter_map(model)
    names = tuple(tensor_names)
    missing = sorted(set(names) - parameters.keys())
    if missing:
        raise KeyError(f"unknown tensor names: {missing[:5]}")
    return {name: tensor_sha256(parameters[name].data) for name in names}


@torch.no_grad()
def quantize_parameters_in_place(
    model: torch.nn.Module,
    tensor_names: Iterable[str],
    spec: QuantizerSpec,
    *,
    capture_original: bool,
) -> tuple[list[TensorQuantizationAudit], dict[str, torch.Tensor]]:
    """Apply diagnostic reconstruction to exactly the requested parameter names."""
    spec.validate()
    parameters = named_parameter_map(model)
    requested = tuple(tensor_names)
    if len(requested) != len(set(requested)):
        raise ValueError("tensor_names contains duplicates")
    missing = sorted(set(requested) - parameters.keys())
    if missing:
        raise KeyError(f"unknown tensor names: {missing[:5]}")

    audits: list[TensorQuantizationAudit] = []
    originals: dict[str, torch.Tensor] = {}
    for name in requested:
        parameter = parameters[name]
        before = parameter.data.detach().clone() if capture_original else parameter.data
        before_hash = tensor_sha256(parameter.data)
        reconstructed, _, scales = quantize_dequantize_tensor(parameter.data, spec)
        if capture_original:
            originals[name] = before
        absolute_error = (reconstructed.float() - parameter.data.float()).abs()
        parameter.data.copy_(reconstructed)
        after_hash = tensor_sha256(parameter.data)
        audits.append(
            TensorQuantizationAudit(
                tensor_name=name,
                parameter_count=parameter.numel(),
                before_sha256=before_hash,
                after_sha256=after_hash,
                changed=before_hash != after_hash,
                source_dtype=str(parameter.dtype),
                reconstructed_dtype=str(reconstructed.dtype),
                max_absolute_error=float(absolute_error.max().item()),
                mean_absolute_error=float(absolute_error.mean().item()),
                nonzero_error_count=int(torch.count_nonzero(absolute_error).item()),
                logical_packed_bytes=logical_quantized_bytes(
                    parameter.numel(), bits=spec.bits, group_size=spec.group_size
                ),
                scale_count=int(scales.numel()),
            )
        )
    return audits, originals


@torch.no_grad()
def restore_parameters(
    model: torch.nn.Module, originals: Mapping[str, torch.Tensor]
) -> None:
    """Restore an exact captured tensor snapshot into a model."""
    parameters = named_parameter_map(model)
    missing = sorted(set(originals) - parameters.keys())
    if missing:
        raise KeyError(f"cannot restore unknown tensor names: {missing[:5]}")
    for name, original in originals.items():
        target = parameters[name]
        if target.shape != original.shape or target.dtype != original.dtype:
            raise ValueError(f"restore metadata mismatch for {name}")
        target.data.copy_(original.to(target.device))
