"""Runtime discovery of Qwen2-VL non-KV module groups."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .io_utils import sha256_json
from .quantization import logical_quantized_bytes

_VISION_BLOCK = re.compile(r"(?:^|\.)visual\.blocks\.(\d+)\.")
_VISION_MERGER = re.compile(r"(?:^|\.)visual\.merger\.")
_DECODER_LAYER = re.compile(r"(?:^|\.)model(?:\.language_model)?\.layers\.(\d+)\.")


@dataclass(frozen=True)
class TensorManifestRecord:
    """Storage and membership metadata for one model parameter."""

    tensor_name: str
    module_name: str
    shape: tuple[int, ...]
    dtype: str
    parameter_count: int
    bf16_bytes: int
    w8_bytes: int
    w4_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        value = asdict(self)
        value["shape"] = list(self.shape)
        return value


@dataclass(frozen=True)
class GroupManifestRecord:
    """A hardware-aligned coarse C01 module group."""

    group_id: str
    family: str
    depth_quartile: int | None
    module_names: tuple[str, ...]
    tensor_names: tuple[str, ...]
    parameter_count: int
    bf16_bytes: int
    w8_bytes: int
    w4_bytes: int
    excluded_tensors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        value = asdict(self)
        value["module_names"] = list(self.module_names)
        value["tensor_names"] = list(self.tensor_names)
        value["excluded_tensors"] = list(self.excluded_tensors)
        return value


def _quartile(index: int, layer_count: int) -> int:
    if layer_count <= 0 or not 0 <= index < layer_count:
        raise ValueError("invalid layer index/count")
    return min(4, math.floor(index * 4 / layer_count) + 1)


def _module_name(tensor_name: str) -> str:
    return tensor_name.rsplit(".", maxsplit=1)[0]


def _eligible_weight(name: str, parameter: torch.nn.Parameter) -> bool:
    lowered = name.lower()
    return (
        name.endswith(".weight")
        and parameter.is_floating_point()
        and parameter.ndim >= 2
        and "norm" not in lowered
        and "embed" not in lowered
        and "rotary" not in lowered
        and "lm_head" not in lowered
    )


def _discover_layer_counts(names: list[str]) -> tuple[int, int]:
    vision_indices = {
        int(match.group(1))
        for name in names
        if (match := _VISION_BLOCK.search(name)) is not None
    }
    decoder_indices = {
        int(match.group(1))
        for name in names
        if (match := _DECODER_LAYER.search(name)) is not None
    }
    if not vision_indices or vision_indices != set(range(max(vision_indices) + 1)):
        raise RuntimeError(
            "actual model has no contiguous visual.blocks layer structure"
        )
    if not decoder_indices or decoder_indices != set(range(max(decoder_indices) + 1)):
        raise RuntimeError(
            "actual model has no contiguous model.layers decoder structure"
        )
    return len(vision_indices), len(decoder_indices)


def _classify_parameter(
    name: str, *, vision_layer_count: int, decoder_layer_count: int
) -> tuple[str, str, int | None] | None:
    vision = _VISION_BLOCK.search(name)
    if vision is not None:
        layer = int(vision.group(1))
        quartile = _quartile(layer, vision_layer_count)
        return f"vision.q{quartile}", "vision_encoder", quartile
    if _VISION_MERGER.search(name) is not None:
        return "bridge.merger", "multimodal_bridge", None
    decoder = _DECODER_LAYER.search(name)
    if decoder is None:
        return None
    layer = int(decoder.group(1))
    quartile = _quartile(layer, decoder_layer_count)
    if ".self_attn." in name:
        return f"decoder.q{quartile}.attention", "decoder_attention", quartile
    if ".mlp." in name:
        return f"decoder.q{quartile}.mlp", "decoder_mlp", quartile
    return None


def build_module_manifest(model: torch.nn.Module) -> dict[str, Any]:
    """Discover actual Qwen2-VL groups and return a self-hashed manifest."""
    parameters = list(model.named_parameters())
    names = [name for name, _ in parameters]
    vision_layer_count, decoder_layer_count = _discover_layer_counts(names)
    grouped_tensors: dict[str, list[TensorManifestRecord]] = defaultdict(list)
    grouped_exclusions: dict[str, list[str]] = defaultdict(list)
    group_metadata: dict[str, tuple[str, int | None]] = {}
    global_exclusions: list[dict[str, str]] = []

    for name, parameter in parameters:
        classification = _classify_parameter(
            name,
            vision_layer_count=vision_layer_count,
            decoder_layer_count=decoder_layer_count,
        )
        if classification is None:
            global_exclusions.append(
                {"tensor_name": name, "reason": "outside preregistered non-KV groups"}
            )
            continue
        group_id, family, quartile = classification
        group_metadata[group_id] = family, quartile
        if not _eligible_weight(name, parameter):
            grouped_exclusions[group_id].append(name)
            continue
        parameter_count = parameter.numel()
        grouped_tensors[group_id].append(
            TensorManifestRecord(
                tensor_name=name,
                module_name=_module_name(name),
                shape=tuple(parameter.shape),
                dtype=str(parameter.dtype),
                parameter_count=parameter_count,
                bf16_bytes=parameter_count * 2,
                w8_bytes=logical_quantized_bytes(
                    parameter_count, bits=8, group_size=128
                ),
                w4_bytes=logical_quantized_bytes(
                    parameter_count, bits=4, group_size=128
                ),
            )
        )

    groups: list[GroupManifestRecord] = []
    tensor_records: list[dict[str, Any]] = []
    for group_id in sorted(group_metadata):
        tensors = sorted(grouped_tensors[group_id], key=lambda item: item.tensor_name)
        if not tensors:
            raise RuntimeError(f"discovered group {group_id} has no eligible tensors")
        family, quartile = group_metadata[group_id]
        record = GroupManifestRecord(
            group_id=group_id,
            family=family,
            depth_quartile=quartile,
            module_names=tuple(sorted({item.module_name for item in tensors})),
            tensor_names=tuple(item.tensor_name for item in tensors),
            parameter_count=sum(item.parameter_count for item in tensors),
            bf16_bytes=sum(item.bf16_bytes for item in tensors),
            w8_bytes=sum(item.w8_bytes for item in tensors),
            w4_bytes=sum(item.w4_bytes for item in tensors),
            excluded_tensors=tuple(sorted(grouped_exclusions[group_id])),
        )
        groups.append(record)
        tensor_records.extend(item.to_dict() for item in tensors)

    expected_group_ids = {
        "bridge.merger",
        *(f"vision.q{index}" for index in range(1, 5)),
        *(f"decoder.q{index}.attention" for index in range(1, 5)),
        *(f"decoder.q{index}.mlp" for index in range(1, 5)),
    }
    actual_group_ids = {group.group_id for group in groups}
    if actual_group_ids != expected_group_ids:
        raise RuntimeError(
            "runtime group discovery does not match the locked 13-group policy: "
            f"missing={sorted(expected_group_ids - actual_group_ids)}, "
            f"extra={sorted(actual_group_ids - expected_group_ids)}"
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "discovery_policy": {
            "vision": "actual visual.blocks layers split into depth quartiles",
            "bridge": "actual visual.merger parameters",
            "decoder_attention": "actual model[.language_model].layers.*.self_attn weights by quartile",
            "decoder_mlp": "actual model[.language_model].layers.*.mlp weights by quartile",
            "eligible": "floating .weight tensors with ndim >= 2",
            "excluded": "embedding, normalization, RoPE, output head, biases, and all other tensors",
        },
        "vision_layer_count": vision_layer_count,
        "decoder_layer_count": decoder_layer_count,
        "groups": [group.to_dict() for group in groups],
        "tensors": tensor_records,
        "global_excluded_tensors": sorted(
            global_exclusions, key=lambda item: item["tensor_name"]
        ),
    }
    payload["manifest_sha256"] = sha256_json(payload)
    return payload


def group_tensor_names(manifest: dict[str, Any], group_id: str) -> tuple[str, ...]:
    """Extract one group's tensor names from a validated manifest payload."""
    matching = [group for group in manifest["groups"] if group["group_id"] == group_id]
    if len(matching) != 1:
        raise KeyError(f"manifest does not contain exactly one group {group_id}")
    return tuple(str(name) for name in matching[0]["tensor_names"])


def all_quantizable_tensor_names(manifest: dict[str, Any]) -> tuple[str, ...]:
    """Return every quantizable tensor name in deterministic order."""
    return tuple(str(item["tensor_name"]) for item in manifest["tensors"])
