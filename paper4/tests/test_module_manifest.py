"""Tests for runtime-discovered 13-group module manifests."""

import torch

from src.ciq_s0.module_manifest import (
    all_quantizable_tensor_names,
    build_module_manifest,
    group_tensor_names,
)


class _VisionBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = torch.nn.Linear(8, 8, bias=False)
        self.mlp = torch.nn.Linear(8, 8, bias=False)
        self.norm = torch.nn.LayerNorm(8)


class _DecoderBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = torch.nn.Linear(8, 8, bias=False)
        self.mlp = torch.nn.Linear(8, 8, bias=False)
        self.input_layernorm = torch.nn.LayerNorm(8)


class _ToyQwen(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = torch.nn.Module()
        self.visual.blocks = torch.nn.ModuleList([_VisionBlock() for _ in range(4)])
        self.visual.merger = torch.nn.Linear(8, 8, bias=True)
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList(
            [_DecoderBlock() for _ in range(4)]
        )
        self.model.language_model.embed_tokens = torch.nn.Embedding(16, 8)
        self.lm_head = torch.nn.Linear(8, 16, bias=False)


def test_manifest_discovers_locked_groups_without_embeddings_or_norms() -> None:
    manifest = build_module_manifest(_ToyQwen().to(torch.bfloat16))
    assert len(manifest["groups"]) == 13
    names = all_quantizable_tensor_names(manifest)
    assert names
    assert not any("norm" in name for name in names)
    assert not any("embed" in name for name in names)
    assert not any("lm_head" in name for name in names)
    assert group_tensor_names(manifest, "decoder.q1.attention") == (
        "model.language_model.layers.0.self_attn.weight",
    )
    assert len(manifest["manifest_sha256"]) == 64
