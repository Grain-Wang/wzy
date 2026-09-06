"""Tests for S1 W8 restoration and exact-byte profile construction."""

from __future__ import annotations

import torch

from src.ciq_s0.quantization import QuantizerSpec, quantize_dequantize_tensor
from src.ciq_s1.precision import (
    W8RestorationSpec,
    apply_restoration_profile,
    profile_logical_bytes,
    quantize_dequantize_w8,
    reset_profile_to_w4,
)
from src.ciq_s1.profiles import (
    additive_gain_for_assignments,
    all_w4_logical_bytes,
    construct_profile,
    deterministic_random_gains,
    restoration_budget_bytes,
)


def _manifest() -> dict[str, object]:
    return {
        "groups": [
            {
                "group_id": "g1",
                "tensor_names": ["first.weight"],
                "w4_bytes": 100,
                "w8_bytes": 140,
                "bf16_bytes": 220,
            },
            {
                "group_id": "g2",
                "tensor_names": ["second.weight"],
                "w4_bytes": 120,
                "w8_bytes": 160,
                "bf16_bytes": 240,
            },
        ]
    }


def test_s0_w4_spec_remains_locked() -> None:
    QuantizerSpec().validate()
    try:
        QuantizerSpec(bits=8).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("S0 accepted a non-W4 quantizer")


def test_w8_is_deterministic_and_has_lower_error_than_w4() -> None:
    tensor = torch.linspace(-1.0, 1.0, 257, dtype=torch.bfloat16)
    first, first_codes, _ = quantize_dequantize_w8(tensor, W8RestorationSpec())
    second, second_codes, _ = quantize_dequantize_w8(tensor, W8RestorationSpec())
    w4, _, _ = quantize_dequantize_tensor(tensor, QuantizerSpec())
    assert torch.equal(first, second)
    assert torch.equal(first_codes, second_codes)
    assert (first.float() - tensor.float()).abs().mean() < (
        w4.float() - tensor.float()
    ).abs().mean()


def test_profile_application_changes_only_protected_tensors_and_restores_w4() -> None:
    model = torch.nn.Module()
    model.first = torch.nn.Linear(16, 2, bias=False)
    model.second = torch.nn.Linear(16, 2, bias=False)
    originals = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    for name, parameter in model.named_parameters():
        reconstructed, _, _ = quantize_dequantize_tensor(
            originals[name], QuantizerSpec()
        )
        parameter.data.copy_(reconstructed)
    first_w4 = model.first.weight.detach().clone()
    second_before = model.second.weight.detach().clone()
    profile = {"g1": "w8"}
    apply_restoration_profile(
        model,
        originals,
        _manifest(),
        profile,
        w8_spec=W8RestorationSpec(),
    )
    assert not torch.equal(model.first.weight, first_w4)
    assert torch.equal(model.second.weight, second_before)
    reset_profile_to_w4(model, originals, _manifest(), profile)
    expected, _, _ = quantize_dequantize_tensor(
        originals["first.weight"], QuantizerSpec()
    )
    assert torch.equal(model.first.weight, expected)


def test_logical_byte_accounting_and_deterministic_budget_packing() -> None:
    manifest = _manifest()
    assert all_w4_logical_bytes(manifest) == 220
    assert profile_logical_bytes(manifest, {"g1": "w8"}) == 260
    budget = restoration_budget_bytes(manifest, 0.20)
    profile = construct_profile(
        manifest,
        {"g1": 2.0, "g2": 1.0},
        budget_fraction=0.20,
        seed=5,
    )
    assert profile.added_bytes <= budget
    assert profile.logical_bytes == profile_logical_bytes(manifest, profile.assignments)
    assert deterministic_random_gains(
        ["g1", "g2"], seed=2, key="x"
    ) == deterministic_random_gains(["g1", "g2"], seed=2, key="x")


def test_additive_gain_uses_measured_gain_instead_of_priority_score() -> None:
    assert (
        additive_gain_for_assignments(
            {"g1": "w8", "g2": "bf16"}, {"g1": 2.0, "g2": 3.0}
        )
        == 4.0
    )
