"""Diagnostic W4 quantizer isolation and restoration tests."""

import torch

from src.ciq_s0.quantization import (
    QuantizerSpec,
    parameter_hashes,
    quantize_dequantize_tensor,
    quantize_parameters_in_place,
    restore_parameters,
    tensor_sha256,
)


def test_quantizer_is_deterministic_and_preserves_metadata() -> None:
    tensor = torch.linspace(-2.0, 2.0, 259, dtype=torch.bfloat16).reshape(7, 37)
    first, first_codes, first_scales = quantize_dequantize_tensor(
        tensor, QuantizerSpec()
    )
    second, second_codes, second_scales = quantize_dequantize_tensor(
        tensor, QuantizerSpec()
    )
    assert first.shape == tensor.shape
    assert first.dtype == tensor.dtype
    assert torch.equal(first, second)
    assert torch.equal(first_codes, second_codes)
    assert torch.equal(first_scales, second_scales)
    assert first_codes.dtype == torch.int8
    assert first_scales.dtype == torch.float32
    assert int(first_codes.min()) >= -7
    assert int(first_codes.max()) <= 7


def test_zero_group_has_finite_zero_reconstruction() -> None:
    tensor = torch.zeros(129, dtype=torch.bfloat16)
    reconstructed, codes, scales = quantize_dequantize_tensor(tensor, QuantizerSpec())
    assert torch.equal(reconstructed, tensor)
    assert torch.count_nonzero(codes) == 0
    assert torch.count_nonzero(scales) == 0


def test_in_place_isolation_and_exact_restore() -> None:
    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 8, bias=False),
        torch.nn.Linear(8, 4, bias=False),
    ).to(torch.bfloat16)
    names = tuple(dict(model.named_parameters()))
    before = parameter_hashes(model, names)
    audits, originals = quantize_parameters_in_place(
        model, (names[0],), QuantizerSpec(), capture_original=True
    )
    after = parameter_hashes(model, names)
    assert audits[0].changed
    assert before[names[0]] != after[names[0]]
    assert before[names[1]] == after[names[1]]
    restore_parameters(model, originals)
    assert parameter_hashes(model, names) == before


def test_tensor_hash_includes_dtype_and_shape() -> None:
    values = torch.tensor([1.0, 2.0], dtype=torch.float32)
    assert tensor_sha256(values) != tensor_sha256(values.to(torch.bfloat16))
    assert tensor_sha256(values) != tensor_sha256(values.reshape(1, 2))
