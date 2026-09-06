"""Teacher-forced target alignment and cache-key tests."""

from types import SimpleNamespace

import torch
import pytest

from src.ciq_s0.inference import (
    answer_only_nll,
    append_answer_labels,
    configure_determinism,
)
from src.ciq_s0.vision_cache import VisionCacheKey, VisionFeatureCache


class _Tokenizer:
    eos_token = "<eos>"

    def __call__(
        self, text: str, *, add_special_tokens: bool, return_tensors: str
    ) -> dict[str, torch.Tensor]:
        assert not add_special_tokens
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[3, 4]], dtype=torch.long)}


class _PerfectNextTokenModel(torch.nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        use_cache: bool,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        logits = torch.full((*input_ids.shape, 8), -20.0)
        for position in range(input_ids.shape[1] - 1):
            logits[:, position, input_ids[:, position + 1]] = 20.0
        return SimpleNamespace(logits=logits)


def test_answer_labels_mask_prompt_and_align_next_tokens() -> None:
    prompt = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
    }
    full, labels, prompt_length = append_answer_labels(
        prompt, tokenizer=_Tokenizer(), answer="cat"
    )
    assert prompt_length == 2
    assert labels.tolist() == [[-100, -100, 3, 4]]
    assert full["input_ids"].tolist() == [[1, 2, 3, 4]]
    assert answer_only_nll(_PerfectNextTokenModel(), full, labels, prompt_length) < 1e-6


def test_vision_cache_key_includes_precision_fingerprint() -> None:
    cache = VisionFeatureCache()
    bf16 = VisionCacheKey("image", "preprocess", "bf16-weights")
    w4 = VisionCacheKey("image", "preprocess", "w4-weights")
    features = torch.tensor([1.0, 2.0])
    cache.put(bf16, features)
    assert torch.equal(cache.get(bf16), features)
    assert cache.get(w4) is None


def test_determinism_requires_cublas_workspace_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        configure_determinism(17)


def test_determinism_records_cublas_workspace_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    settings = configure_determinism(17)
    assert settings["cublas_workspace_config"] == ":4096:8"
