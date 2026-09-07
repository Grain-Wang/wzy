"""Deterministic S1 inference with fixed-support gold-position distributions."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from PIL import Image

from ..ciq_s0.inference import (
    append_answer_labels,
    generate_answer,
    prepare_prompt_inputs,
)

from .distributions import (
    FixedSupportDistribution,
    build_fixed_support_distribution,
    compare_to_fixed_support,
)


@dataclass(frozen=True)
class S1InferenceResult:
    """Generation and teacher-forced statistics for one precision condition."""

    prediction: str
    answer_nll: float
    prompt_token_count: int
    answer_token_count: int
    target_token_sha256: str
    reference_distribution: FixedSupportDistribution | None
    js_divergence: float | None
    position_js_divergences: list[float] | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        value = asdict(self)
        if self.reference_distribution is not None:
            value["reference_distribution"] = self.reference_distribution.to_dict()
        return value


def _token_sha256(token_ids: torch.Tensor) -> str:
    values = token_ids.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


@torch.inference_mode()
def run_s1_inference_case(
    model: torch.nn.Module,
    processor: Any,
    *,
    image: Image.Image,
    question: str,
    answer: str,
    instruction: str,
    device: torch.device,
    generation_settings: dict[str, Any],
    distribution_top_k: int,
    bf16_reference: FixedSupportDistribution | None,
) -> S1InferenceResult:
    """Run one case and either freeze or compare its compact distribution."""
    prompt_inputs = prepare_prompt_inputs(
        processor,
        image=image,
        question=question,
        instruction=instruction,
        device=device,
    )
    return run_s1_prepared_case(
        model,
        processor,
        prompt_inputs=prompt_inputs,
        answer=answer,
        generation_settings=generation_settings,
        distribution_top_k=distribution_top_k,
        bf16_reference=bf16_reference,
    )


@torch.inference_mode()
def run_s1_prepared_case(
    model: torch.nn.Module,
    processor: Any,
    *,
    prompt_inputs: dict[str, torch.Tensor],
    answer: str,
    generation_settings: dict[str, Any],
    distribution_top_k: int,
    bf16_reference: FixedSupportDistribution | None,
) -> S1InferenceResult:
    """Evaluate generation, answer NLL, and JS from prepared prompt tensors."""
    prediction = generate_answer(model, processor, prompt_inputs, generation_settings)
    full_inputs, labels, prompt_length = append_answer_labels(
        prompt_inputs, tokenizer=processor.tokenizer, answer=answer
    )
    outputs = model(**full_inputs, use_cache=False)
    answer_logits = outputs.logits[:, prompt_length - 1 : -1, :].float().squeeze(0)
    targets = labels[:, prompt_length:].squeeze(0)
    if answer_logits.shape[0] != targets.shape[0]:
        raise AssertionError("S1 answer logit/target positions are misaligned")
    loss = functional.cross_entropy(answer_logits, targets, reduction="mean")
    answer_nll = float(loss.item())
    if not math.isfinite(answer_nll):
        raise RuntimeError("S1 teacher-forced answer NLL is non-finite")
    if bf16_reference is None:
        reference_distribution = build_fixed_support_distribution(
            answer_logits, top_k=distribution_top_k
        )
        js_divergence = None
        position_js = None
    else:
        reference_distribution = None
        js_divergence, position_js = compare_to_fixed_support(
            answer_logits, bf16_reference
        )
    return S1InferenceResult(
        prediction=prediction,
        answer_nll=answer_nll,
        prompt_token_count=prompt_length,
        answer_token_count=int(targets.numel()),
        target_token_sha256=_token_sha256(targets),
        reference_distribution=reference_distribution,
        js_divergence=js_divergence,
        position_js_divergences=position_js,
    )
