"""Qwen2-VL deterministic generation and answer-only teacher-forced NLL."""

from __future__ import annotations

import hashlib
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from .evaluation import normalize_answer


@dataclass(frozen=True)
class InferenceResult:
    """Generated answer plus answer-only teacher-forced loss audit."""

    prediction: str
    normalized_prediction: str
    answer_nll: float
    prompt_token_count: int
    answer_token_count: int
    first_answer_label_position: int
    last_answer_label_position: int
    target_token_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def configure_determinism(seed: int) -> dict[str, Any]:
    """Configure deterministic inference and return the applied settings."""
    cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cublas_workspace_config not in {":4096:8", ":16:8"}:
        raise RuntimeError("C01-S0 requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    return {
        "seed": seed,
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cublas_workspace_config": cublas_workspace_config,
    }


def load_rgb_image(path: Path) -> Image.Image:
    """Load an image eagerly as RGB."""
    with Image.open(path) as source:
        return source.convert("RGB")


def build_messages(
    image: Image.Image, question: str, instruction: str
) -> list[dict[str, Any]]:
    """Build the locked one-turn Qwen chat message."""
    return build_text_messages(image, f"{instruction}\nQuestion: {question}")


def build_text_messages(image: Image.Image, prompt: str) -> list[dict[str, Any]]:
    """Build a one-turn Qwen chat message from an exact text prompt."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def prepare_prompt_inputs(
    processor: Any,
    *,
    image: Image.Image,
    question: str,
    instruction: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Apply the locked model chat template and move tensors to one device."""
    messages = build_messages(image, question, instruction)
    return prepare_messages_inputs(processor, messages=messages, device=device)


def prepare_text_prompt_inputs(
    processor: Any,
    *,
    image: Image.Image,
    prompt: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Apply the model chat template to an exact prompt and image."""
    messages = build_text_messages(image, prompt)
    return prepare_messages_inputs(processor, messages=messages, device=device)


def prepare_messages_inputs(
    processor: Any,
    *,
    messages: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Tokenize prepared Qwen messages and move tensors to one device."""
    encoded = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in encoded.items()
    }


def append_answer_labels(
    prompt_inputs: dict[str, torch.Tensor],
    *,
    tokenizer: Any,
    answer: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
    """Append fixed gold tokens and mask every prompt position from NLL."""
    eos_token = tokenizer.eos_token or ""
    answer_text = answer.strip() + eos_token
    answer_ids = tokenizer(
        answer_text,
        add_special_tokens=False,
        return_tensors="pt",
    )[
        "input_ids"
    ].to(prompt_inputs["input_ids"].device)
    if answer_ids.numel() == 0:
        raise ValueError("gold answer tokenization is empty")
    prompt_length = int(prompt_inputs["input_ids"].shape[1])
    full_inputs = dict(prompt_inputs)
    full_inputs["input_ids"] = torch.cat(
        [prompt_inputs["input_ids"], answer_ids], dim=1
    )
    full_inputs["attention_mask"] = torch.cat(
        [
            prompt_inputs["attention_mask"],
            torch.ones_like(answer_ids, dtype=prompt_inputs["attention_mask"].dtype),
        ],
        dim=1,
    )
    labels = torch.full_like(full_inputs["input_ids"], -100)
    labels[:, prompt_length:] = answer_ids
    return full_inputs, labels, prompt_length


@torch.inference_mode()
def generate_answer(
    model: torch.nn.Module,
    processor: Any,
    prompt_inputs: dict[str, torch.Tensor],
    generation_settings: dict[str, Any],
) -> str:
    """Generate deterministically and decode only newly generated tokens."""
    outputs = model.generate(**prompt_inputs, **generation_settings)
    prompt_length = int(prompt_inputs["input_ids"].shape[1])
    prediction = processor.batch_decode(
        outputs[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return prediction.strip()


@torch.inference_mode()
def answer_only_nll(
    model: torch.nn.Module,
    full_inputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    prompt_length: int,
) -> float:
    """Compute mean next-token NLL exclusively over appended answer tokens."""
    if torch.any(labels[:, :prompt_length] != -100):
        raise AssertionError("prompt labels must all be masked")
    if torch.any(labels[:, prompt_length:] < 0):
        raise AssertionError("all answer labels must be active token IDs")
    outputs = model(**full_inputs, use_cache=False)
    logits = outputs.logits
    answer_logits = logits[:, prompt_length - 1 : -1, :].float()
    answer_targets = labels[:, prompt_length:]
    if answer_logits.shape[:2] != answer_targets.shape:
        raise AssertionError("answer logit/target positions are misaligned")
    loss = functional.cross_entropy(
        answer_logits.reshape(-1, answer_logits.shape[-1]),
        answer_targets.reshape(-1),
        reduction="mean",
    )
    value = float(loss.item())
    if not math.isfinite(value):
        raise RuntimeError("teacher-forced answer NLL is not finite")
    return value


def _token_sha256(token_ids: torch.Tensor) -> str:
    values = token_ids.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def run_inference_case(
    model: torch.nn.Module,
    processor: Any,
    *,
    image: Image.Image,
    question: str,
    answer: str,
    instruction: str,
    device: torch.device,
    generation_settings: dict[str, Any],
) -> InferenceResult:
    """Run generation and correctly masked teacher-forced NLL for one sample."""
    prompt_inputs = prepare_prompt_inputs(
        processor,
        image=image,
        question=question,
        instruction=instruction,
        device=device,
    )
    prediction = generate_answer(model, processor, prompt_inputs, generation_settings)
    full_inputs, labels, prompt_length = append_answer_labels(
        prompt_inputs, tokenizer=processor.tokenizer, answer=answer
    )
    nll = answer_only_nll(model, full_inputs, labels, prompt_length)
    answer_ids = labels[:, prompt_length:]
    return InferenceResult(
        prediction=prediction,
        normalized_prediction=normalize_answer(prediction),
        answer_nll=nll,
        prompt_token_count=prompt_length,
        answer_token_count=int(answer_ids.shape[1]),
        first_answer_label_position=prompt_length,
        last_answer_label_position=int(labels.shape[1] - 1),
        target_token_sha256=_token_sha256(answer_ids),
    )


@torch.inference_mode()
def extract_visual_features(
    model: torch.nn.Module, prompt_inputs: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Execute the actual Qwen visual tower for an image batch."""
    if "pixel_values" not in prompt_inputs or "image_grid_thw" not in prompt_inputs:
        raise KeyError("Qwen prompt inputs lack pixel_values or image_grid_thw")
    visual = getattr(model, "visual", None)
    if visual is None:
        raise AttributeError("loaded model has no visual module")
    pixel_values = prompt_inputs["pixel_values"].to(dtype=visual.dtype)
    return visual(pixel_values, grid_thw=prompt_inputs["image_grid_thw"])


def tensor_value_sha256(tensor: torch.Tensor) -> str:
    """Hash a runtime tensor's exact shape, dtype, and values."""
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
