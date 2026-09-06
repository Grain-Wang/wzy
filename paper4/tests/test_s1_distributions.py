"""Tests for fixed-support Jensen-Shannon metrics."""

from __future__ import annotations

import math

import torch

from src.ciq_s1.distributions import (
    build_fixed_support_distribution,
    compare_to_fixed_support,
    jensen_shannon_divergence,
    project_logits_to_fixed_support,
)
from src.ciq_s1.evaluation import aggregate_condition_metrics


def test_identical_logits_have_zero_js() -> None:
    logits = torch.tensor([[1.0, 2.0, -1.0, 0.5], [0.2, 0.1, 0.3, 0.4]])
    reference = build_fixed_support_distribution(logits, top_k=2)
    mean_js, position_js = compare_to_fixed_support(logits, reference)
    assert mean_js < 1e-12
    assert max(position_js) < 1e-12


def test_known_binary_js_value() -> None:
    first = torch.tensor([[1.0, 0.0]])
    second = torch.tensor([[0.0, 1.0]])
    value = float(jensen_shannon_divergence(first, second).item())
    assert math.isclose(value, math.log(2.0), rel_tol=1e-6)


def test_fixed_support_probability_normalization() -> None:
    logits = torch.tensor([[3.0, 1.0, 0.0, -2.0]])
    reference = build_fixed_support_distribution(logits, top_k=2)
    projected = project_logits_to_fixed_support(logits + 0.7, reference)
    assert torch.allclose(projected.sum(dim=-1), torch.ones(1))
    assert math.isclose(sum(reference.probabilities[0]), 1.0, rel_tol=1e-6)


def test_baseline_aggregation_ignores_unavailable_js() -> None:
    row = {
        "image_id": "image",
        "official_score": 1.0,
        "normalized_exact_score": 1.0,
        "answer_nll": 0.5,
        "empty_generation": False,
        "malformed_generation": False,
        "generated_word_count": 1,
        "question_token_length": 4,
        "answer_token_length": 1,
        "js_divergence": None,
    }
    summary = aggregate_condition_metrics([row])
    assert "mean_js_divergence" not in summary
