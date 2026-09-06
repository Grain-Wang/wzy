"""Tests for the amended official GQA-compatible S1 evaluator."""

from __future__ import annotations

from src.ciq_s1.evaluation import (
    attach_scores,
    canonical_gqa_prediction,
    gqa_official_exact_score,
)


def test_official_score_uses_exact_answer_not_vqa_normalization() -> None:
    assert gqa_official_exact_score("Yes", "yes") == 1.0
    assert gqa_official_exact_score("The cat.", "cat") == 0.0
    scores = attach_scores("The cat.", "cat", maximum_words=16)
    assert scores["official_score"] == 0.0
    assert scores["normalized_exact_score"] == 1.0


def test_gqa_serialization_only_strips_and_lowercases() -> None:
    assert canonical_gqa_prediction("  Short Phrase  ") == "short phrase"
    assert canonical_gqa_prediction("two.") == "two."
