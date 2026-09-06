"""Automatic evaluator integrity tests."""

from src.ciq_s0.evaluation import (
    normalized_exact_score,
    normalize_answer,
    vqa_soft_score,
)


def test_normalize_answer() -> None:
    assert normalize_answer(" The, TWO cats!\n") == "2 cats"


def test_normalized_exact_score() -> None:
    assert normalized_exact_score("A blue car.", "blue car") == 1.0
    assert normalized_exact_score("red", "blue") == 0.0


def test_vqa_soft_score() -> None:
    assert vqa_soft_score("dog", ["dog", "Dog!", "cat"]) == 2 / 3
    assert vqa_soft_score("dog", ["dog", "dog", "dog", "cat"]) == 1.0
