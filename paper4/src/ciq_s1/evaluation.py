"""Official GQA-compatible exact scoring plus non-primary audit metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..ciq_s0.evaluation import normalized_exact_score


def canonical_gqa_prediction(text: str) -> str:
    """Serialize a generated short answer into GQA's lowercase answer vocabulary."""
    return text.strip().lower()


def gqa_official_exact_score(prediction: str, reference: str) -> float:
    """Return GQA standard 0/1 exact accuracy after fixed output serialization."""
    return float(
        canonical_gqa_prediction(prediction) == canonical_gqa_prediction(reference)
    )


def generation_flags(prediction: str, *, maximum_words: int) -> dict[str, bool]:
    """Flag empty and structurally malformed generations without using correctness."""
    stripped = prediction.strip()
    return {
        "empty_generation": not stripped,
        "malformed_generation": bool(stripped)
        and len(stripped.split()) > maximum_words,
    }


def aggregate_condition_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate locked S1 quality and distribution metrics."""
    values = list(rows)
    if not values:
        raise ValueError("cannot aggregate an empty condition")

    def mean(name: str) -> float:
        return sum(float(row[name]) for row in values) / len(values)

    nlls = sorted(float(row["answer_nll"]) for row in values)
    midpoint = len(nlls) // 2
    median_nll = (
        nlls[midpoint] if len(nlls) % 2 else (nlls[midpoint - 1] + nlls[midpoint]) / 2.0
    )
    result: dict[str, Any] = {
        "sample_count": len(values),
        "image_count": len({str(row["image_id"]) for row in values}),
        "official_score": mean("official_score"),
        "normalized_exact": mean("normalized_exact_score"),
        "mean_answer_nll": mean("answer_nll"),
        "median_answer_nll": median_nll,
        "empty_generation_rate": mean("empty_generation"),
        "malformed_generation_rate": mean("malformed_generation"),
        "mean_generated_word_count": mean("generated_word_count"),
        "mean_question_token_length": mean("question_token_length"),
        "mean_answer_token_length": mean("answer_token_length"),
    }
    optional = (
        "js_divergence",
        "answer_flip",
        "bf16_correct_to_condition_wrong",
        "delta_nll",
    )
    for name in optional:
        if all(row.get(name) is not None for row in values):
            result[f"mean_{name}"] = mean(name)
    return result


def attach_scores(
    prediction: str, reference: str, *, maximum_words: int
) -> dict[str, Any]:
    """Attach official, audit, and structural generation fields."""
    flags = generation_flags(prediction, maximum_words=maximum_words)
    return {
        "official_prediction": canonical_gqa_prediction(prediction),
        "official_score": gqa_official_exact_score(prediction, reference),
        "normalized_exact_score": normalized_exact_score(prediction, reference),
        "generated_word_count": len(prediction.strip().split()),
        **flags,
    }
