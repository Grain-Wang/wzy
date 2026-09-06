"""Tests for same-image sensitivity-structure metrics."""

from __future__ import annotations

import math

from src.ciq_s1.structure import (
    cosine_distance,
    jaccard_similarity,
    same_image_pair_metrics,
    spearman_correlation,
    standardize_sensitivity_vectors,
    top_k_indices,
)


def test_rank_cosine_and_jaccard_metrics() -> None:
    assert math.isclose(spearman_correlation([1, 2, 3], [1, 2, 3]), 1.0)
    assert math.isclose(spearman_correlation([1, 2, 3], [3, 2, 1]), -1.0)
    assert math.isclose(cosine_distance([1, 0], [0, 1]), 1.0)
    first = top_k_indices([3, 2, 1], top_k=2)
    second = top_k_indices([3, 1, 2], top_k=2)
    assert math.isclose(jaccard_similarity(first, second), 1 / 3)


def test_pair_metrics_only_compare_queries_within_an_image() -> None:
    rows = [
        {"image_id": "a", "question_id": "q1", "query_family": "reasoning"},
        {"image_id": "a", "question_id": "q2", "query_family": "reasoning"},
        {"image_id": "a", "question_id": "q3", "query_family": "spatial_relation"},
        {"image_id": "b", "question_id": "q4", "query_family": "reasoning"},
    ]
    sensitivities = {
        "q1": [1.0, 2.0, 3.0],
        "q2": [1.0, 2.0, 3.0],
        "q3": [3.0, 2.0, 1.0],
        "q4": [1.0, 2.0, 3.0],
    }
    metrics = same_image_pair_metrics(rows, sensitivities, top_k=1)
    assert len(metrics) == 3
    assert sum(bool(row["same_family"]) for row in metrics) == 1


def test_groupwise_standardization_has_zero_mean() -> None:
    standardized = standardize_sensitivity_vectors(
        {"q1": [1.0, 4.0], "q2": [2.0, 4.0], "q3": [3.0, 4.0]}
    )
    assert math.isclose(
        sum(row[0] for row in standardized.values()), 0.0, abs_tol=1e-12
    )
    assert {row[1] for row in standardized.values()} == {0.0}
