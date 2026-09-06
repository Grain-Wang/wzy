"""Same-image sensitivity geometry and bootstrap summaries for C01-S1."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    """Return Spearman rank correlation with average tied ranks."""
    left = _average_ranks(np.asarray(first, dtype=np.float64))
    right = _average_ranks(np.asarray(second, dtype=np.float64))
    if left.std() == 0 or right.std() == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def cosine_distance(first: Sequence[float], second: Sequence[float]) -> float:
    """Return cosine distance, with two zero vectors treated as identical."""
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 0.0 if np.linalg.norm(left - right) == 0 else 1.0
    return 1.0 - float(np.dot(left, right) / denominator)


def top_k_indices(values: Sequence[float], *, top_k: int) -> frozenset[int]:
    """Return deterministic indices of the largest sensitivity entries."""
    if not 0 < top_k <= len(values):
        raise ValueError("top_k is outside the vector length")
    ordered = sorted(
        range(len(values)), key=lambda index: (-float(values[index]), index)
    )
    return frozenset(ordered[:top_k])


def jaccard_similarity(first: frozenset[int], second: frozenset[int]) -> float:
    """Return set Jaccard similarity."""
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def same_image_pair_metrics(
    sample_rows: Sequence[Mapping[str, Any]],
    sensitivities: Mapping[str, Sequence[float]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Measure sensitivity distance for every same-image query pair."""
    by_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        by_image[str(row["image_id"])].append(row)
    metrics: list[dict[str, Any]] = []
    for image_id, rows in sorted(by_image.items()):
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1 :]:
                left_id = str(left["question_id"])
                right_id = str(right["question_id"])
                first = sensitivities[left_id]
                second = sensitivities[right_id]
                first_top = top_k_indices(first, top_k=top_k)
                second_top = top_k_indices(second, top_k=top_k)
                same_family = str(left["query_family"]) == str(right["query_family"])
                metrics.append(
                    {
                        "image_id": image_id,
                        "left_question_id": left_id,
                        "right_question_id": right_id,
                        "left_family": left["query_family"],
                        "right_family": right["query_family"],
                        "same_family": same_family,
                        "spearman": spearman_correlation(first, second),
                        "cosine_distance": cosine_distance(first, second),
                        "top_k_jaccard": jaccard_similarity(first_top, second_top),
                        "protected_group_identity_change": float(
                            first_top != second_top
                        ),
                    }
                )
    return metrics


def clustered_mean_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_name: str,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    """Return an image-cluster bootstrap interval for a descriptive mean."""
    by_image: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_image[str(row["image_id"])].append(float(row[value_name]))
    images = sorted(by_image)
    observed = float(
        np.mean([value for values in by_image.values() for value in values])
    )
    generator = random.Random(seed)
    bootstraps: list[float] = []
    for _ in range(replicates):
        sampled = generator.choices(images, k=len(images))
        values = [value for image_id in sampled for value in by_image[image_id]]
        bootstraps.append(float(np.mean(values)))
    lower, upper = np.quantile(bootstraps, [0.025, 0.975]).tolist()
    return {"mean": observed, "lower_95": float(lower), "upper_95": float(upper)}


def family_group_heatmap(
    sample_rows: Sequence[Mapping[str, Any]],
    sensitivities: Mapping[str, Sequence[float]],
    group_ids: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Aggregate mean ΔNLL into a query-family × group matrix."""
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in sample_rows:
        family = str(row["query_family"])
        vector = sensitivities[str(row["question_id"])]
        for group_id, sensitivity in zip(group_ids, vector, strict=True):
            values[family][group_id].append(float(sensitivity))
    return {
        family: {
            group_id: float(np.mean(group_values))
            for group_id, group_values in group_mapping.items()
        }
        for family, group_mapping in values.items()
    }


def standardize_sensitivity_vectors(
    sensitivities: Mapping[str, Sequence[float]],
) -> dict[str, list[float]]:
    """Z-standardize every group across the frozen confirmatory questions."""
    question_ids = sorted(sensitivities)
    matrix = np.asarray(
        [sensitivities[item] for item in question_ids], dtype=np.float64
    )
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales == 0] = 1.0
    standardized = (matrix - means) / scales
    return {
        question_id: standardized[index].tolist()
        for index, question_id in enumerate(question_ids)
    }


def ranking_stability_bootstrap(
    sample_rows: Sequence[Mapping[str, Any]],
    sensitivities: Mapping[str, Sequence[float]],
    *,
    top_k: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap whole images and compare top-k ranking identities to full data."""
    by_image: dict[str, list[str]] = defaultdict(list)
    for row in sample_rows:
        by_image[str(row["image_id"])].append(str(row["question_id"]))
    images = sorted(by_image)

    def mean_vector(question_ids: Sequence[str]) -> np.ndarray:
        return np.asarray([sensitivities[item] for item in question_ids]).mean(axis=0)

    all_questions = [
        question_id for image_id in images for question_id in by_image[image_id]
    ]
    reference = top_k_indices(mean_vector(all_questions), top_k=top_k)
    generator = random.Random(seed)
    similarities: list[float] = []
    identities: list[bool] = []
    for _ in range(replicates):
        sampled = generator.choices(images, k=len(images))
        questions = [
            question_id for image_id in sampled for question_id in by_image[image_id]
        ]
        selected = top_k_indices(mean_vector(questions), top_k=top_k)
        similarities.append(jaccard_similarity(reference, selected))
        identities.append(selected == reference)
    return {
        "top_k": top_k,
        "median_jaccard": float(np.median(similarities)),
        "lower_95_jaccard": float(np.quantile(similarities, 0.025)),
        "exact_identity_rate": float(np.mean(identities)),
        "reference_indices": sorted(reference),
    }
