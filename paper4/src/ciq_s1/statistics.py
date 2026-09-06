"""Image-grouped cross-fitting, bootstrap, and permutation tests for C01-S1."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


FAMILIES = (
    "global_perception",
    "fine_grained_recognition",
    "spatial_relation",
    "reasoning",
)
COVARIATES = (
    "question_token_length",
    "answer_token_length",
    "visual_token_count",
)


@dataclass(frozen=True)
class CrossfitResult:
    """Per-observation errors from one group-specific nested-model comparison."""

    image_ids: tuple[str, ...]
    additive_squared_errors: tuple[float, ...]
    interaction_squared_errors: tuple[float, ...]

    @property
    def partial_r2(self) -> float:
        """Return held-out partial R² for this result."""
        additive = sum(self.additive_squared_errors)
        if additive <= 0:
            return 0.0
        return float((additive - sum(self.interaction_squared_errors)) / additive)


def image_grouped_folds(
    image_ids: Sequence[str], *, fold_count: int, seed: int
) -> tuple[tuple[str, ...], ...]:
    """Assign unique images to deterministic, disjoint cross-validation folds."""
    unique = sorted(
        set(image_ids),
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    if not 2 <= fold_count <= len(unique):
        raise ValueError("fold_count must be between two and the image count")
    folds = tuple(tuple(unique[index::fold_count]) for index in range(fold_count))
    flattened = [image_id for fold in folds for image_id in fold]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(unique):
        raise AssertionError("image-grouped folds overlap or omit images")
    return folds


def _feature_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    means: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    covariates = np.asarray(
        [[float(row[name]) for name in COVARIATES] for row in rows], dtype=np.float64
    )
    if means is None:
        means = covariates.mean(axis=0)
    if scales is None:
        scales = covariates.std(axis=0)
        scales[scales == 0] = 1.0
    standardized = (covariates - means) / scales
    family_columns = np.asarray(
        [
            [float(str(row["query_family"]) == family) for family in FAMILIES[1:]]
            for row in rows
        ],
        dtype=np.float64,
    )
    matrix = np.column_stack(
        [np.ones(len(rows), dtype=np.float64), family_columns, standardized]
    )
    return matrix, means, scales


def crossfit_group(
    rows: Sequence[Mapping[str, Any]], *, fold_count: int, seed: int
) -> CrossfitResult:
    """Cross-fit additive and image×family effects with LOO held-out cell queries."""
    if len({str(row["prompt_template_sha256"]) for row in rows}) != 1:
        raise ValueError("prompt template is not controlled within the S1 analysis")
    folds = image_grouped_folds(
        [str(row["image_id"]) for row in rows], fold_count=fold_count, seed=seed
    )
    additive_errors: list[float] = []
    interaction_errors: list[float] = []
    output_images: list[str] = []
    for fold in folds:
        test_images = set(fold)
        train = [row for row in rows if str(row["image_id"]) not in test_images]
        test = [row for row in rows if str(row["image_id"]) in test_images]
        train_matrix, means, scales = _feature_matrix(train)
        train_targets = np.asarray(
            [float(row["sensitivity"]) for row in train], dtype=np.float64
        )
        coefficients = np.linalg.lstsq(train_matrix, train_targets, rcond=None)[0]
        test_matrix, _, _ = _feature_matrix(test, means=means, scales=scales)
        base_predictions = test_matrix @ coefficients
        residuals = (
            np.asarray([float(row["sensitivity"]) for row in test], dtype=np.float64)
            - base_predictions
        )
        image_indices: dict[str, list[int]] = defaultdict(list)
        cell_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
        for position, row in enumerate(test):
            image_id = str(row["image_id"])
            family = str(row["query_family"])
            image_indices[image_id].append(position)
            cell_indices[(image_id, family)].append(position)
        for index, row in enumerate(test):
            image_id = str(row["image_id"])
            family = str(row["query_family"])
            image_positions = image_indices[image_id]
            cell_positions = cell_indices[(image_id, family)]
            if len(image_positions) < 2 or len(cell_positions) < 2:
                raise ValueError(
                    "cross-fit requires repeated held-out image-family cells"
                )
            image_offset = (
                float(residuals[image_positions].sum()) - float(residuals[index])
            ) / (len(image_positions) - 1)
            cell_offset = (
                float(residuals[cell_positions].sum()) - float(residuals[index])
            ) / (len(cell_positions) - 1)
            additive_prediction = base_predictions[index] + image_offset
            interaction_prediction = base_predictions[index] + cell_offset
            target = float(row["sensitivity"])
            additive_errors.append((target - additive_prediction) ** 2)
            interaction_errors.append((target - interaction_prediction) ** 2)
            output_images.append(image_id)
    return CrossfitResult(
        image_ids=tuple(output_images),
        additive_squared_errors=tuple(additive_errors),
        interaction_squared_errors=tuple(interaction_errors),
    )


def pooled_partial_r2(results: Sequence[CrossfitResult]) -> float:
    """Pool groups using preregistered SSE weighting."""
    additive = sum(sum(result.additive_squared_errors) for result in results)
    interaction = sum(sum(result.interaction_squared_errors) for result in results)
    if additive <= 0:
        return 0.0
    return float((additive - interaction) / additive)


def image_grouped_bootstrap(
    results: Sequence[CrossfitResult], *, replicates: int, seed: int
) -> dict[str, Any]:
    """Bootstrap precomputed cross-fit losses by whole image clusters."""
    if replicates < 100:
        raise ValueError("bootstrap needs at least 100 replicates")
    by_image: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for result in results:
        for image_id, additive, interaction in zip(
            result.image_ids,
            result.additive_squared_errors,
            result.interaction_squared_errors,
            strict=True,
        ):
            by_image[image_id][0] += additive
            by_image[image_id][1] += interaction
    image_ids = sorted(by_image)
    generator = random.Random(seed)
    values: list[float] = []
    for _ in range(replicates):
        sampled = generator.choices(image_ids, k=len(image_ids))
        additive = sum(by_image[image_id][0] for image_id in sampled)
        interaction = sum(by_image[image_id][1] for image_id in sampled)
        values.append((additive - interaction) / additive if additive > 0 else 0.0)
    lower, upper = np.quantile(np.asarray(values), [0.025, 0.975]).tolist()
    return {
        "replicates": replicates,
        "lower_95": float(lower),
        "upper_95": float(upper),
        "median": float(np.median(values)),
    }


def _permuted_families(
    rows: Sequence[Mapping[str, Any]], *, generator: random.Random
) -> list[dict[str, Any]]:
    by_image: dict[str, list[int]] = defaultdict(list)
    copied = [dict(row) for row in rows]
    for index, row in enumerate(copied):
        by_image[str(row["image_id"])].append(index)
    for indices in by_image.values():
        labels = [str(copied[index]["query_family"]) for index in indices]
        generator.shuffle(labels)
        for index, label in zip(indices, labels, strict=True):
            copied[index]["query_family"] = label
    return copied


def within_image_permutation_test(
    rows_by_group: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    fold_count: int,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Test interaction structure by shuffling family labels within each image."""
    if permutations < 99:
        raise ValueError("permutation test needs at least 99 shuffles")
    observed_results = [
        crossfit_group(rows, fold_count=fold_count, seed=seed)
        for rows in rows_by_group.values()
    ]
    observed = pooled_partial_r2(observed_results)
    generator = random.Random(seed)
    null_values: list[float] = []
    for _ in range(permutations):
        permuted_results = [
            crossfit_group(
                _permuted_families(rows, generator=generator),
                fold_count=fold_count,
                seed=seed,
            )
            for rows in rows_by_group.values()
        ]
        null_values.append(pooled_partial_r2(permuted_results))
    null = np.asarray(null_values, dtype=np.float64)
    p_value = (1 + int(np.count_nonzero(null >= observed))) / (permutations + 1)
    standard_deviation = float(null.std(ddof=1))
    effect_size = (
        (observed - float(null.mean())) / standard_deviation
        if standard_deviation > 0
        else math.inf
    )
    return {
        "observed_partial_r2": observed,
        "permutations": permutations,
        "p_value": p_value,
        "null_mean": float(null.mean()),
        "null_standard_deviation": standard_deviation,
        "standardized_effect_size": effect_size,
    }
