"""Tests for image-grouped C01-S1 inference statistics."""

from __future__ import annotations

import json
from collections import defaultdict

from src.ciq_s1.statistics import (
    crossfit_group,
    image_grouped_bootstrap,
    image_grouped_folds,
    pooled_partial_r2,
    within_image_permutation_test,
)


def _rows(group_shift: float = 0.0) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    families = (
        "fine_grained_recognition",
        "spatial_relation",
        "reasoning",
    )
    family_effect = {
        "fine_grained_recognition": -1.0,
        "spatial_relation": 0.0,
        "reasoning": 1.0,
    }
    for image in range(15):
        for family_index, family in enumerate(families):
            cell_interaction = ((image + family_index) % 3 - 1) * 2.0
            for repetition in range(2):
                rows.append(
                    {
                        "image_id": f"i{image:02d}",
                        "query_family": family,
                        "question_token_length": 5 + repetition,
                        "answer_token_length": 2,
                        "visual_token_count": 64 + image,
                        "prompt_template_sha256": "locked",
                        "sensitivity": group_shift
                        + 0.1 * image
                        + family_effect[family]
                        + cell_interaction
                        + 0.01 * repetition,
                    }
                )
    return rows


def test_image_grouped_folds_are_disjoint_and_complete() -> None:
    image_ids = [str(index // 2) for index in range(30)]
    folds = image_grouped_folds(image_ids, fold_count=5, seed=9)
    seen: set[str] = set()
    for fold in folds:
        assert not seen.intersection(fold)
        seen.update(fold)
    assert seen == set(image_ids)


def test_crossfit_detects_repeated_cell_interaction() -> None:
    first = crossfit_group(_rows(), fold_count=5, seed=4)
    second = crossfit_group(_rows(0.5), fold_count=5, seed=4)
    assert first.partial_r2 > 0.5
    assert isinstance(first.partial_r2, float)
    json.dumps({"partial_r2": first.partial_r2, "passes": first.partial_r2 > 0})
    pooled = pooled_partial_r2([first, second])
    assert pooled > 0.5
    bootstrap = image_grouped_bootstrap([first, second], replicates=200, seed=4)
    assert bootstrap["lower_95"] > 0.0


def test_within_image_permutation_sanity() -> None:
    result = within_image_permutation_test(
        {"g1": _rows(), "g2": _rows(0.5)},
        fold_count=5,
        permutations=99,
        seed=11,
    )
    assert result["observed_partial_r2"] > result["null_mean"]
    assert result["p_value"] <= 0.05


def test_every_synthetic_cell_is_repeated() -> None:
    cells: defaultdict[tuple[str, str], int] = defaultdict(int)
    for row in _rows():
        cells[(str(row["image_id"]), str(row["query_family"]))] += 1
    assert set(cells.values()) == {2}
