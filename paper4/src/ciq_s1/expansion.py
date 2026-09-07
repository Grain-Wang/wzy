"""Deterministic bounded replication-depth expansion for C01."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .sampling import QUERY_FAMILIES, _stable_rank, classify_s1_query_family


def select_expansion_rows(
    rows: Iterable[Mapping[str, Any]],
    original_question_ids: set[str],
    *,
    questions_per_cell: int = 4,
    seed: int = 20260906,
) -> list[dict[str, Any]]:
    """Keep original rows and deterministically add rows to uniform cells."""
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        image = str(row.get("image_id", row.get("imageId", row.get("image", ""))))
        question = str(row.get("question_id", row.get("id", "")))
        family = classify_s1_query_family(row)
        if image and question and family in QUERY_FAMILIES:
            item = dict(row)
            item["image_id"] = image
            item["question_id"] = question
            item["query_family"] = family
            cells[(image, family)].append(item)
    selected: list[dict[str, Any]] = []
    for key in sorted(cells):
        # GQA global cells are retained descriptively only when they cannot
        # meet the preregistered replication depth; confirmatory cells remain
        # uniform across the three non-global families.
        if key[1] == "global_perception" and len(cells[key]) < questions_per_cell:
            continue
        candidates = cells[key]
        old = [r for r in candidates if r["question_id"] in original_question_ids]
        if len(old) > questions_per_cell:
            raise ValueError(f"original cell exceeds target: {key}")
        old_ids = {r["question_id"] for r in old}
        ranked = sorted(
            (r for r in candidates if r["question_id"] not in old_ids),
            key=lambda r: _stable_rank(seed, r["question_id"]),
        )
        if len(old) + len(ranked) < questions_per_cell:
            raise ValueError(f"cell lacks replication depth {key}")
        chosen = old + ranked[: questions_per_cell - len(old)]
        selected.extend(sorted(chosen, key=lambda r: r["question_id"]))
    return selected


def replication_distribution(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize deterministic image×family cell support."""
    cells: dict[tuple[str, str], int] = defaultdict(int)
    images: dict[str, set[str]] = defaultdict(set)
    families: dict[str, int] = defaultdict(int)
    for row in records:
        image, family = str(row["image_id"]), str(row["query_family"])
        cells[(image, family)] += 1
        images[image].add(family)
        families[family] += 1
    values = sorted(cells.values())
    if not values:
        raise ValueError("empty expansion")
    return {
        "images": len(images),
        "questions": sum(values),
        "cells": len(values),
        "mean_questions_per_cell": sum(values) / len(values),
        "median_questions_per_cell": values[len(values) // 2],
        "minimum_questions_per_cell": values[0],
        "cell_support_distribution": {str(v): values.count(v) for v in sorted(set(values))},
        "family_question_counts": dict(sorted(families.items())),
        "images_per_family": {
            family: sum(family in fs for fs in images.values())
            for family in QUERY_FAMILIES
        },
    }
