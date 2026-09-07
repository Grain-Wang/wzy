"""Deterministic balanced-selection tests for C02-S0."""

from __future__ import annotations

from collections import Counter

from src.causalquant.s0_manifest import select_smoke_records


def _record(
    expression_id: int,
    image_id: int,
    ref_id: int,
    ann_id: int,
    text: str,
    categories: list[str],
) -> dict[str, object]:
    return {
        "ann_id": ann_id,
        "area_mismatch": 0.0,
        "bbox": [1.0, 1.0, 4.0, 4.0],
        "categories": categories,
        "control_bbox": [12.0, 12.0, 4.0, 4.0],
        "expression_id": expression_id,
        "height": 20.0,
        "image_id": image_id,
        "ref_id": ref_id,
        "text": text,
        "width": 20.0,
    }


def test_selection_is_balanced_and_repeatable() -> None:
    """Hash ordering returns the same same-image pairs and exact quotas."""
    records = [
        _record(1, 10, 101, 1001, "red car", ["attribute"]),
        _record(2, 10, 102, 1002, "car on the right", ["spatial"]),
        _record(3, 20, 103, 1003, "car with a person", ["relational"]),
        _record(4, 20, 104, 1004, "car", ["recognition"]),
    ]
    metadata = {
        ref_id: {"ann_id": 1000 + index, "category_id": 3, "file_name": f"{ref_id}.jpg"}
        for index, ref_id in enumerate((101, 102, 103, 104), start=1)
    }
    first = select_smoke_records(
        records,
        metadata,
        seed=7,
        image_count=2,
        expressions_per_image=2,
        expressions_per_family=1,
    )
    second = select_smoke_records(
        records,
        metadata,
        seed=7,
        image_count=2,
        expressions_per_image=2,
        expressions_per_family=1,
    )
    assert first == second
    assert len({row["image_id"] for row in first}) == 2
    assert Counter(row["semantic_family"] for row in first) == Counter(
        {"attribute": 1, "spatial": 1, "relational": 1, "recognition": 1}
    )
