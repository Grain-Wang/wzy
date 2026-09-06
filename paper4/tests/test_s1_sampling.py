"""Tests for the independent C01-S1 sampler and frozen taxonomy."""

from __future__ import annotations

from collections import Counter, defaultdict

from src.ciq_s1.sampling import (
    S1SampleRecord,
    classify_s1_query_family,
    select_s1_rows,
    validate_sampling_constraints,
    whole_image_execution_order,
)


def _row(image: int, family: str, repetition: int) -> dict[str, object]:
    types = {
        "global_perception": {"semantic": "global", "structural": "verify"},
        "fine_grained_recognition": {"semantic": "attr", "structural": "query"},
        "spatial_relation": {"semantic": "rel", "structural": "query"},
        "reasoning": {"semantic": "attr", "structural": "logical"},
    }[family]
    return {
        "id": f"q{image:03d}-{family}-{repetition}",
        "imageId": f"i{image:03d}",
        "question": f"question {family} {repetition}",
        "answer": "yes",
        "types": types,
        "semantic": [{"operation": "query", "argument": "x", "dependencies": []}],
        "isBalanced": True,
    }


def _pool() -> list[dict[str, object]]:
    return [
        _row(image, family, repetition)
        for image in range(100)
        for family in (
            "global_perception",
            "fine_grained_recognition",
            "spatial_relation",
            "reasoning",
        )
        for repetition in range(3)
    ]


def _select(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return select_s1_rows(
        rows,
        image_count=80,
        questions_per_image=6,
        families_per_image=3,
        questions_per_cell=2,
        minimum_images_per_family=15,
        seed=7,
    )


def test_s1_selection_is_deterministic_and_output_independent() -> None:
    first = _select(_pool())
    second_pool = _pool()
    for index, row in enumerate(second_pool):
        row["model_output"] = "correct" if index % 2 else "wrong"
    second = _select(reversed(second_pool))
    assert [row["_question_id"] for row in first] == [
        row["_question_id"] for row in second
    ]


def test_s1_selection_satisfies_repeated_cell_constraints() -> None:
    selected = _select(_pool())
    by_image: dict[str, list[dict[str, object]]] = defaultdict(list)
    cells: Counter[tuple[str, str]] = Counter()
    for row in selected:
        image_id = str(row["_image_id"])
        family = str(row["_query_family"])
        by_image[image_id].append(row)
        cells[(image_id, family)] += 1
    assert len(by_image) == 80
    assert len(selected) == 480
    assert all(len(rows) == 6 for rows in by_image.values())
    assert all(
        len({row["_query_family"] for row in rows}) == 3 for rows in by_image.values()
    )
    assert set(cells.values()) == {2}
    for family in (
        "global_perception",
        "fine_grained_recognition",
        "spatial_relation",
        "reasoning",
    ):
        assert len({image for image, value in cells if value == family}) >= 15


def test_nested_gqa_metadata_has_precedence_and_reasoning_precedes_relation() -> None:
    nested = {
        "question": "where is it",
        "types": {"semantic": "attr", "structural": "query"},
        "semantic": "global",
    }
    assert classify_s1_query_family(nested) == "fine_grained_recognition"
    relational_reasoning = {
        "question": "same relation",
        "types": {"semantic": "rel", "structural": "compare"},
        "semantic": [{"operation": "same color", "argument": "x"}],
    }
    assert classify_s1_query_family(relational_reasoning) == "reasoning"


def test_whole_image_execution_order_keeps_groups_contiguous() -> None:
    records = [
        S1SampleRecord(
            image_id=f"i{image}",
            question_id=f"q{image}-{question}",
            query_text="question",
            answer="yes",
            query_family="reasoning",
            image_path=f"i{image}.png",
            image_file_sha256="hash",
            image_width=10,
            image_height=10,
            functional_metadata={},
        )
        for image in range(3)
        for question in range(2)
    ]
    ordered = whole_image_execution_order(records, seed=3)
    positions: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(ordered):
        positions[record.image_id].append(index)
    assert all(
        max(values) - min(values) + 1 == len(values) for values in positions.values()
    )


def test_manifest_validation_reports_counts() -> None:
    selected = _select(_pool())
    records = [
        S1SampleRecord(
            image_id=str(row["_image_id"]),
            question_id=str(row["_question_id"]),
            query_text=str(row["question"]),
            answer=str(row["answer"]),
            query_family=str(row["_query_family"]),
            image_path="image.png",
            image_file_sha256="hash",
            image_width=10,
            image_height=10,
            functional_metadata={},
        )
        for row in selected
    ]
    summary = validate_sampling_constraints(records, minimum_images_per_family=15)
    assert summary["image_count"] == 80
    assert summary["question_count"] == 480
    assert summary["repeated_image_family_cells"] == 240
