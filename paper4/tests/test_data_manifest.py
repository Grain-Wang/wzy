"""S0 sample selection and manifest-freeze tests."""

from pathlib import Path

from src.ciq_s0.data_manifest import (
    QUERY_FAMILIES,
    SampleRecord,
    classify_query_family,
    freeze_sample_manifest,
    randomized_image_group_order,
    select_s0_rows,
)


def _rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    definitions = (
        ("global", "query", "Is this indoors or outdoors?"),
        ("attr", "query", "What color is the car?"),
        ("rel", "query", "What is left of the tree?"),
        ("obj", "compare", "Are the two objects the same?"),
    )
    for image_index in range(8):
        for question_index, (semantic, structural, question) in enumerate(definitions):
            rows.append(
                {
                    "image_id": f"image-{image_index}",
                    "question_id": f"question-{image_index}-{question_index}",
                    "question": question,
                    "answer": "yes",
                    "semantic": semantic,
                    "structural": structural,
                }
            )
    return rows


def test_selection_is_deterministic_replicated_and_covers_families() -> None:
    first = select_s0_rows(_rows(), image_count=8, question_count=32, seed=17)
    second = select_s0_rows(_rows(), image_count=8, question_count=32, seed=17)
    assert [row["_question_id"] for row in first] == [
        row["_question_id"] for row in second
    ]
    assert len(first) == 32
    assert len({row["_image_id"] for row in first}) == 8
    assert {row["_query_family"] for row in first} == set(QUERY_FAMILIES)


def test_classifier_uses_nested_gqa_types_before_semantic_program() -> None:
    row = {
        "question": "What is left of the tree?",
        "semantic": ["select: tree", "relate: left"],
        "types": {
            "semantic": "rel",
            "structural": "query",
            "detailed": "rel",
        },
    }
    assert classify_query_family(row) == "spatial_relation"


def test_execution_order_shuffles_whole_image_groups_deterministically() -> None:
    records = [
        SampleRecord(
            image_id=str(image_id),
            question_id=f"{image_id}-{question_id}",
            query_text="question",
            answer="answer",
            query_family="reasoning",
            image_path=f"{image_id}.png",
            image_file_sha256="a" * 64,
            image_width=10,
            image_height=10,
        )
        for image_id in range(8)
        for question_id in range(4)
    ]
    first = randomized_image_group_order(records, seed=17)
    second = randomized_image_group_order(records, seed=17)
    assert [record.question_id for record in first] == [
        record.question_id for record in second
    ]
    image_order = list(dict.fromkeys(record.image_id for record in first))
    assert image_order != [str(index) for index in range(8)]
    assert all(
        len({record.image_id for record in first[offset : offset + 4]}) == 1
        for offset in range(0, len(first), 4)
    )


def test_manifest_freeze_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "sample_manifest.jsonl"
    record = SampleRecord(
        image_id="1",
        question_id="q1",
        query_text="What is shown?",
        answer="cat",
        query_family="global_perception",
        image_path="paper4/data/C01/S0/images/1.png",
        image_file_sha256="a" * 64,
        image_width=10,
        image_height=10,
        question_token_length=4,
        answer_token_length=1,
        preprocessing_sha256="b" * 64,
    )
    digest = freeze_sample_manifest(path, [record])
    assert len(digest) == 64
    assert path.with_suffix(".jsonl.sha256").is_file()
    try:
        freeze_sample_manifest(path, [record])
    except FileExistsError:
        pass
    else:
        raise AssertionError("frozen manifest was overwritten")
