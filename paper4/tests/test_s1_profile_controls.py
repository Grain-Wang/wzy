"""Tests for leakage-controlled fixed-budget profile definitions."""

from __future__ import annotations

from src.ciq_s1.profile_controls import (
    build_profile_definitions,
    select_execution_images,
)


def _manifest() -> dict[str, object]:
    return {
        "groups": [
            {"group_id": "g1", "w4_bytes": 100, "w8_bytes": 120, "bf16_bytes": 160},
            {"group_id": "g2", "w4_bytes": 100, "w8_bytes": 120, "bf16_bytes": 160},
        ]
    }


def test_profile_control_construction_is_deterministic_and_complete() -> None:
    rows = []
    sensitivities = {}
    families = ("reasoning", "spatial_relation", "fine_grained_recognition")
    for image in range(4):
        for family in families:
            for repeat in range(2):
                question_id = f"q-{image}-{family}-{repeat}"
                rows.append(
                    {
                        "image_id": str(image),
                        "question_id": question_id,
                        "query_family": family,
                    }
                )
                sensitivities[question_id] = {
                    "g1": float(image + repeat),
                    "g2": float(repeat),
                }
    execution = select_execution_images(
        [str(index) for index in range(4)], image_count=2, seed=1
    )
    first = build_profile_definitions(
        rows,
        sensitivities,
        _manifest(),
        execution_image_ids=execution,
        budget_fractions=[0.1, 0.2],
        seed=1,
    )
    second = build_profile_definitions(
        rows,
        sensitivities,
        _manifest(),
        execution_image_ids=execution,
        budget_fractions=[0.1, 0.2],
        seed=1,
    )
    assert first == second
    assert len(first) == 2 * 6 * 6 * 2
    assert {item["policy"] for item in first} == {
        "global_static",
        "per_query_family",
        "leave_one_query_out_per_image",
        "per_image_family",
        "per_query_oracle",
        "random_equal_byte",
    }
    assert all(item["added_bytes"] <= item["budget_bytes"] for item in first)
    random_rows = [item for item in first if item["policy"] == "random_equal_byte"]
    assert all(item["estimated_gain"] <= 3.0 for item in random_rows)
