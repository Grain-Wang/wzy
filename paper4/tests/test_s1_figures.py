"""Smoke-test the dependency-free scientific figure renderer."""

from __future__ import annotations

from pathlib import Path

from src.ciq_s1.analysis import _make_figures
from src.ciq_s1.structure import same_image_pair_metrics


def test_required_figures_render_without_matplotlib(tmp_path: Path) -> None:
    families = (
        "fine_grained_recognition",
        "global_perception",
        "reasoning",
        "spatial_relation",
    )
    group_ids = [f"g{index}" for index in range(13)]
    rows = []
    vectors = {}
    for image in range(8):
        for question in range(6):
            question_id = f"q{image}-{question}"
            family = families[(question // 2 + image) % len(families)]
            rows.append(
                {
                    "image_id": f"i{image}",
                    "question_id": question_id,
                    "query_family": family,
                }
            )
            vectors[question_id] = [
                ((image + 1) * (question + 2) * (group + 3) % 17 - 8) / 10
                for group in range(13)
            ]
    heatmap = {
        family: {
            group_id: sum(
                vectors[str(row["question_id"])][group_index]
                for row in rows
                if row["query_family"] == family
            )
            / sum(row["query_family"] == family for row in rows)
            for group_index, group_id in enumerate(group_ids)
        }
        for family in families
    }
    pairs = same_image_pair_metrics(rows, vectors, top_k=3)
    headrooms = [
        {
            "budget_fraction": budget,
            "sample_count": 48,
            "image_count": 8,
            "official_score_headroom": value,
            "official_score_headroom_lower_95": value - 0.005,
            "official_score_headroom_upper_95": value + 0.005,
            "relative_nll_recovery_headroom": value * 5,
            "relative_nll_recovery_headroom_lower_95": value * 5 - 0.02,
            "relative_nll_recovery_headroom_upper_95": value * 5 + 0.02,
        }
        for budget, value in zip((0.05, 0.10, 0.20), (0.01, 0.02, 0.03), strict=True)
    ]
    _make_figures(
        tmp_path,
        sample_rows=rows,
        group_ids=group_ids,
        heatmap=heatmap,
        pair_rows=pairs,
        headrooms=headrooms,
        sensitivity_vectors=vectors,
        heatmap_vectors=vectors,
    )
    expected = {
        "c01_s1_query_type_group_heatmap.png",
        "c01_s1_same_image_distance.png",
        "c01_s1_equal_byte_oracle_headroom.png",
        "c01_s1_topk_overlap_matrix.png",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    assert all((tmp_path / name).stat().st_size > 1000 for name in expected)
