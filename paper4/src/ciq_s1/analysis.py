"""Confirmatory C01-S1 statistics, figures, tables, and gate decision."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..ciq_s0.io_utils import read_json, write_json
from .profile_controls import policy_headroom_bootstrap
from .reporting import render_s1_report
from .statistics import (
    crossfit_group,
    image_grouped_bootstrap,
    pooled_partial_r2,
    within_image_permutation_test,
)
from .structure import (
    clustered_mean_interval,
    family_group_heatmap,
    jaccard_similarity,
    ranking_stability_bootstrap,
    same_image_pair_metrics,
    standardize_sensitivity_vectors,
    top_k_indices,
)


NON_QUERY_CONTROLS = (
    "global_static",
    "per_query_family",
    "leave_one_query_out_per_image",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _directory_size(path: Path) -> int:
    return (
        sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        if path.exists()
        else 0
    )


def _load_sweep(
    raw_directory: Path, module_manifest: Mapping[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[float]]]:
    rows_by_group: dict[str, list[dict[str, Any]]] = {}
    per_question: dict[str, dict[str, float]] = defaultdict(dict)
    group_ids = [str(group["group_id"]) for group in module_manifest["groups"]]
    for group_id in group_ids:
        path = raw_directory / f"single_group__{group_id.replace('.', '__')}.jsonl"
        rows = _read_jsonl(path)
        converted: list[dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            value["sensitivity"] = float(row["delta_nll"])
            converted.append(value)
            per_question[str(row["question_id"])][group_id] = float(row["delta_nll"])
        rows_by_group[group_id] = converted
    vectors = {
        question_id: [values[group_id] for group_id in group_ids]
        for question_id, values in per_question.items()
    }
    return rows_by_group, vectors


def _profile_execution_map(
    raw_directory: Path, definitions: Sequence[Mapping[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    signatures = sorted({str(item["profile_signature"]) for item in definitions})
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for signature in signatures:
        for row in _read_jsonl(
            raw_directory / "profiles" / f"profile__{signature}.jsonl"
        ):
            result[(str(row["question_id"]), signature)] = row
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty table")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _family_replication(
    definitions: Sequence[Mapping[str, Any]],
    executions: Mapping[tuple[str, str], Mapping[str, Any]],
    bf16: Mapping[str, Mapping[str, Any]],
    all_w4: Mapping[str, Mapping[str, Any]],
    *,
    budget: float,
    control: str,
) -> dict[str, Any]:
    lookup = {
        (str(item["question_id"]), str(item["policy"])): executions[
            (str(item["question_id"]), str(item["profile_signature"]))
        ]
        for item in definitions
        if float(item["budget_fraction"]) == budget
        and str(item["policy"]) in {"per_query_oracle", control}
    }
    family_questions: dict[str, list[str]] = defaultdict(list)
    for item in definitions:
        if (
            float(item["budget_fraction"]) == budget
            and str(item["policy"]) == "per_query_oracle"
        ):
            family_questions[str(item["query_family"])].append(str(item["question_id"]))
    result: dict[str, Any] = {}
    for family, question_ids in sorted(family_questions.items()):
        image_count = len(
            {
                str(item["image_id"])
                for item in definitions
                if float(item["budget_fraction"]) == budget
                and str(item["policy"]) == "per_query_oracle"
                and str(item["query_family"]) == family
            }
        )
        score_gain = float(
            np.mean(
                [
                    float(lookup[(item, "per_query_oracle")]["official_score"])
                    - float(lookup[(item, control)]["official_score"])
                    for item in question_ids
                ]
            )
        )
        denominator = float(
            np.mean([float(all_w4[item]["answer_nll"]) for item in question_ids])
            - np.mean([float(bf16[item]["answer_nll"]) for item in question_ids])
        )
        nll_gain = float(
            np.mean(
                [
                    float(lookup[(item, control)]["answer_nll"])
                    - float(lookup[(item, "per_query_oracle")]["answer_nll"])
                    for item in question_ids
                ]
            )
        )
        recovery_gain = nll_gain / denominator if denominator > 0 else float("nan")
        result[family] = {
            "sample_count": len(question_ids),
            "image_count": image_count,
            "official_score_headroom": score_gain,
            "relative_nll_recovery_headroom": recovery_gain,
            "directionally_consistent": score_gain >= 0
            and recovery_gain >= 0
            and (score_gain > 0 or recovery_gain > 0),
        }
    return result


def _confound_summary(
    sample_rows: Sequence[Mapping[str, Any]],
    sensitivity_vectors: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    magnitudes = np.asarray(
        [
            np.mean(np.abs(sensitivity_vectors[str(row["question_id"])]))
            for row in sample_rows
        ],
        dtype=np.float64,
    )
    correlations: dict[str, float] = {}
    for field in (
        "question_token_length",
        "answer_token_length",
        "visual_token_count",
    ):
        values = np.asarray([float(row[field]) for row in sample_rows])
        correlations[field] = (
            float(np.corrcoef(values, magnitudes)[0, 1])
            if values.std() > 0 and magnitudes.std() > 0
            else 0.0
        )
    return {
        "absolute_sensitivity_pearson_correlations": correlations,
        "prompt_hash_count": len(
            {str(row["prompt_template_sha256"]) for row in sample_rows}
        ),
        "taxonomy_uses_model_output": False,
        "samples_filtered_by_correctness": False,
        "vision_cache_enabled": False,
    }


def _make_figures(
    figure_directory: Path,
    *,
    sample_rows: Sequence[Mapping[str, Any]],
    group_ids: Sequence[str],
    heatmap: Mapping[str, Mapping[str, float]],
    pair_rows: Sequence[Mapping[str, Any]],
    headrooms: Sequence[Mapping[str, Any]],
    sensitivity_vectors: Mapping[str, Sequence[float]],
    heatmap_vectors: Mapping[str, Sequence[float]],
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    def font(size: int) -> Any:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()

    def save(canvas: Any, name: str) -> None:
        canvas.save(figure_directory / name, format="PNG", optimize=True)

    def diverging(value: float, limit: float) -> tuple[int, int, int]:
        strength = min(1.0, abs(value) / limit) if limit > 0 else 0.0
        pale = int(245 - 130 * strength)
        return (245, pale, pale) if value >= 0 else (pale, pale, 245)

    figure_directory.mkdir(parents=True, exist_ok=True)
    families = sorted(heatmap)
    family_counts = {
        family: sum(str(row["query_family"]) == family for row in sample_rows)
        for family in families
    }
    matrix = np.asarray(
        [
            [float(heatmap[family][group_id]) for group_id in group_ids]
            for family in families
        ]
    )
    by_family_image: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in sample_rows:
        by_family_image[str(row["query_family"])][str(row["image_id"])].append(
            str(row["question_id"])
        )
    generator = np.random.default_rng(20260906)
    heatmap_intervals: dict[tuple[str, str], tuple[float, float]] = {}
    for family_index, family in enumerate(families):
        family_images = sorted(by_family_image[family])
        for group_index, _ in enumerate(group_ids):
            bootstrap_values = []
            for _ in range(500):
                sampled = generator.choice(
                    family_images, size=len(family_images), replace=True
                )
                values = [
                    heatmap_vectors[question_id][group_index]
                    for image_id in sampled
                    for question_id in by_family_image[family][str(image_id)]
                ]
                bootstrap_values.append(float(np.mean(values)))
            lower, upper = np.quantile(bootstrap_values, [0.025, 0.975])
            heatmap_intervals[(family, group_ids[group_index])] = (
                float(lower),
                float(upper),
            )
    cell_width, cell_height = 120, 120
    x_origin, y_origin = 330, 135
    canvas = Image.new(
        "RGB",
        (x_origin + cell_width * len(group_ids) + 40, y_origin + 4 * cell_height + 80),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (25, 20),
        "Query type × 13-group standardized ΔNLL sensitivity (cell: mean, 95% image-bootstrap CI)",
        fill="black",
        font=font(24),
    )
    limit = float(np.max(np.abs(matrix))) or 1.0
    for column, group_id in enumerate(group_ids):
        draw.multiline_text(
            (x_origin + column * cell_width + 3, 58),
            group_id.replace("decoder.", "dec.\n").replace("vision.", "vis.\n"),
            fill="black",
            font=font(12),
            align="center",
        )
    for row_index, family in enumerate(families):
        draw.text(
            (10, y_origin + row_index * cell_height + 45),
            f"{family} (n={family_counts[family]})",
            fill="black",
            font=font(16),
        )
        for column, group_id in enumerate(group_ids):
            value = float(heatmap[family][group_id])
            lower, upper = heatmap_intervals[(family, group_id)]
            bounds = (
                x_origin + column * cell_width,
                y_origin + row_index * cell_height,
                x_origin + (column + 1) * cell_width,
                y_origin + (row_index + 1) * cell_height,
            )
            draw.rectangle(
                bounds, fill=diverging(value, limit), outline="white", width=2
            )
            draw.multiline_text(
                (bounds[0] + 5, bounds[1] + 35),
                f"{value:.3f}\n[{lower:.3f},{upper:.3f}]",
                fill="black",
                font=font(12),
                align="center",
            )
    save(canvas, "c01_s1_query_type_group_heatmap.png")

    same = [float(row["cosine_distance"]) for row in pair_rows if row["same_family"]]
    cross = [
        float(row["cosine_distance"]) for row in pair_rows if not row["same_family"]
    ]
    same_interval = clustered_mean_interval(
        [row for row in pair_rows if row["same_family"]],
        value_name="cosine_distance",
        replicates=1000,
        seed=20260906,
    )
    cross_interval = clustered_mean_interval(
        [row for row in pair_rows if not row["same_family"]],
        value_name="cosine_distance",
        replicates=1000,
        seed=20260906,
    )
    canvas = Image.new("RGB", (1200, 720), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (30, 20),
        "Same-image sensitivity-distance distribution",
        fill="black",
        font=font(28),
    )
    plot = (100, 100, 1130, 600)
    draw.rectangle(plot, outline="black", width=2)
    draw.text((plot[0], plot[3] + 5), "cosine distance 0", fill="black", font=font(14))
    draw.text((plot[2] - 120, plot[3] + 5), "2", fill="black", font=font(14))
    bins = np.linspace(0, 2, 31)
    same_hist, _ = np.histogram(same, bins=bins, density=True)
    cross_hist, _ = np.histogram(cross, bins=bins, density=True)
    maximum = max(float(same_hist.max()), float(cross_hist.max()), 1e-9)
    for histogram, color in ((same_hist, "#2468B4"), (cross_hist, "#D24A3A")):
        points = []
        for index, value in enumerate(histogram):
            x = plot[0] + int((index + 0.5) / len(histogram) * (plot[2] - plot[0]))
            y = plot[3] - int(float(value) / maximum * (plot[3] - plot[1]))
            points.append((x, y))
        draw.line(points, fill=color, width=4)
    draw.text(
        (110, 620),
        f"same family n={len(same)}: mean={same_interval['mean']:.3f}, "
        f"95% image-bootstrap CI [{same_interval['lower_95']:.3f}, {same_interval['upper_95']:.3f}]",
        fill="#2468B4",
        font=font(18),
    )
    draw.text(
        (110, 655),
        f"cross family n={len(cross)}: mean={cross_interval['mean']:.3f}, "
        f"95% image-bootstrap CI [{cross_interval['lower_95']:.3f}, {cross_interval['upper_95']:.3f}]",
        fill="#D24A3A",
        font=font(18),
    )
    save(canvas, "c01_s1_same_image_distance.png")

    budgets = [float(item["budget_fraction"]) * 100 for item in headrooms]
    score = [float(item["official_score_headroom"]) * 100 for item in headrooms]
    score_low = [
        (
            float(item["official_score_headroom"])
            - float(item["official_score_headroom_lower_95"])
        )
        * 100
        for item in headrooms
    ]
    score_high = [
        (
            float(item["official_score_headroom_upper_95"])
            - float(item["official_score_headroom"])
        )
        * 100
        for item in headrooms
    ]
    recovery = [
        float(item["relative_nll_recovery_headroom"]) * 100 for item in headrooms
    ]
    recovery_low = [
        (
            float(item["relative_nll_recovery_headroom"])
            - float(item["relative_nll_recovery_headroom_lower_95"])
        )
        * 100
        for item in headrooms
    ]
    recovery_high = [
        (
            float(item["relative_nll_recovery_headroom_upper_95"])
            - float(item["relative_nll_recovery_headroom"])
        )
        * 100
        for item in headrooms
    ]
    canvas = Image.new("RGB", (1500, 700), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (30, 20),
        f"Equal-byte oracle headroom; n={headrooms[0]['sample_count']}, "
        f"images={headrooms[0]['image_count']}; bars=95% image-bootstrap CI",
        fill="black",
        font=font(26),
    )

    def curve(
        bounds: tuple[int, int, int, int],
        values: Sequence[float],
        lows: Sequence[float],
        highs: Sequence[float],
        threshold: float,
        title: str,
    ) -> None:
        draw.rectangle(bounds, outline="black", width=2)
        minimum = min(
            [
                *values,
                *(value - error for value, error in zip(values, lows, strict=True)),
                0.0,
                threshold,
            ]
        )
        maximum = max(
            [
                *values,
                *(value + error for value, error in zip(values, highs, strict=True)),
                0.0,
                threshold,
            ]
        )
        margin = max(0.1, 0.1 * (maximum - minimum))
        minimum -= margin
        maximum += margin

        def point(index: int, value: float) -> tuple[int, int]:
            x = (
                bounds[0]
                + 80
                + int(index / max(1, len(values) - 1) * (bounds[2] - bounds[0] - 160))
            )
            y = (
                bounds[3]
                - 70
                - int(
                    (value - minimum)
                    / (maximum - minimum)
                    * (bounds[3] - bounds[1] - 140)
                )
            )
            return x, y

        threshold_y = point(0, threshold)[1]
        draw.line(
            (bounds[0], threshold_y, bounds[2], threshold_y), fill="#C02020", width=2
        )
        points = [point(index, value) for index, value in enumerate(values)]
        draw.line(points, fill="#1E5AA8", width=4)
        for index, (x, y) in enumerate(points):
            lower_y = point(index, values[index] - lows[index])[1]
            upper_y = point(index, values[index] + highs[index])[1]
            draw.line((x, lower_y, x, upper_y), fill="black", width=2)
            draw.line((x - 7, lower_y, x + 7, lower_y), fill="black", width=2)
            draw.line((x - 7, upper_y, x + 7, upper_y), fill="black", width=2)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#1E5AA8")
            draw.multiline_text(
                (x - 48, y - 58),
                f"{values[index]:.2f}\n"
                f"[{values[index] - lows[index]:+.2f},"
                f"{values[index] + highs[index]:+.2f}]",
                fill="#1E5AA8",
                font=font(12),
                align="center",
            )
            draw.text(
                (x - 18, bounds[3] - 55),
                f"{budgets[index]:.0f}%",
                fill="black",
                font=font(16),
            )
        draw.text((bounds[0] + 15, bounds[1] + 12), title, fill="black", font=font(20))
        draw.text(
            (bounds[0] + 15, threshold_y - 25),
            f"threshold={threshold:g}",
            fill="#C02020",
            font=font(14),
        )

    curve(
        (40, 90, 730, 650),
        score,
        score_low,
        score_high,
        1.5,
        "Official-score headroom (points)",
    )
    curve(
        (770, 90, 1460, 650),
        recovery,
        recovery_low,
        recovery_high,
        15.0,
        "Relative NLL-recovery headroom (pp)",
    )
    save(canvas, "c01_s1_equal_byte_oracle_headroom.png")

    family_top: dict[str, frozenset[int]] = {}
    for family in families:
        question_ids = [
            str(row["question_id"])
            for row in sample_rows
            if str(row["query_family"]) == family
        ]
        mean_vector = np.asarray(
            [sensitivity_vectors[item] for item in question_ids]
        ).mean(axis=0)
        family_top[family] = top_k_indices(mean_vector, top_k=3)
    overlap = np.asarray(
        [
            [
                jaccard_similarity(family_top[left], family_top[right])
                for right in families
            ]
            for left in families
        ]
    )
    overlap_intervals: dict[tuple[int, int], tuple[float, float]] = {}
    for row_index in range(len(families)):
        for column_index in range(len(families)):
            if row_index == column_index:
                lower, upper = 1.0, 1.0
            else:
                pair_values = []
                for _ in range(500):
                    selected_sets = []
                    for family in (families[row_index], families[column_index]):
                        family_images = sorted(by_family_image[family])
                        sampled = generator.choice(
                            family_images, size=len(family_images), replace=True
                        )
                        question_ids = [
                            question_id
                            for image_id in sampled
                            for question_id in by_family_image[family][str(image_id)]
                        ]
                        vector = np.asarray(
                            [sensitivity_vectors[item] for item in question_ids]
                        ).mean(axis=0)
                        selected_sets.append(top_k_indices(vector, top_k=3))
                    pair_values.append(
                        jaccard_similarity(selected_sets[0], selected_sets[1])
                    )
                lower, upper = np.quantile(pair_values, [0.025, 0.975])
            overlap_intervals[(row_index, column_index)] = (float(lower), float(upper))
    canvas = Image.new("RGB", (1200, 1000), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (30, 20),
        "Top-3 protected-group overlap (Jaccard; cell text: full estimate, 95% image-bootstrap CI)",
        fill="black",
        font=font(24),
    )
    cell = 170
    x_origin, y_origin = 360, 220
    for index, family in enumerate(families):
        label = f"{family}\n(n={family_counts[family]})"
        draw.multiline_text(
            (x_origin + index * cell, 100), label, fill="black", font=font(15)
        )
        draw.multiline_text(
            (20, y_origin + index * cell + 50), label, fill="black", font=font(15)
        )
    for row_index in range(len(families)):
        for column_index in range(len(families)):
            value = float(overlap[row_index, column_index])
            lower, upper = overlap_intervals[(row_index, column_index)]
            bounds = (
                x_origin + column_index * cell,
                y_origin + row_index * cell,
                x_origin + (column_index + 1) * cell,
                y_origin + (row_index + 1) * cell,
            )
            shade = int(245 - 160 * value)
            draw.rectangle(bounds, fill=(shade, shade, 250), outline="white", width=3)
            draw.multiline_text(
                (bounds[0] + 25, bounds[1] + 55),
                f"{value:.2f}\n[{lower:.2f},{upper:.2f}]",
                fill="black",
                font=font(17),
                align="center",
            )
    save(canvas, "c01_s1_topk_overlap_matrix.png")


def analyze_s1(config_path: Path, repository_root: Path) -> dict[str, Any]:
    """Reconstruct every confirmatory statistic and apply the locked gate."""
    config = read_json(config_path)
    raw_directory = repository_root / str(config["paths"]["raw_output"])
    processed_directory = repository_root / str(config["paths"]["processed_output"])
    table_directory = repository_root / str(config["paths"]["table_output"])
    figure_directory = repository_root / str(config["paths"]["figure_output"])
    profile_summary = read_json(processed_directory / "s1_profile_summary.json")
    if profile_summary["outcome"] != "PASS":
        raise RuntimeError("S1 analysis requires completed executed profiles")
    module_manifest = read_json(raw_directory / "module_manifest.json")
    group_ids = [str(group["group_id"]) for group in module_manifest["groups"]]
    sample_rows = _read_jsonl(raw_directory / "formal_manifest.jsonl")
    rows_by_group, sensitivity_vectors = _load_sweep(raw_directory, module_manifest)
    js_rows_by_group = {
        group_id: [{**row, "sensitivity": float(row["js_divergence"])} for row in rows]
        for group_id, rows in rows_by_group.items()
    }
    crossfit_results = {
        group_id: crossfit_group(
            rows,
            fold_count=int(config["analysis"]["cross_validation_folds"]),
            seed=int(config["analysis"]["seed"]),
        )
        for group_id, rows in rows_by_group.items()
    }
    partial_r2 = pooled_partial_r2(list(crossfit_results.values()))
    bootstrap = image_grouped_bootstrap(
        list(crossfit_results.values()),
        replicates=int(config["analysis"]["bootstrap_replicates"]),
        seed=int(config["analysis"]["seed"]),
    )
    permutation = within_image_permutation_test(
        rows_by_group,
        fold_count=int(config["analysis"]["cross_validation_folds"]),
        permutations=int(config["analysis"]["permutation_replicates"]),
        seed=int(config["analysis"]["seed"]),
    )
    js_crossfit_results = {
        group_id: crossfit_group(
            rows,
            fold_count=int(config["analysis"]["cross_validation_folds"]),
            seed=int(config["analysis"]["seed"]),
        )
        for group_id, rows in js_rows_by_group.items()
    }
    js_partial_r2 = pooled_partial_r2(list(js_crossfit_results.values()))
    js_bootstrap = image_grouped_bootstrap(
        list(js_crossfit_results.values()),
        replicates=int(config["analysis"]["bootstrap_replicates"]),
        seed=int(config["analysis"]["seed"]),
    )
    js_permutation = within_image_permutation_test(
        js_rows_by_group,
        fold_count=int(config["analysis"]["cross_validation_folds"]),
        permutations=int(config["analysis"]["permutation_replicates"]),
        seed=int(config["analysis"]["seed"]),
    )
    pair_rows = same_image_pair_metrics(sample_rows, sensitivity_vectors, top_k=3)
    same_rows = [row for row in pair_rows if bool(row["same_family"])]
    cross_rows = [row for row in pair_rows if not bool(row["same_family"])]
    pair_summary = {
        "same_family": {
            metric: clustered_mean_interval(
                same_rows,
                value_name=metric,
                replicates=int(config["analysis"]["bootstrap_replicates"]),
                seed=int(config["analysis"]["seed"]),
            )
            for metric in (
                "spearman",
                "cosine_distance",
                "top_k_jaccard",
                "protected_group_identity_change",
            )
        },
        "cross_family": {
            metric: clustered_mean_interval(
                cross_rows,
                value_name=metric,
                replicates=int(config["analysis"]["bootstrap_replicates"]),
                seed=int(config["analysis"]["seed"]),
            )
            for metric in (
                "spearman",
                "cosine_distance",
                "top_k_jaccard",
                "protected_group_identity_change",
            )
        },
        "same_pair_count": len(same_rows),
        "cross_pair_count": len(cross_rows),
    }
    ranking_stability = ranking_stability_bootstrap(
        sample_rows,
        sensitivity_vectors,
        top_k=3,
        replicates=int(config["analysis"]["bootstrap_replicates"]),
        seed=int(config["analysis"]["seed"]),
    )
    standardized_vectors = standardize_sensitivity_vectors(sensitivity_vectors)
    heatmap = family_group_heatmap(sample_rows, standardized_vectors, group_ids)

    definitions = _read_jsonl(raw_directory / "profile_definitions.jsonl")
    executions = _profile_execution_map(raw_directory, definitions)
    bf16 = {
        str(row["question_id"]): row
        for row in _read_jsonl(raw_directory / "bf16_predictions.jsonl")
    }
    all_w4 = {
        str(row["question_id"]): row
        for row in _read_jsonl(raw_directory / "all_w4_predictions.jsonl")
    }
    recoverable_ids = [
        question_id
        for question_id, row in all_w4.items()
        if bool(bf16[question_id]["official_score"]) and not bool(row["official_score"])
    ]
    sample_by_question = {str(row["question_id"]): row for row in sample_rows}
    recoverable_family_counts: dict[str, int] = defaultdict(int)
    for question_id in recoverable_ids:
        recoverable_family_counts[
            str(sample_by_question[question_id]["query_family"])
        ] += 1
    aggregates = profile_summary["aggregates"]
    headrooms: list[dict[str, Any]] = []
    for budget in [float(item) for item in config["profiles"]["budget_fractions"]]:
        candidates = [
            item
            for item in aggregates
            if float(item["budget_fraction"]) == budget
            and str(item["policy"]) in NON_QUERY_CONTROLS
        ]
        strongest = max(
            candidates,
            key=lambda item: (
                float(item["official_score"]),
                float(item["relative_nll_recovery"]),
            ),
        )
        headrooms.append(
            policy_headroom_bootstrap(
                definitions,
                executions,
                bf16,
                all_w4,
                budget_fraction=budget,
                oracle_policy="per_query_oracle",
                control_policy=str(strongest["policy"]),
                replicates=int(config["analysis"]["bootstrap_replicates"]),
                seed=int(config["analysis"]["seed"]),
            )
        )
    best_headroom = max(
        headrooms,
        key=lambda item: (
            float(item["official_score_headroom"]),
            float(item["relative_nll_recovery_headroom"]),
        ),
    )
    replication = _family_replication(
        definitions,
        executions,
        bf16,
        all_w4,
        budget=float(best_headroom["budget_fraction"]),
        control=str(best_headroom["control_policy"]),
    )
    adequate_replications = sum(
        bool(value["directionally_consistent"])
        for value in replication.values()
        if int(value["image_count"])
        >= int(config["dataset"]["minimum_images_per_family"])
    )
    ranking_is_stable = (
        float(ranking_stability["median_jaccard"]) >= 0.5
        and float(ranking_stability["lower_95_jaccard"]) >= 0.2
    )
    interaction_pass = bool(
        partial_r2 >= 0.10
        and float(bootstrap["lower_95"]) > 0.05
        and float(permutation["p_value"]) < 0.01
    )
    profile_pass = bool(
        float(best_headroom["official_score_headroom"]) >= 0.015
        and float(best_headroom["relative_nll_recovery_headroom"]) >= 0.15
        and float(best_headroom["official_score_headroom_lower_95"]) > 0
        and float(best_headroom["relative_nll_recovery_headroom_lower_95"]) > 0
    )
    strong_pass = bool(
        interaction_pass
        and profile_pass
        and adequate_replications >= 2
        and ranking_is_stable
    )
    robust_negative = bool(
        float(bootstrap["upper_95"]) < 0.05
        and all(float(item["official_score_headroom"]) < 0.005 for item in headrooms)
        and all(
            float(item["relative_nll_recovery_headroom"]) < 0.05 for item in headrooms
        )
    )
    outcome = (
        "STRONG PASS"
        if strong_pass
        else ("NEGATIVE" if robust_negative else "INCONCLUSIVE")
    )
    confounds = _confound_summary(sample_rows, sensitivity_vectors)
    summary = {
        "outcome": outcome,
        "interaction": {
            "partial_r2": partial_r2,
            "bootstrap": bootstrap,
            "permutation": permutation,
            "per_group_partial_r2": {
                group_id: result.partial_r2
                for group_id, result in crossfit_results.items()
            },
        },
        "js_interaction": {
            "partial_r2": js_partial_r2,
            "bootstrap": js_bootstrap,
            "permutation": js_permutation,
            "per_group_partial_r2": {
                group_id: result.partial_r2
                for group_id, result in js_crossfit_results.items()
            },
        },
        "same_image_structure": pair_summary,
        "ranking_stability": ranking_stability,
        "heatmap": heatmap,
        "profile_aggregates": aggregates,
        "headrooms": headrooms,
        "best_headroom": best_headroom,
        "query_family_replication": replication,
        "adequate_directional_replication_count": adequate_replications,
        "confounds": confounds,
        "recoverable_tail": {
            "bf16_correct_w4_wrong_count": len(recoverable_ids),
            "fraction": len(recoverable_ids) / len(all_w4),
            "family_counts": dict(sorted(recoverable_family_counts.items())),
            "worth_bounded_backup_probe": len(recoverable_ids) >= 20
            and len(recoverable_family_counts) >= 2,
        },
        "gate_components": {
            "interaction_pass": interaction_pass,
            "profile_pass": profile_pass,
            "replication_pass": adequate_replications >= 2,
            "ranking_stability_pass": ranking_is_stable,
            "robust_negative": robust_negative,
        },
        "storage_bytes": {
            "raw": _directory_size(raw_directory),
            "processed": _directory_size(processed_directory),
            "tables": _directory_size(table_directory),
            "figures_before_write": _directory_size(figure_directory),
        },
    }
    write_json(processed_directory / "s1_confirmatory_summary.json", summary)
    _write_csv(
        table_directory / "c01_s1_interaction.csv",
        [
            {"group_id": group_id, "partial_r2": result.partial_r2}
            for group_id, result in crossfit_results.items()
        ],
    )
    _write_csv(table_directory / "c01_s1_profiles.csv", aggregates)
    _write_csv(
        table_directory / "c01_s1_sensitivity_heatmap.csv",
        [
            {"query_family": family, **values}
            for family, values in sorted(heatmap.items())
        ],
    )
    _make_figures(
        figure_directory,
        sample_rows=sample_rows,
        group_ids=group_ids,
        heatmap=heatmap,
        pair_rows=pair_rows,
        headrooms=headrooms,
        sensitivity_vectors=sensitivity_vectors,
        heatmap_vectors=standardized_vectors,
    )
    summary["storage_bytes"] = {
        "raw": _directory_size(raw_directory),
        "processed": _directory_size(processed_directory),
        "tables": _directory_size(table_directory),
        "figures": _directory_size(figure_directory),
        "materialized_images": _directory_size(
            repository_root / str(config["paths"]["image_directory"])
        ),
    }
    write_json(processed_directory / "s1_confirmatory_summary.json", summary)
    render_s1_report(repository_root=repository_root, config=config, summary=summary)
    return summary
