"""Integrity aggregation and sanity figures for C02-S0."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from ..ciq_s0.io_utils import read_json, write_json, write_jsonl


CONDITIONS = (
    "bf16_original",
    "bf16_relevant_cf",
    "bf16_irrelevant_cf",
    "w4_original",
    "w4_relevant_cf",
    "w4_irrelevant_cf",
)


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty collection")
    value = sum(values) / len(values)
    if not math.isfinite(value):
        raise RuntimeError("non-finite aggregate")
    return value


def calculate_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calculate registered relevant/irrelevant effects and interaction Delta."""
    by_expression: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        expression_id = int(row["expression_id"])
        condition = str(row["condition"])
        if condition in by_expression[expression_id]:
            raise ValueError(f"duplicate {condition} for expression {expression_id}")
        by_expression[expression_id][condition] = row

    effects: list[dict[str, Any]] = []
    for expression_id, conditions in sorted(by_expression.items()):
        if set(conditions) != set(CONDITIONS):
            missing = sorted(set(CONDITIONS) - set(conditions))
            raise ValueError(f"expression {expression_id} lacks conditions: {missing}")
        nll = {
            condition: float(conditions[condition]["answer_nll"])
            for condition in CONDITIONS
        }
        if not all(math.isfinite(value) for value in nll.values()):
            raise RuntimeError(f"non-finite NLL for expression {expression_id}")
        bf_relevant = nll["bf16_relevant_cf"] - nll["bf16_original"]
        bf_irrelevant = nll["bf16_irrelevant_cf"] - nll["bf16_original"]
        w4_relevant = nll["w4_relevant_cf"] - nll["w4_original"]
        w4_irrelevant = nll["w4_irrelevant_cf"] - nll["w4_original"]
        delta = (w4_relevant - w4_irrelevant) - (bf_relevant - bf_irrelevant)
        effects.append(
            {
                "bf16_irrelevant_effect": bf_irrelevant,
                "bf16_relevant_effect": bf_relevant,
                "delta": delta,
                "expression_id": expression_id,
                "image_id": int(conditions["bf16_original"]["image_id"]),
                "semantic_family": str(conditions["bf16_original"]["semantic_family"]),
                "target": str(conditions["bf16_original"]["target"]),
                "w4_irrelevant_effect": w4_irrelevant,
                "w4_relevant_effect": w4_relevant,
            }
        )
    return effects


def _reference_distribution_valid(row: dict[str, Any]) -> bool:
    reference = row.get("reference_distribution")
    if not isinstance(reference, dict):
        return False
    probabilities = reference.get("probabilities")
    token_ids = reference.get("token_ids")
    if not isinstance(probabilities, list) or not isinstance(token_ids, list):
        return False
    if len(probabilities) != len(token_ids) or not probabilities:
        return False
    return all(
        len(probability_row) == len(token_row) + 1
        and all(
            math.isfinite(float(value)) and float(value) >= 0
            for value in probability_row
        )
        and abs(sum(float(value) for value in probability_row) - 1.0) <= 1e-5
        for probability_row, token_row in zip(probabilities, token_ids, strict=True)
    )


def summarize_s0(
    *,
    rows: list[dict[str, Any]],
    effects: list[dict[str, Any]],
    pixel_summary: dict[str, Any],
    preflight_summary: dict[str, Any],
    remaining_summary: dict[str, Any],
    manifest_hash: str,
) -> dict[str, Any]:
    """Build the machine-readable C02-S0 integrity decision."""
    condition_counts = Counter(str(row["condition"]) for row in rows)
    expected_count = len(effects)
    nll_finite = all(math.isfinite(float(row["answer_nll"])) for row in rows)
    nll_alignment = all(
        int(row["answer_token_count"]) > 0 and int(row["prompt_token_count"]) > 0
        for row in rows
    )
    original_rows = [row for row in rows if row["condition"] == "bf16_original"]
    comparison_rows = [row for row in rows if row["condition"] != "bf16_original"]
    js_finite = all(
        row.get("js_divergence") is not None
        and math.isfinite(float(row["js_divergence"]))
        and 0.0 <= float(row["js_divergence"]) <= math.log(2.0) + 1e-6
        for row in comparison_rows
    )
    reference_distributions_valid = all(
        _reference_distribution_valid(row) for row in original_rows
    )
    target_hashes: dict[int, set[str]] = defaultdict(set)
    target_text: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        expression_id = int(row["expression_id"])
        target_hashes[expression_id].add(str(row["target_token_sha256"]))
        target_text[expression_id].add(str(row["target"]))
    target_identity = all(
        len(values) == 1 for values in target_hashes.values()
    ) and all(len(values) == 1 for values in target_text.values())
    formula_check = all(
        math.isclose(
            float(effect["delta"]),
            (
                float(effect["w4_relevant_effect"])
                - float(effect["w4_irrelevant_effect"])
            )
            - (
                float(effect["bf16_relevant_effect"])
                - float(effect["bf16_irrelevant_effect"])
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for effect in effects
    )
    delta_finite_rate = sum(
        math.isfinite(float(effect["delta"])) for effect in effects
    ) / len(effects)
    complete_conditions = all(
        condition_counts[condition] == expected_count for condition in CONDITIONS
    )
    nll_integrity = bool(
        nll_finite
        and nll_alignment
        and preflight_summary["repeatability_pass"]
        and remaining_summary["repeatability_pass"]
        and formula_check
        and complete_conditions
    )
    js_integrity = bool(
        js_finite
        and reference_distributions_valid
        and remaining_summary["js_repeatability_pass"]
    )
    cache_isolation = bool(remaining_summary["cache_isolation"]["pass"])
    pixel_integrity = all(
        bool(pixel_summary[key])
        for key in (
            "relevant_pixel_integrity",
            "irrelevant_pixel_integrity",
            "area_match",
            "deterministic_counterfactuals",
            "non_overlap",
        )
    )
    all_integrity = bool(
        nll_integrity
        and js_integrity
        and target_identity
        and cache_isolation
        and pixel_integrity
        and delta_finite_rate == 1.0
    )
    outcome = "PASS" if all_integrity else "INTEGRITY FAILURE"
    summary = {
        "area_match": bool(pixel_summary["area_match"]),
        "bf16_original_accuracy": float(preflight_summary["accuracy"]),
        "cache_leakage": not cache_isolation,
        "condition_counts": dict(sorted(condition_counts.items())),
        "critical_bugs": list(remaining_summary.get("critical_bugs", [])),
        "critical_remaining_concern": (
            "S0 is an integrity smoke test and supplies no scientific effect claim."
        ),
        "delta_finite_rate": delta_finite_rate,
        "delta_mean": _mean([float(row["delta"]) for row in effects]),
        "deterministic_counterfactuals": bool(
            pixel_summary["deterministic_counterfactuals"]
        ),
        "expression_count": len(effects),
        "gpu_hours": float(preflight_summary["gpu_hours"])
        + float(remaining_summary["gpu_hours"]),
        "image_count": len({int(row["image_id"]) for row in original_rows}),
        "irrelevant_cf_pixel_integrity": bool(
            pixel_summary["irrelevant_pixel_integrity"]
        ),
        "js_integrity": js_integrity,
        "manifest_sha256": manifest_hash,
        "mean_bf16_irrelevant_effect": _mean(
            [float(row["bf16_irrelevant_effect"]) for row in effects]
        ),
        "mean_bf16_relevant_effect": _mean(
            [float(row["bf16_relevant_effect"]) for row in effects]
        ),
        "mean_w4_irrelevant_effect": _mean(
            [float(row["w4_irrelevant_effect"]) for row in effects]
        ),
        "mean_w4_relevant_effect": _mean(
            [float(row["w4_relevant_effect"]) for row in effects]
        ),
        "nll_integrity": nll_integrity,
        "non_overlap": bool(pixel_summary["non_overlap"]),
        "outcome": outcome,
        "peak_memory_bytes": max(
            int(preflight_summary["peak_memory_bytes"]),
            int(remaining_summary["peak_memory_bytes"]),
        ),
        "recommendation": "ENTER C02-S1" if outcome == "PASS" else "STOP",
        "relevant_cf_pixel_integrity": bool(pixel_summary["relevant_pixel_integrity"]),
        "sign_convention_verified": formula_check,
        "target_identity": target_identity,
        "task_formulation": str(preflight_summary["task_formulation"]),
    }
    return summary


def create_effect_figure(path: Path, summary: dict[str, Any]) -> None:
    """Create a dependency-light sanity bar chart for the four mean effects."""
    labels = ("BF rel", "BF irr", "W4 rel", "W4 irr")
    values = (
        float(summary["mean_bf16_relevant_effect"]),
        float(summary["mean_bf16_irrelevant_effect"]),
        float(summary["mean_w4_relevant_effect"]),
        float(summary["mean_w4_irrelevant_effect"]),
    )
    canvas = Image.new("RGB", (1000, 600), "white")
    draw = ImageDraw.Draw(canvas)
    left, right, top, bottom = 100, 950, 90, 500
    maximum = max(0.05, *(abs(value) for value in values)) * 1.2
    zero_y = top + round((maximum / (2 * maximum)) * (bottom - top))
    draw.line((left, zero_y, right, zero_y), fill="black", width=2)
    draw.line((left, top, left, bottom), fill="black", width=2)
    colors = ("#4C78A8", "#9ECAE1", "#F58518", "#FFBF79")
    bar_width = 120
    gap = 70
    for index, (label, value, color) in enumerate(
        zip(labels, values, colors, strict=True)
    ):
        x0 = left + 80 + index * (bar_width + gap)
        height = round(abs(value) / (2 * maximum) * (bottom - top))
        y0, y1 = (zero_y - height, zero_y) if value >= 0 else (zero_y, zero_y + height)
        draw.rectangle((x0, y0, x0 + bar_width, y1), fill=color, outline="black")
        draw.text((x0 + 20, bottom + 20), label, fill="black")
        draw.text(
            (x0 + 8, y0 - 22 if value >= 0 else y1 + 5), f"{value:.5f}", fill="black"
        )
    draw.text((left, 25), "C02-S0 mean NLL effects (sanity only)", fill="black")
    draw.text(
        (left, 50), f"Delta mean: {float(summary['delta_mean']):.6f}", fill="black"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=False)


def load_and_analyze(
    *,
    raw_directory: Path,
    processed_directory: Path,
    figure_directory: Path,
    manifest_hash: str,
) -> dict[str, Any]:
    """Load completed artifacts, aggregate effects, and persist the S0 summary."""
    rows: list[dict[str, Any]] = []
    for name in ("bf16_original.jsonl", "remaining_conditions.jsonl"):
        rows.extend(
            json.loads(line)
            for line in (raw_directory / name).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    effects = calculate_effects(rows)
    summary = summarize_s0(
        rows=rows,
        effects=effects,
        pixel_summary=read_json(processed_directory / "pixel_integrity_summary.json"),
        preflight_summary=read_json(
            processed_directory / "bf16_preflight_summary.json"
        ),
        remaining_summary=read_json(processed_directory / "remaining_integrity.json"),
        manifest_hash=manifest_hash,
    )
    write_jsonl(processed_directory / "effects.jsonl", effects)
    write_json(processed_directory / "summary.json", summary)
    create_effect_figure(figure_directory / "mean_effects.png", summary)
    return summary
