"""Leakage-controlled profile definitions and executed-profile aggregation."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .profiles import (
    additive_gain_for_assignments,
    construct_profile,
    deterministic_random_gains,
)


POLICIES = (
    "global_static",
    "per_query_family",
    "leave_one_query_out_per_image",
    "per_image_family",
    "per_query_oracle",
    "random_equal_byte",
)


def select_execution_images(
    image_ids: Sequence[str], *, image_count: int, seed: int
) -> tuple[str, ...]:
    """Select the held-out profile-execution images without model outputs."""
    unique = sorted(
        set(image_ids),
        key=lambda value: hashlib.sha256(
            f"{seed}:profile:{value}".encode()
        ).hexdigest(),
    )
    if not 1 <= image_count < len(unique):
        raise ValueError("profile execution image count must be a proper subset")
    return tuple(unique[:image_count])


def _mean_vector(
    question_ids: Sequence[str],
    sensitivities: Mapping[str, Mapping[str, float]],
    group_ids: Sequence[str],
) -> dict[str, float]:
    if not question_ids:
        raise ValueError("cannot form a control from an empty support set")
    return {
        group_id: float(
            np.mean(
                [sensitivities[question_id][group_id] for question_id in question_ids]
            )
        )
        for group_id in group_ids
    }


def build_profile_definitions(
    sample_rows: Sequence[Mapping[str, Any]],
    sensitivities: Mapping[str, Mapping[str, float]],
    manifest: dict[str, Any],
    *,
    execution_image_ids: Sequence[str],
    budget_fractions: Sequence[float],
    seed: int,
) -> list[dict[str, Any]]:
    """Construct all six fixed-budget policies with explicit leakage controls."""
    group_ids = [str(group["group_id"]) for group in manifest["groups"]]
    if set(sensitivities) != {str(row["question_id"]) for row in sample_rows}:
        raise ValueError("sensitivity matrix does not cover the formal manifest")
    execution_images = set(execution_image_ids)
    construction = [
        row for row in sample_rows if str(row["image_id"]) not in execution_images
    ]
    evaluation = [
        row for row in sample_rows if str(row["image_id"]) in execution_images
    ]
    global_vector = _mean_vector(
        [str(row["question_id"]) for row in construction], sensitivities, group_ids
    )
    task_vectors = {
        family: _mean_vector(
            [
                str(row["question_id"])
                for row in construction
                if str(row["query_family"]) == family
            ],
            sensitivities,
            group_ids,
        )
        for family in sorted({str(row["query_family"]) for row in evaluation})
    }
    definitions: list[dict[str, Any]] = []
    for row in evaluation:
        question_id = str(row["question_id"])
        image_id = str(row["image_id"])
        family = str(row["query_family"])
        image_support = [
            str(other["question_id"])
            for other in evaluation
            if str(other["image_id"]) == image_id
            and str(other["question_id"]) != question_id
        ]
        cell_support = [
            str(other["question_id"])
            for other in evaluation
            if str(other["image_id"]) == image_id
            and str(other["query_family"]) == family
            and str(other["question_id"]) != question_id
        ]
        policy_vectors = {
            "global_static": global_vector,
            "per_query_family": task_vectors[family],
            "leave_one_query_out_per_image": _mean_vector(
                image_support, sensitivities, group_ids
            ),
            "per_image_family": _mean_vector(cell_support, sensitivities, group_ids),
            "per_query_oracle": dict(sensitivities[question_id]),
            "random_equal_byte": deterministic_random_gains(
                group_ids, seed=seed, key=question_id
            ),
        }
        for policy in POLICIES:
            for budget_fraction in budget_fractions:
                profile = construct_profile(
                    manifest,
                    policy_vectors[policy],
                    budget_fraction=float(budget_fraction),
                    seed=seed,
                )
                profile_payload = profile.to_dict()
                profile_payload["estimated_gain"] = additive_gain_for_assignments(
                    profile.assignments,
                    sensitivities[question_id],
                )
                signature = hashlib.sha256(
                    repr(sorted(profile.assignments.items())).encode()
                ).hexdigest()[:16]
                definitions.append(
                    {
                        "question_id": question_id,
                        "image_id": image_id,
                        "query_family": family,
                        "policy": policy,
                        "budget_fraction": float(budget_fraction),
                        "profile_signature": signature,
                        **profile_payload,
                    }
                )
    return definitions


def aggregate_executed_profiles(
    definitions: Sequence[Mapping[str, Any]],
    executions: Mapping[tuple[str, str], Mapping[str, Any]],
    bf16_rows: Mapping[str, Mapping[str, Any]],
    all_w4_rows: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate actual multi-group execution separately from additive estimates."""
    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for definition in definitions:
        grouped[
            (str(definition["policy"]), float(definition["budget_fraction"]))
        ].append(definition)
    summaries: list[dict[str, Any]] = []
    for (policy, budget), items in sorted(grouped.items()):
        profile_rows = [
            executions[(str(item["question_id"]), str(item["profile_signature"]))]
            for item in items
        ]
        question_ids = [str(item["question_id"]) for item in items]
        bf16_nll = float(
            np.mean([float(bf16_rows[item]["answer_nll"]) for item in question_ids])
        )
        w4_nll = float(
            np.mean([float(all_w4_rows[item]["answer_nll"]) for item in question_ids])
        )
        profile_nll = float(np.mean([float(row["answer_nll"]) for row in profile_rows]))
        denominator = w4_nll - bf16_nll
        recovery = (
            (w4_nll - profile_nll) / denominator if denominator > 0 else float("nan")
        )
        executed_gain = w4_nll - profile_nll
        additive_gain = float(
            np.mean([float(item["estimated_gain"]) for item in items])
        )
        summaries.append(
            {
                "policy": policy,
                "budget_fraction": budget,
                "sample_count": len(items),
                "image_count": len({str(item["image_id"]) for item in items}),
                "official_score": float(
                    np.mean([float(row["official_score"]) for row in profile_rows])
                ),
                "mean_answer_nll": profile_nll,
                "mean_js_divergence": float(
                    np.mean([float(row["js_divergence"]) for row in profile_rows])
                ),
                "official_gain_vs_all_w4": float(
                    np.mean(
                        [
                            float(row["official_score"])
                            - float(all_w4_rows[question_id]["official_score"])
                            for row, question_id in zip(
                                profile_rows, question_ids, strict=True
                            )
                        ]
                    )
                ),
                "relative_nll_recovery": recovery,
                "executed_nll_gain": executed_gain,
                "additive_estimated_nll_gain": additive_gain,
                "non_additivity_error": executed_gain - additive_gain,
                "mean_added_bytes": float(
                    np.mean([float(item["added_bytes"]) for item in items])
                ),
                "maximum_budget_bytes": int(items[0]["budget_bytes"]),
            }
        )
    return summaries


def policy_headroom_bootstrap(
    definitions: Sequence[Mapping[str, Any]],
    executions: Mapping[tuple[str, str], Mapping[str, Any]],
    bf16_rows: Mapping[str, Mapping[str, Any]],
    all_w4_rows: Mapping[str, Mapping[str, Any]],
    *,
    budget_fraction: float,
    oracle_policy: str,
    control_policy: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap oracle-control score and recovery headroom by whole images."""
    lookup = {
        (str(item["question_id"]), str(item["policy"])): executions[
            (str(item["question_id"]), str(item["profile_signature"]))
        ]
        for item in definitions
        if float(item["budget_fraction"]) == budget_fraction
        and str(item["policy"]) in {oracle_policy, control_policy}
    }
    image_questions: dict[str, list[str]] = defaultdict(list)
    for item in definitions:
        if (
            float(item["budget_fraction"]) == budget_fraction
            and str(item["policy"]) == oracle_policy
        ):
            image_questions[str(item["image_id"])].append(str(item["question_id"]))
    images = sorted(image_questions)

    def statistics(question_ids: Sequence[str]) -> tuple[float, float]:
        score_difference = float(
            np.mean(
                [
                    float(lookup[(question_id, oracle_policy)]["official_score"])
                    - float(lookup[(question_id, control_policy)]["official_score"])
                    for question_id in question_ids
                ]
            )
        )
        bf16_nll = float(
            np.mean([float(bf16_rows[item]["answer_nll"]) for item in question_ids])
        )
        w4_nll = float(
            np.mean([float(all_w4_rows[item]["answer_nll"]) for item in question_ids])
        )
        denominator = w4_nll - bf16_nll
        if denominator <= 0:
            return score_difference, float("nan")
        oracle_nll = float(
            np.mean(
                [
                    float(lookup[(item, oracle_policy)]["answer_nll"])
                    for item in question_ids
                ]
            )
        )
        control_nll = float(
            np.mean(
                [
                    float(lookup[(item, control_policy)]["answer_nll"])
                    for item in question_ids
                ]
            )
        )
        return score_difference, (control_nll - oracle_nll) / denominator

    all_questions = [
        question_id for image_id in images for question_id in image_questions[image_id]
    ]
    observed_score, observed_recovery = statistics(all_questions)
    generator = random.Random(seed)
    score_values: list[float] = []
    recovery_values: list[float] = []
    for _ in range(replicates):
        sampled_images = generator.choices(images, k=len(images))
        sampled_questions = [
            question_id
            for image_id in sampled_images
            for question_id in image_questions[image_id]
        ]
        score, recovery = statistics(sampled_questions)
        score_values.append(score)
        if np.isfinite(recovery):
            recovery_values.append(recovery)
    score_interval = np.quantile(score_values, [0.025, 0.975]).tolist()
    recovery_interval = np.quantile(recovery_values, [0.025, 0.975]).tolist()
    return {
        "budget_fraction": budget_fraction,
        "control_policy": control_policy,
        "sample_count": len(all_questions),
        "image_count": len(images),
        "official_score_headroom": observed_score,
        "official_score_headroom_lower_95": float(score_interval[0]),
        "official_score_headroom_upper_95": float(score_interval[1]),
        "relative_nll_recovery_headroom": observed_recovery,
        "relative_nll_recovery_headroom_lower_95": float(recovery_interval[0]),
        "relative_nll_recovery_headroom_upper_95": float(recovery_interval[1]),
        "bootstrap_replicates": replicates,
    }
