"""Registered C02 effect-sign and integrity aggregation tests."""

from __future__ import annotations

import math

from src.causalquant.s0_analysis import CONDITIONS, calculate_effects


def test_registered_delta_sign_convention() -> None:
    """Delta equals the W4-minus-BF relevant-versus-irrelevant interaction."""
    nll = {
        "bf16_original": 1.0,
        "bf16_relevant_cf": 1.4,
        "bf16_irrelevant_cf": 1.1,
        "w4_original": 1.2,
        "w4_relevant_cf": 1.45,
        "w4_irrelevant_cf": 1.25,
    }
    rows = [
        {
            "answer_nll": nll[condition],
            "condition": condition,
            "expression_id": 1,
            "image_id": 10,
            "semantic_family": "attribute",
            "target": "car",
        }
        for condition in CONDITIONS
    ]
    result = calculate_effects(rows)[0]
    assert math.isclose(result["bf16_relevant_effect"], 0.4)
    assert math.isclose(result["bf16_irrelevant_effect"], 0.1)
    assert math.isclose(result["w4_relevant_effect"], 0.25)
    assert math.isclose(result["w4_irrelevant_effect"], 0.05)
    assert math.isclose(result["delta"], -0.1)
