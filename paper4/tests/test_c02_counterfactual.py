"""Exact pixel-support tests for C02 counterfactual construction."""

from __future__ import annotations

import numpy as np

from src.causalquant.counterfactual import (
    PixelBox,
    audit_occlusion,
    intersection_area,
    rasterize_equal_shape_box,
)


def test_equal_shape_rasterization_and_non_overlap() -> None:
    """Matched float boxes retain exact integer area and do not overlap."""
    relevant = rasterize_equal_shape_box(
        [1.2, 2.4, 4.7, 3.2], image_width=16, image_height=12
    )
    irrelevant = rasterize_equal_shape_box(
        [9.0, 7.0, 4.7, 3.2], image_width=16, image_height=12
    )
    assert relevant.area == irrelevant.area == 15
    assert intersection_area(relevant, irrelevant) == 0


def test_occlusion_is_confined_and_deterministic() -> None:
    """The median fill changes no pixel outside the requested rectangle."""
    source = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    box = PixelBox(5, 4, 10, 9)
    output, audit = audit_occlusion(source, box, ring_width=2)
    assert audit.outside_bitwise_identical
    assert audit.changes_confined_to_box
    assert audit.deterministic
    assert audit.output_pixel_sha256 == audit.repeated_output_pixel_sha256
    assert np.array_equal(source[:4], output[:4])
