"""Deterministic, equal-area rectangular occlusions and pixel audits."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PixelBox:
    """Half-open integer pixel rectangle."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        """Return rectangle width in pixels."""
        return self.right - self.left

    @property
    def height(self) -> int:
        """Return rectangle height in pixels."""
        return self.bottom - self.top

    @property
    def area(self) -> int:
        """Return rectangle area in pixels."""
        return self.width * self.height

    def to_list(self) -> list[int]:
        """Return canonical xyxy coordinates."""
        return [self.left, self.top, self.right, self.bottom]


@dataclass(frozen=True)
class OcclusionAudit:
    """Exact audit for one deterministic occlusion."""

    box: list[int]
    box_area: int
    changed_pixel_count: int
    outside_bitwise_identical: bool
    changes_confined_to_box: bool
    fill_rgb: list[int]
    source_pixel_sha256: str
    output_pixel_sha256: str
    repeated_output_pixel_sha256: str
    deterministic: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record."""
        return asdict(self)


def rasterize_equal_shape_box(
    bbox_xywh: list[float], *, image_width: int, image_height: int
) -> PixelBox:
    """Rasterize xywh with rounded anchor and size, preserving full area."""
    if len(bbox_xywh) != 4:
        raise ValueError("bbox must contain x, y, width, height")
    x, y, width, height = (float(value) for value in bbox_xywh)
    pixel_width = max(1, round(width))
    pixel_height = max(1, round(height))
    left = round(x)
    top = round(y)
    box = PixelBox(left, top, left + pixel_width, top + pixel_height)
    if (
        box.left < 0
        or box.top < 0
        or box.right > image_width
        or box.bottom > image_height
    ):
        raise ValueError("rounded bbox does not fit inside the image")
    return box


def intersection_area(first: PixelBox, second: PixelBox) -> int:
    """Return exact integer-pixel intersection area."""
    width = max(0, min(first.right, second.right) - max(first.left, second.left))
    height = max(0, min(first.bottom, second.bottom) - max(first.top, second.top))
    return width * height


def pixel_sha256(pixels: np.ndarray) -> str:
    """Hash array shape, dtype, and exact pixel bytes."""
    contiguous = np.ascontiguousarray(pixels)
    digest = hashlib.sha256()
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _border_ring_mask(
    *, image_width: int, image_height: int, box: PixelBox, ring_width: int
) -> np.ndarray:
    if ring_width <= 0:
        raise ValueError("ring_width must be positive")
    outer = PixelBox(
        max(0, box.left - ring_width),
        max(0, box.top - ring_width),
        min(image_width, box.right + ring_width),
        min(image_height, box.bottom + ring_width),
    )
    mask = np.zeros((image_height, image_width), dtype=bool)
    mask[outer.top : outer.bottom, outer.left : outer.right] = True
    mask[box.top : box.bottom, box.left : box.right] = False
    return mask


def border_ring_median_occlusion(
    source: np.ndarray, box: PixelBox, *, ring_width: int
) -> tuple[np.ndarray, list[int]]:
    """Fill a box with its outside border-ring per-channel uint8 median."""
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("source must be an RGB uint8 array")
    image_height, image_width, _ = source.shape
    ring_mask = _border_ring_mask(
        image_width=image_width,
        image_height=image_height,
        box=box,
        ring_width=ring_width,
    )
    ring_pixels = source[ring_mask]
    if ring_pixels.size == 0:
        raise ValueError("bbox has no exterior border-ring pixels")
    fill = np.median(ring_pixels, axis=0).astype(np.uint8)
    output = source.copy()
    output[box.top : box.bottom, box.left : box.right] = fill
    return output, [int(value) for value in fill]


def audit_occlusion(
    source: np.ndarray,
    box: PixelBox,
    *,
    ring_width: int,
) -> tuple[np.ndarray, OcclusionAudit]:
    """Apply an occlusion twice and verify its exact pixel support and hash."""
    output, fill = border_ring_median_occlusion(source, box, ring_width=ring_width)
    repeated, repeated_fill = border_ring_median_occlusion(
        source, box, ring_width=ring_width
    )
    changed = np.any(source != output, axis=2)
    allowed = np.zeros(changed.shape, dtype=bool)
    allowed[box.top : box.bottom, box.left : box.right] = True
    output_hash = pixel_sha256(output)
    repeated_hash = pixel_sha256(repeated)
    deterministic = output_hash == repeated_hash and fill == repeated_fill
    outside_equal = bool(np.array_equal(source[~allowed], output[~allowed]))
    confined = not bool(np.any(changed & ~allowed))
    audit = OcclusionAudit(
        box=box.to_list(),
        box_area=box.area,
        changed_pixel_count=int(np.count_nonzero(changed)),
        outside_bitwise_identical=outside_equal,
        changes_confined_to_box=confined,
        fill_rgb=fill,
        source_pixel_sha256=pixel_sha256(source),
        output_pixel_sha256=output_hash,
        repeated_output_pixel_sha256=repeated_hash,
        deterministic=deterministic,
    )
    if not outside_equal or not confined or not deterministic:
        raise AssertionError("counterfactual pixel integrity failed")
    return output, audit


def load_rgb_array(path: Path) -> np.ndarray:
    """Load exact decoded RGB pixels into a writable uint8 array."""
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)


def save_rgb_png(path: Path, pixels: np.ndarray) -> None:
    """Save RGB pixels losslessly as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG", optimize=False)
