"""Deterministic RefCOCOg smoke selection and frozen manifest creation."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..ciq_s0.io_utils import canonical_json_bytes, sha256_file, write_jsonl
from .counterfactual import (
    intersection_area,
    rasterize_equal_shape_box,
)


COCO_CATEGORY_NAMES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

SEMANTIC_FAMILIES = ("attribute", "spatial", "relational", "recognition")


def assigned_semantic_family(categories: list[str]) -> str:
    """Assign one primary family with a frozen, mutually exclusive priority."""
    category_set = set(categories)
    for family in ("recognition", "spatial", "relational", "attribute"):
        if family in category_set:
            return family
    raise ValueError(f"record has no supported semantic category: {categories}")


def load_usable_records(path: Path) -> list[dict[str, Any]]:
    """Load the already-validated RefCOCOg usable-record artifact."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_ref_metadata(path: Path, ref_ids: set[int]) -> dict[int, dict[str, Any]]:
    """Read category and image-name metadata for usable reference IDs only."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "manifest freeze requires the existing pyarrow reader"
        ) from error
    table = parquet.read_table(
        path,
        columns=["ref_id", "ann_id", "category_id", "raw_image_info"],
        filters=[("ref_id", "in", sorted(ref_ids))],
    )
    selected = table.to_pylist()
    metadata: dict[int, dict[str, Any]] = {}
    for row in selected:
        ref_id = int(row["ref_id"])
        if ref_id not in ref_ids:
            continue
        image_info = json.loads(str(row["raw_image_info"]))
        metadata[ref_id] = {
            "ann_id": int(row["ann_id"]),
            "category_id": int(row["category_id"]),
            "file_name": str(image_info["file_name"]),
        }
    if set(metadata) != ref_ids:
        missing = sorted(ref_ids - set(metadata))
        raise RuntimeError(f"missing source metadata for refs: {missing[:5]}")
    return metadata


def _candidate_record(
    source: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any] | None:
    width = int(round(float(source["width"])))
    height = int(round(float(source["height"])))
    relevant_box = rasterize_equal_shape_box(
        source["bbox"], image_width=width, image_height=height
    )
    irrelevant_box = rasterize_equal_shape_box(
        source["control_bbox"], image_width=width, image_height=height
    )
    if relevant_box.area != irrelevant_box.area:
        raise AssertionError("matched source boxes changed area during rasterization")
    if intersection_area(relevant_box, irrelevant_box) != 0:
        return None
    category_id = int(metadata["category_id"])
    category_name = COCO_CATEGORY_NAMES[category_id]
    pattern = rf"(?<!\w){re.escape(category_name)}(?!\w)"
    if re.search(pattern, str(source["text"]).lower()) is None:
        return None
    return {
        **source,
        "width": width,
        "height": height,
        "category_id": category_id,
        "target": category_name,
        "semantic_family": assigned_semantic_family(source["categories"]),
        "file_name": str(metadata["file_name"]),
        "relevant_box_xyxy": relevant_box.to_list(),
        "irrelevant_box_xyxy": irrelevant_box.to_list(),
        "mask_area_pixels": relevant_box.area,
    }


def select_smoke_records(
    records: list[dict[str, Any]],
    metadata_by_ref: dict[int, dict[str, Any]],
    *,
    seed: int,
    image_count: int,
    expressions_per_image: int,
    expressions_per_family: int,
) -> list[dict[str, Any]]:
    """Select balanced same-image pairs by fixed geometry and hash ordering."""
    if expressions_per_image != 2:
        raise ValueError("C02-S0 selection is locked to two expressions per image")
    expected_expressions = image_count * expressions_per_image
    if expressions_per_family * len(SEMANTIC_FAMILIES) != expected_expressions:
        raise ValueError("family quota does not equal the requested sample size")
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for source in records:
        candidate = _candidate_record(source, metadata_by_ref[int(source["ref_id"])])
        if candidate is not None:
            by_image[int(candidate["image_id"])].append(candidate)

    pairs: list[tuple[bool, str, int, dict[str, Any], dict[str, Any]]] = []
    for image_id, image_records in by_image.items():
        for first, second in itertools.combinations(image_records, 2):
            if first["semantic_family"] == second["semantic_family"]:
                continue
            distinct_objects = int(first["ann_id"]) != int(second["ann_id"])
            tie_hash = hashlib.sha256(
                (
                    f"{seed}:{image_id}:{first['expression_id']}:"
                    f"{second['expression_id']}"
                ).encode("ascii")
            ).hexdigest()
            pairs.append((not distinct_objects, tie_hash, image_id, first, second))

    selected: list[dict[str, Any]] = []
    selected_images: set[int] = set()
    counts: Counter[str] = Counter()
    for _ in range(image_count):
        ranked: list[
            tuple[int, bool, int, str, int, dict[str, Any], dict[str, Any]]
        ] = []
        for non_distinct, tie_hash, image_id, first, second in pairs:
            if image_id in selected_images:
                continue
            proposed = counts.copy()
            proposed[str(first["semantic_family"])] += 1
            proposed[str(second["semantic_family"])] += 1
            if any(
                proposed[family] > expressions_per_family
                for family in SEMANTIC_FAMILIES
            ):
                continue
            filled = sum(
                max(0, expressions_per_family - counts[family])
                - max(0, expressions_per_family - proposed[family])
                for family in SEMANTIC_FAMILIES
            )
            scarcity = sum(
                expressions_per_family - counts[str(item["semantic_family"])]
                for item in (first, second)
            )
            ranked.append(
                (-filled, non_distinct, -scarcity, tie_hash, image_id, first, second)
            )
        if not ranked:
            raise RuntimeError(
                "deterministic selection could not satisfy family quotas"
            )
        _, _, _, _, image_id, first, second = min(ranked)
        selected_images.add(image_id)
        for item in (first, second):
            counts[str(item["semantic_family"])] += 1
            selected.append(item)

    if counts != Counter(
        {family: expressions_per_family for family in SEMANTIC_FAMILIES}
    ):
        raise AssertionError(f"semantic balance failed: {counts}")
    if len(selected) != expected_expressions or len(selected_images) != image_count:
        raise AssertionError("selected sample size does not match the locked request")
    return selected


def build_manifest_records(
    selected: list[dict[str, Any]],
    *,
    prompt_template: str,
    image_directory: str,
    counterfactual_directory: str,
    selection_seed: int,
) -> list[dict[str, Any]]:
    """Create canonical manifest rows without consulting any model output."""
    manifest: list[dict[str, Any]] = []
    for item in selected:
        expression_id = int(item["expression_id"])
        file_name = str(item["file_name"])
        row = {
            "ann_id": int(item["ann_id"]),
            "category_id": int(item["category_id"]),
            "expression_id": expression_id,
            "height": int(item["height"]),
            "image_id": int(item["image_id"]),
            "irrelevant_box_xyxy": item["irrelevant_box_xyxy"],
            "irrelevant_cf_path": (
                f"{counterfactual_directory}/{expression_id}_irrelevant.png"
            ),
            "mask_area_pixels": int(item["mask_area_pixels"]),
            "original_image_path": f"{image_directory}/{file_name}",
            "prompt": prompt_template.replace("<expression>", str(item["text"])),
            "ref_id": int(item["ref_id"]),
            "relevant_box_xyxy": item["relevant_box_xyxy"],
            "relevant_cf_path": f"{counterfactual_directory}/{expression_id}_relevant.png",
            "selection_seed": selection_seed,
            "semantic_categories": list(item["categories"]),
            "semantic_family": str(item["semantic_family"]),
            "source_record_sha256": hashlib.sha256(
                canonical_json_bytes(
                    {
                        key: item[key]
                        for key in (
                            "ann_id",
                            "bbox",
                            "categories",
                            "control_bbox",
                            "expression_id",
                            "image_id",
                            "ref_id",
                            "text",
                        )
                    }
                )
            ).hexdigest(),
            "target": str(item["target"]),
            "text": str(item["text"]),
            "width": int(item["width"]),
        }
        manifest.append(row)
    return manifest


def freeze_manifest(path: Path, records: list[dict[str, Any]]) -> str:
    """Write a manifest once and reject any attempted content change."""
    if path.exists():
        existing = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if existing != records:
            raise RuntimeError("refusing to modify the frozen C02-S0 manifest")
    else:
        write_jsonl(path, records)
    digest = sha256_file(path)
    hash_path = path.with_suffix(path.suffix + ".sha256")
    expected_line = f"{digest}  {path.name}\n"
    if hash_path.exists() and hash_path.read_text(encoding="ascii") != expected_line:
        raise RuntimeError("frozen C02-S0 manifest hash sidecar disagrees")
    if not hash_path.exists():
        hash_path.write_text(expected_line, encoding="ascii")
    return digest
