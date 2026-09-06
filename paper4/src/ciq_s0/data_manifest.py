"""Deterministic GQA S0 sampling and frozen sample-manifest construction."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from .io_utils import sha256_json, write_jsonl
from .vision_cache import image_sha256

QUERY_FAMILIES = (
    "global_perception",
    "fine_grained_recognition",
    "spatial_relation",
    "reasoning",
)


@dataclass(frozen=True)
class SampleRecord:
    """One frozen C01-S0 image-question record."""

    image_id: str
    question_id: str
    query_text: str
    answer: str
    query_family: str
    image_path: str
    image_file_sha256: str
    image_width: int
    image_height: int
    question_token_length: int = -1
    answer_token_length: int = -1
    preprocessing_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record."""
        return asdict(self)


def _first_present(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    raise KeyError(f"none of the required fields are present: {names}")


def _metadata_value(row: Mapping[str, Any], names: Sequence[str]) -> str:
    types = row.get("types")
    if isinstance(types, Mapping):
        for name in names:
            short_name = name.removeprefix("type_")
            value = types.get(short_name)
            if value is not None:
                return str(value).lower()
    for name in names:
        value = row.get(name)
        if value is not None and isinstance(value, (str, int, float, bool)):
            return str(value).lower()
    return ""


def classify_query_family(row: Mapping[str, Any]) -> str:
    """Map frozen GQA metadata/text rules to the four S0 query families."""
    semantic = _metadata_value(row, ("type_semantic", "semantic"))
    structural = _metadata_value(row, ("type_structural", "structural"))
    question = str(_first_present(row, ("question", "query", "text"))).lower()

    if semantic == "global":
        return "global_perception"
    if semantic in {"rel", "relation"} or any(
        phrase in question
        for phrase in (
            "left of",
            "right of",
            "behind",
            "in front of",
            "above",
            "below",
            "next to",
            "where is",
        )
    ):
        return "spatial_relation"
    if structural in {"compare", "logical"} or any(
        phrase in question
        for phrase in ("same", "either", "both", "greater", "less than", "or not")
    ):
        return "reasoning"
    return "fine_grained_recognition"


def _stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def select_s0_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    image_count: int,
    question_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select 8--16 same-image groups and 30--60 questions before inference."""
    if not 8 <= image_count <= 16:
        raise ValueError("S0 image_count must be in [8, 16]")
    if not 30 <= question_count <= 60:
        raise ValueError("S0 question_count must be in [30, 60]")
    if question_count < image_count * 3:
        raise ValueError("S0 needs at least three questions per selected image")

    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        image_id = str(_first_present(row, ("image_id", "imageId", "source_id")))
        question_id = str(_first_present(row, ("question_id", "questionId", "id")))
        row["_image_id"] = image_id
        row["_question_id"] = question_id
        row["_query_family"] = classify_query_family(row)
        by_image[image_id].append(row)

    candidates: list[tuple[str, list[dict[str, Any]]]] = []
    for image_id, group in by_image.items():
        families = {str(row["_query_family"]) for row in group}
        if len(group) >= 4 and len(families) >= 2:
            candidates.append((image_id, group))
    candidates.sort(key=lambda item: _stable_rank(seed, item[0]))
    if len(candidates) < image_count:
        raise RuntimeError("GQA split has too few replicated multi-family image groups")

    selected_groups = candidates[:image_count]
    per_image_base = question_count // image_count
    extra = question_count % image_count
    selected: list[dict[str, Any]] = []
    for image_position, (image_id, group) in enumerate(selected_groups):
        target = per_image_base + int(image_position < extra)
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_family[str(row["_query_family"])].append(row)
        for family_rows in by_family.values():
            family_rows.sort(
                key=lambda row: _stable_rank(seed, str(row["_question_id"]))
            )
        chosen: list[dict[str, Any]] = []
        for family in QUERY_FAMILIES:
            if by_family[family] and len(chosen) < target:
                chosen.append(by_family[family].pop(0))
        remaining = [row for family_rows in by_family.values() for row in family_rows]
        remaining.sort(key=lambda row: _stable_rank(seed, str(row["_question_id"])))
        chosen.extend(remaining[: target - len(chosen)])
        if len(chosen) != target:
            raise RuntimeError(f"image {image_id} cannot supply {target} S0 questions")
        selected.extend(chosen)

    covered = {str(row["_query_family"]) for row in selected}
    if set(QUERY_FAMILIES) - covered:
        raise RuntimeError(
            f"S0 sample misses required query families: {set(QUERY_FAMILIES) - covered}"
        )
    return sorted(
        selected, key=lambda row: (str(row["_image_id"]), str(row["_question_id"]))
    )


def randomized_image_group_order(
    records: Sequence[SampleRecord], *, seed: int
) -> list[SampleRecord]:
    """Randomize whole image groups deterministically without splitting a group."""
    by_image: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        by_image[record.image_id].append(record)
    image_ids = sorted(by_image)
    random.Random(seed).shuffle(image_ids)
    return [record for image_id in image_ids for record in by_image[image_id]]


def _as_pil_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, Mapping):
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        if value.get("path") is not None:
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(BytesIO(value)).convert("RGB")
    raise TypeError(f"unsupported image representation: {type(value)!r}")


def materialize_s0_records(
    selected_rows: Sequence[Mapping[str, Any]],
    image_rows: Iterable[Mapping[str, Any]],
    *,
    repository_root: Path,
    image_directory: Path,
) -> list[SampleRecord]:
    """Save only selected images and construct portable relative-path records."""
    required_ids = {str(row["_image_id"]) for row in selected_rows}
    images: dict[str, Image.Image] = {}
    for row in image_rows:
        image_id = str(_first_present(row, ("image_id", "imageId", "source_id", "id")))
        if image_id in required_ids:
            images[image_id] = _as_pil_image(_first_present(row, ("image", "media")))
    missing = sorted(required_ids - images.keys())
    if missing:
        raise RuntimeError(f"image table is missing selected IDs: {missing}")

    image_directory.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, tuple[Path, str, int, int]] = {}
    for image_id in sorted(images):
        image = images[image_id]
        path = image_directory / f"{image_id}.png"
        image.save(path, format="PNG", optimize=False)
        metadata[image_id] = (
            path,
            image_sha256(path),
            image.width,
            image.height,
        )

    records: list[SampleRecord] = []
    for row in selected_rows:
        image_id = str(row["_image_id"])
        path, file_hash, width, height = metadata[image_id]
        answer = str(_first_present(row, ("answer", "answers")))
        if isinstance(row.get("answers"), Sequence) and not isinstance(
            row.get("answers"), str
        ):
            answers = row["answers"]
            if answers:
                answer = str(answers[0])
        records.append(
            SampleRecord(
                image_id=image_id,
                question_id=str(row["_question_id"]),
                query_text=str(_first_present(row, ("question", "query", "text"))),
                answer=answer,
                query_family=str(row["_query_family"]),
                image_path=str(path.relative_to(repository_root)),
                image_file_sha256=file_hash,
                image_width=width,
                image_height=height,
            )
        )
    return records


def finalize_token_metadata(
    records: Sequence[SampleRecord],
    *,
    tokenizer: Any,
    preprocessing_metadata: Mapping[str, Any],
) -> list[SampleRecord]:
    """Add frozen token lengths and a shared preprocessing hash."""
    preprocessing_hash = sha256_json(dict(preprocessing_metadata))
    finalized: list[SampleRecord] = []
    for record in records:
        question_ids = tokenizer(record.query_text, add_special_tokens=False)[
            "input_ids"
        ]
        answer_ids = tokenizer(record.answer, add_special_tokens=False)["input_ids"]
        finalized.append(
            replace(
                record,
                question_token_length=len(question_ids),
                answer_token_length=len(answer_ids),
                preprocessing_sha256=preprocessing_hash,
            )
        )
    return finalized


def freeze_sample_manifest(path: Path, records: Sequence[SampleRecord]) -> str:
    """Write the selected metadata exactly once and return its byte-level hash."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {path}")
    write_jsonl(path, (record.to_dict() for record in records))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest
