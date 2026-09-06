"""Deterministic GQA sampling and immutable C01-S1 manifest construction."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

from ..ciq_s0.data_manifest import (
    SampleRecord,
    materialize_s0_records,
    randomized_image_group_order,
)
from ..ciq_s0.inference import load_rgb_image, prepare_prompt_inputs
from ..ciq_s0.io_utils import sha256_json, write_jsonl


QUERY_FAMILIES = (
    "global_perception",
    "fine_grained_recognition",
    "spatial_relation",
    "reasoning",
)
_REASONING_OPERATIONS = frozenset({"and", "or", "same", "different", "common"})


@dataclass(frozen=True)
class S1SampleRecord:
    """One pre-inference C01-S1 image-question record."""

    image_id: str
    question_id: str
    query_text: str
    answer: str
    query_family: str
    image_path: str
    image_file_sha256: str
    image_width: int
    image_height: int
    functional_metadata: dict[str, Any]
    question_token_length: int = -1
    answer_token_length: int = -1
    visual_token_count: int = -1
    preprocessing_sha256: str = ""
    prompt_template_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record."""
        return asdict(self)


def _first_present(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    raise KeyError(f"none of the required fields are present: {names}")


def _type_value(row: Mapping[str, Any], name: str) -> str:
    """Read official nested GQA types before any legacy flat alias."""
    nested = row.get("types")
    if isinstance(nested, Mapping) and nested.get(name) is not None:
        return str(nested[name]).strip().lower()
    for alias in (f"type_{name}", name):
        value = row.get(alias)
        if isinstance(value, (str, int, float, bool)):
            return str(value).strip().lower()
    return ""


def _program_operations(row: Mapping[str, Any]) -> tuple[str, ...]:
    program = row.get("semantic")
    if not isinstance(program, Sequence) or isinstance(program, (str, bytes)):
        return ()
    operations: list[str] = []
    for step in program:
        if isinstance(step, Mapping) and step.get("operation") is not None:
            operations.append(str(step["operation"]).strip().lower().split()[0])
    return tuple(operations)


def classify_s1_query_family(row: Mapping[str, Any]) -> str:
    """Apply the frozen metadata/program-only S1 query taxonomy."""
    semantic = _type_value(row, "semantic")
    structural = _type_value(row, "structural")
    operations = set(_program_operations(row))
    if semantic == "global":
        return "global_perception"
    if structural in {"compare", "logical"} or operations & _REASONING_OPERATIONS:
        return "reasoning"
    if semantic in {"rel", "relation"}:
        return "spatial_relation"
    return "fine_grained_recognition"


def _stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def select_s1_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    image_count: int,
    questions_per_image: int,
    families_per_image: int,
    questions_per_cell: int,
    minimum_images_per_family: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select whole GQA image groups before observing any model output."""
    if not 80 <= image_count <= 100:
        raise ValueError("S1 image_count must be in [80, 100]")
    if questions_per_image < 6:
        raise ValueError("S1 requires at least six questions per image")
    if families_per_image < 3:
        raise ValueError("S1 requires at least three query families per image")
    if questions_per_cell < 2:
        raise ValueError("S1 requires at least two questions per retained cell")
    if questions_per_image != families_per_image * questions_per_cell:
        raise ValueError("locked S1 sampler uses exactly two questions in three cells")

    by_image: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for source in rows:
        row = dict(source)
        image_id = str(_first_present(row, ("image_id", "imageId", "source_id")))
        question_id = str(_first_present(row, ("question_id", "questionId", "id")))
        family = classify_s1_query_family(row)
        row["_image_id"] = image_id
        row["_question_id"] = question_id
        row["_query_family"] = family
        by_image[image_id][family].append(row)

    eligible: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for image_id, family_rows in by_image.items():
        retained = {
            family: values
            for family, values in family_rows.items()
            if len(values) >= questions_per_cell
        }
        if len(retained) >= families_per_image:
            eligible[image_id] = retained
    if len(eligible) < image_count:
        raise RuntimeError("GQA split cannot satisfy the preregistered repeated cells")

    ordered_ids = sorted(eligible, key=lambda value: _stable_rank(seed, value))
    selected_ids: list[str] = []
    reserved_family: dict[str, set[str]] = defaultdict(set)
    for family in QUERY_FAMILIES:
        candidates = [
            image_id
            for image_id in ordered_ids
            if family in eligible[image_id]
            and len(reserved_family[image_id]) < families_per_image
        ]
        for image_id in candidates:
            if (
                sum(
                    family in values
                    for values in (reserved_family[item] for item in selected_ids)
                )
                >= minimum_images_per_family
            ):
                break
            if image_id not in selected_ids:
                selected_ids.append(image_id)
            reserved_family[image_id].add(family)
        represented = sum(
            family in reserved_family[image_id] for image_id in selected_ids
        )
        if represented < minimum_images_per_family:
            raise RuntimeError(f"family {family} cannot reach the locked image minimum")
    selected_ids.extend(
        image_id for image_id in ordered_ids if image_id not in selected_ids
    )
    selected_ids = selected_ids[:image_count]

    family_image_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for image_id in selected_ids:
        mandatory = sorted(reserved_family[image_id])
        if len(mandatory) > families_per_image:
            raise RuntimeError("family reservation overfilled an image")
        remaining = sorted(
            (family for family in eligible[image_id] if family not in mandatory),
            key=lambda family: (
                family_image_counts[family],
                _stable_rank(seed, f"{image_id}:{family}"),
            ),
        )
        chosen_families = mandatory + remaining[: families_per_image - len(mandatory)]
        if len(chosen_families) != families_per_image:
            raise RuntimeError(f"image {image_id} has insufficient retained families")
        for family in chosen_families:
            family_image_counts[family] += 1
            ordered_rows = sorted(
                eligible[image_id][family],
                key=lambda row: _stable_rank(seed, str(row["_question_id"])),
            )
            selected.extend(ordered_rows[:questions_per_cell])

    expected_questions = image_count * questions_per_image
    if len(selected) != expected_questions:
        raise AssertionError("S1 selection did not produce the locked sample count")
    if any(
        family_image_counts[family] < minimum_images_per_family
        for family in QUERY_FAMILIES
    ):
        raise AssertionError("S1 family image minimum was lost during cell allocation")
    return sorted(
        selected, key=lambda row: (str(row["_image_id"]), str(row["_question_id"]))
    )


def _functional_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "types": row.get("types"),
        "program": row.get("semantic"),
        "semantic_string": row.get("semanticStr"),
        "groups": row.get("groups"),
        "is_balanced": row.get("isBalanced"),
    }


def materialize_s1_records(
    selected_rows: Sequence[Mapping[str, Any]],
    image_rows: Iterable[Mapping[str, Any]],
    *,
    repository_root: Path,
    image_directory: Path,
) -> list[S1SampleRecord]:
    """Reuse S0 image materialization and attach full frozen GQA metadata."""
    base_records = materialize_s0_records(
        selected_rows,
        image_rows,
        repository_root=repository_root,
        image_directory=image_directory,
    )
    rows_by_question = {str(row["_question_id"]): row for row in selected_rows}
    records: list[S1SampleRecord] = []
    for base in base_records:
        row = rows_by_question[base.question_id]
        records.append(
            S1SampleRecord(
                image_id=base.image_id,
                question_id=base.question_id,
                query_text=base.query_text,
                answer=base.answer,
                query_family=base.query_family,
                image_path=base.image_path,
                image_file_sha256=base.image_file_sha256,
                image_width=base.image_width,
                image_height=base.image_height,
                functional_metadata=_functional_metadata(row),
            )
        )
    return records


def finalize_s1_metadata(
    records: Sequence[S1SampleRecord],
    *,
    processor: Any,
    instruction: str,
    preprocessing_metadata: Mapping[str, Any],
    repository_root: Path,
) -> list[S1SampleRecord]:
    """Attach token, visual-token, prompt, and preprocessing hashes."""
    preprocessing_hash = sha256_json(dict(preprocessing_metadata))
    prompt_hash = sha256_json(
        {"chat_template": processor.chat_template, "instruction": instruction}
    )
    visual_counts: dict[str, int] = {}
    for record in records:
        if record.image_id in visual_counts:
            continue
        prompt_inputs = prepare_prompt_inputs(
            processor,
            image=load_rgb_image(repository_root / record.image_path),
            question=record.query_text,
            instruction=instruction,
            device=cast(Any, "cpu"),
        )
        image_token_id = processor.tokenizer.convert_tokens_to_ids(
            processor.image_token
        )
        visual_counts[record.image_id] = int(
            (prompt_inputs["input_ids"] == image_token_id).sum().item()
        )
        if visual_counts[record.image_id] <= 0:
            raise RuntimeError("processor produced no visual token placeholders")

    finalized: list[S1SampleRecord] = []
    for record in records:
        question_ids = processor.tokenizer(record.query_text, add_special_tokens=False)[
            "input_ids"
        ]
        answer_ids = processor.tokenizer(record.answer, add_special_tokens=False)[
            "input_ids"
        ]
        finalized.append(
            replace(
                record,
                question_token_length=len(question_ids),
                answer_token_length=len(answer_ids),
                visual_token_count=visual_counts[record.image_id],
                preprocessing_sha256=preprocessing_hash,
                prompt_template_sha256=prompt_hash,
            )
        )
    return finalized


def freeze_s1_manifest(path: Path, records: Sequence[S1SampleRecord]) -> str:
    """Write an immutable manifest plus byte-level SHA-256 sidecar."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen S1 manifest: {path}")
    write_jsonl(path, (record.to_dict() for record in records))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def whole_image_execution_order(
    records: Sequence[S1SampleRecord], *, seed: int
) -> list[S1SampleRecord]:
    """Reuse the S0-tested whole-image shuffler without splitting image groups."""
    shuffled = randomized_image_group_order(
        cast(Sequence[SampleRecord], records), seed=seed
    )
    return cast(list[S1SampleRecord], shuffled)


def validate_sampling_constraints(
    records: Sequence[S1SampleRecord], *, minimum_images_per_family: int
) -> dict[str, Any]:
    """Fail closed if the frozen selection violates any S1 sampling invariant."""
    by_image: dict[str, list[S1SampleRecord]] = defaultdict(list)
    cells: Counter[tuple[str, str]] = Counter()
    for record in records:
        by_image[record.image_id].append(record)
        cells[(record.image_id, record.query_family)] += 1
    if any(len(group) < 6 for group in by_image.values()):
        raise AssertionError("an S1 image has fewer than six questions")
    if any(
        len({record.query_family for record in group}) < 3
        for group in by_image.values()
    ):
        raise AssertionError("an S1 image has fewer than three families")
    if any(count < 2 for count in cells.values()):
        raise AssertionError(
            "an S1 retained image-family cell has fewer than two questions"
        )
    family_images = {
        family: len(
            {record.image_id for record in records if record.query_family == family}
        )
        for family in QUERY_FAMILIES
    }
    if any(value < minimum_images_per_family for value in family_images.values()):
        raise AssertionError("an S1 family has insufficient image support")
    return {
        "image_count": len(by_image),
        "question_count": len(records),
        "repeated_image_family_cells": len(cells),
        "family_question_counts": dict(
            Counter(record.query_family for record in records)
        ),
        "family_image_counts": family_images,
    }
