"""Phased C02-S0 counterfactual pipeline integrity runner."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from ..ciq_s0.evaluation import normalized_exact_score
from ..ciq_s0.inference import (
    configure_determinism,
    extract_visual_features,
    prepare_text_prompt_inputs,
    tensor_value_sha256,
)
from ..ciq_s0.io_utils import (
    read_json,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)
from ..ciq_s0.module_manifest import all_quantizable_tensor_names, build_module_manifest
from ..ciq_s0.quantization import QuantizerSpec, quantize_parameters_in_place
from ..ciq_s0.vision_cache import VisionCacheKey, VisionFeatureCache
from ..ciq_s1.distributions import FixedSupportDistribution
from ..ciq_s1.inference import run_s1_prepared_case
from .counterfactual import (
    PixelBox,
    audit_occlusion,
    intersection_area,
    load_rgb_array,
    pixel_sha256,
    save_rgb_png,
)
from .s0_analysis import load_and_analyze
from .s0_manifest import (
    build_manifest_records,
    freeze_manifest,
    load_ref_metadata,
    load_usable_records,
    select_smoke_records,
)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def collect_environment() -> dict[str, Any]:
    """Collect non-sensitive execution environment metadata."""
    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "NONE",
        "numpy": _package_version("numpy"),
        "pillow": _package_version("Pillow"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": _package_version("transformers"),
    }


def _paths(config: dict[str, Any], repository_root: Path) -> dict[str, Path]:
    return {key: repository_root / str(value) for key, value in config["paths"].items()}


def _load_manifest(
    path: Path, expected_hash: str | None = None
) -> list[dict[str, Any]]:
    if expected_hash is not None and sha256_file(path) != expected_hash:
        raise RuntimeError("C02-S0 manifest hash changed after freeze")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def freeze_s0_manifest(config: dict[str, Any], repository_root: Path) -> dict[str, Any]:
    """Select samples deterministically and freeze the pre-inference manifest."""
    paths = _paths(config, repository_root)
    usable = load_usable_records(paths["usable_records"])
    metadata = load_ref_metadata(
        paths["source_parquet"], {int(row["ref_id"]) for row in usable}
    )
    selection = config["selection"]
    selected = select_smoke_records(
        usable,
        metadata,
        seed=int(config["seed"]),
        image_count=int(selection["image_count"]),
        expressions_per_image=int(selection["expressions_per_image"]),
        expressions_per_family=int(selection["expressions_per_family"]),
    )
    manifest = build_manifest_records(
        selected,
        prompt_template=str(config["task"]["prompt_template"]),
        image_directory=str(config["paths"]["image_directory"]),
        counterfactual_directory=str(config["paths"]["counterfactual_directory"]),
        selection_seed=int(config["seed"]),
    )
    digest = freeze_manifest(paths["manifest"], manifest)
    summary = {
        "expression_count": len(manifest),
        "family_counts": dict(
            sorted(Counter(str(row["semantic_family"]) for row in manifest).items())
        ),
        "image_count": len({int(row["image_id"]) for row in manifest}),
        "manifest_sha256": digest,
        "selection_uses_model_outputs": False,
    }
    write_json(paths["processed_output"] / "manifest_summary.json", summary)
    return summary


def _file_sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _download_images(
    manifest: list[dict[str, Any]], config: dict[str, Any], repository_root: Path
) -> list[dict[str, Any]]:
    base_url = str(config["dataset"]["coco_image_base_url"]).rstrip("/")
    by_path = {str(row["original_image_path"]): row for row in manifest}
    inventory: list[dict[str, Any]] = []
    for relative_path, row in sorted(by_path.items()):
        destination = repository_root / relative_path
        url = f"{base_url}/{destination.name}"
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            if not payload:
                raise RuntimeError(f"empty COCO image response for {destination.name}")
            destination.write_bytes(payload)
        payload = destination.read_bytes()
        with Image.open(destination) as image:
            rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        expected_shape = (int(row["height"]), int(row["width"]), 3)
        if rgb.shape != expected_shape:
            raise RuntimeError(
                f"image dimensions disagree for {destination.name}: "
                f"{rgb.shape} != {expected_shape}"
            )
        inventory.append(
            {
                "decoded_pixel_sha256": pixel_sha256(rgb),
                "file_name": destination.name,
                "file_sha256": _file_sha256_bytes(payload),
                "height": rgb.shape[0],
                "path": relative_path,
                "source_url": url,
                "width": rgb.shape[1],
            }
        )
    return inventory


def _box(values: list[int]) -> PixelBox:
    return PixelBox(*(int(value) for value in values))


def _create_counterfactual_montage(
    manifest: list[dict[str, Any]], repository_root: Path, output_path: Path
) -> None:
    rows: list[Image.Image] = []
    cell_size = (260, 210)
    for record in manifest:
        cells: list[Image.Image] = []
        for label, key in (
            ("original", "original_image_path"),
            ("relevant", "relevant_cf_path"),
            ("irrelevant", "irrelevant_cf_path"),
        ):
            with Image.open(repository_root / str(record[key])) as source:
                image = source.convert("RGB")
                image.thumbnail((240, 170))
            cell = Image.new("RGB", cell_size, "white")
            cell.paste(image, ((cell_size[0] - image.width) // 2, 25))
            draw = ImageDraw.Draw(cell)
            draw.text((5, 5), f"{record['expression_id']} {label}", fill="black")
            cells.append(cell)
        row_image = Image.new("RGB", (cell_size[0] * 3, cell_size[1]), "white")
        for index, cell in enumerate(cells):
            row_image.paste(cell, (index * cell_size[0], 0))
        rows.append(row_image)
    montage = Image.new("RGB", (cell_size[0] * 3, cell_size[1] * len(rows)), "white")
    for index, row_image in enumerate(rows):
        montage.paste(row_image, (0, index * cell_size[1]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output_path, format="PNG", optimize=False)


def prepare_counterfactuals(
    config: dict[str, Any], repository_root: Path
) -> dict[str, Any]:
    """Acquire only manifest images, generate both CFs, and audit exact pixels."""
    paths = _paths(config, repository_root)
    digest = sha256_file(paths["manifest"])
    manifest = _load_manifest(paths["manifest"], digest)
    inventory = _download_images(manifest, config, repository_root)
    write_json(paths["processed_output"] / "image_inventory.json", inventory)
    audits: list[dict[str, Any]] = []
    ring_width = int(config["counterfactual"]["border_ring_width"])
    for record in manifest:
        source = load_rgb_array(repository_root / str(record["original_image_path"]))
        relevant_box = _box(record["relevant_box_xyxy"])
        irrelevant_box = _box(record["irrelevant_box_xyxy"])
        if relevant_box.area != irrelevant_box.area:
            raise AssertionError("relevant and irrelevant raster areas differ")
        if intersection_area(relevant_box, irrelevant_box) != 0:
            raise AssertionError("relevant and irrelevant raster regions overlap")
        relevant, relevant_audit = audit_occlusion(
            source, relevant_box, ring_width=ring_width
        )
        irrelevant, irrelevant_audit = audit_occlusion(
            source, irrelevant_box, ring_width=ring_width
        )
        save_rgb_png(repository_root / str(record["relevant_cf_path"]), relevant)
        save_rgb_png(repository_root / str(record["irrelevant_cf_path"]), irrelevant)
        audits.append(
            {
                "area_match": relevant_box.area == irrelevant_box.area,
                "expression_id": int(record["expression_id"]),
                "image_id": int(record["image_id"]),
                "non_overlap": intersection_area(relevant_box, irrelevant_box) == 0,
                "irrelevant": irrelevant_audit.to_dict(),
                "relevant": relevant_audit.to_dict(),
            }
        )
    write_jsonl(paths["processed_output"] / "pixel_integrity.jsonl", audits)
    summary = {
        "area_match": all(bool(row["area_match"]) for row in audits),
        "deterministic_counterfactuals": all(
            bool(row[condition]["deterministic"])
            for row in audits
            for condition in ("relevant", "irrelevant")
        ),
        "expression_count": len(audits),
        "image_count": len(inventory),
        "irrelevant_pixel_integrity": all(
            bool(row["irrelevant"]["outside_bitwise_identical"])
            and bool(row["irrelevant"]["changes_confined_to_box"])
            for row in audits
        ),
        "manifest_sha256": digest,
        "non_overlap": all(bool(row["non_overlap"]) for row in audits),
        "operator": "per-channel median of 5-pixel exterior border ring",
        "relevant_pixel_integrity": all(
            bool(row["relevant"]["outside_bitwise_identical"])
            and bool(row["relevant"]["changes_confined_to_box"])
            for row in audits
        ),
    }
    write_json(paths["processed_output"] / "pixel_integrity_summary.json", summary)
    _create_counterfactual_montage(
        manifest,
        repository_root,
        paths["figure_output"] / "counterfactual_montage.png",
    )
    return summary


def _load_model_and_processor(
    config: dict[str, Any], repository_root: Path
) -> tuple[torch.nn.Module, Any]:
    model_config = config["model"]
    snapshot = repository_root / str(model_config["local_snapshot"])
    if not snapshot.is_dir():
        raise RuntimeError("locked Qwen2-VL snapshot is missing")
    processor = AutoProcessor.from_pretrained(
        snapshot,
        cache_dir=repository_root / str(config["paths"]["model_cache"]),
        min_pixels=int(model_config["min_pixels"]),
        max_pixels=int(model_config["max_pixels"]),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=bool(model_config["use_fast_image_processor"]),
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        snapshot,
        cache_dir=repository_root / str(config["paths"]["model_cache"]),
        torch_dtype=torch.bfloat16,
        attn_implementation=str(model_config["attention_implementation"]),
        local_files_only=True,
        trust_remote_code=False,
    ).to("cuda:0")
    model.eval()
    return model, processor


def _generation_settings(config: dict[str, Any], processor: Any) -> dict[str, Any]:
    settings = dict(config["inference"]["generation"])
    settings["pad_token_id"] = processor.tokenizer.pad_token_id
    settings["eos_token_id"] = processor.tokenizer.eos_token_id
    return settings


def _reference(value: dict[str, Any]) -> FixedSupportDistribution:
    return FixedSupportDistribution(
        token_ids=[[int(token) for token in row] for row in value["token_ids"]],
        probabilities=[
            [float(probability) for probability in row]
            for row in value["probabilities"]
        ],
    )


def _run_case(
    model: torch.nn.Module,
    processor: Any,
    *,
    record: dict[str, Any],
    image_path: Path,
    prompt: str,
    condition: str,
    config: dict[str, Any],
    reference: FixedSupportDistribution | None,
) -> dict[str, Any]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        prompt_inputs = prepare_text_prompt_inputs(
            processor,
            image=image,
            prompt=prompt,
            device=torch.device("cuda:0"),
        )
    result = run_s1_prepared_case(
        model,
        processor,
        prompt_inputs=prompt_inputs,
        answer=str(record["target"]),
        generation_settings=_generation_settings(config, processor),
        distribution_top_k=int(config["distribution"]["top_k"]),
        bf16_reference=reference,
    )
    payload = {
        "condition": condition,
        "expression_id": int(record["expression_id"]),
        "image_id": int(record["image_id"]),
        "normalized_exact_score": normalized_exact_score(
            result.prediction, str(record["target"])
        ),
        "prompt_sha256": sha256_json(prompt),
        "semantic_family": str(record["semantic_family"]),
        "target": str(record["target"]),
    }
    payload.update(result.to_dict())
    return payload


def _assert_result_repeat(first: dict[str, Any], second: dict[str, Any]) -> None:
    for key in ("prediction", "target_token_sha256"):
        if first[key] != second[key]:
            raise AssertionError(f"deterministic repeat changed {key}")
    for key in ("answer_nll", "js_divergence"):
        if first.get(key) is None and second.get(key) is None:
            continue
        if not math.isclose(
            float(first[key]), float(second[key]), rel_tol=0.0, abs_tol=1e-6
        ):
            raise AssertionError(f"deterministic repeat changed {key}")


def _preflight_stats(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    accuracy = sum(float(row["normalized_exact_score"]) for row in rows) / len(rows)
    empty_rate = sum(not str(row["prediction"]).strip() for row in rows) / len(rows)
    malformed_rate = sum(
        len(str(row["prediction"]).split())
        > int(config["gates"]["bf16_original"]["maximum_prediction_words"])
        for row in rows
    ) / len(rows)
    finite_rate = sum(math.isfinite(float(row["answer_nll"])) for row in rows) / len(
        rows
    )
    gate = config["gates"]["bf16_original"]
    valid = bool(
        accuracy >= float(gate["minimum_accuracy"])
        and empty_rate <= float(gate["maximum_empty_rate"])
        and malformed_rate <= float(gate["maximum_malformed_rate"])
        and finite_rate == 1.0
    )
    return {
        "accuracy": accuracy,
        "empty_rate": empty_rate,
        "finite_nll_rate": finite_rate,
        "malformed_rate": malformed_rate,
        "valid": valid,
    }


def run_bf16_original(config: dict[str, Any], repository_root: Path) -> dict[str, Any]:
    """Run only BF16+Original, applying at most one fixed prompt correction."""
    if not torch.cuda.is_available():
        raise RuntimeError("C02-S0 BF16 preflight requires CUDA")
    paths = _paths(config, repository_root)
    manifest_hash = sha256_file(paths["manifest"])
    manifest = _load_manifest(paths["manifest"], manifest_hash)
    configure_determinism(int(config["seed"]))
    environment = collect_environment()
    if environment["python"].split(".")[:2] != ["3", "12"]:
        raise RuntimeError("AGENTS.md requires Python 3.12")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    model, processor = _load_model_and_processor(config, repository_root)
    prompt_templates = (
        ("PRIMARY", str(config["task"]["prompt_template"])),
        (
            "ONE DETERMINISTIC CORRECTION",
            str(config["task"]["corrected_prompt_template"]),
        ),
    )
    active_rows: list[dict[str, Any]] = []
    active_stats: dict[str, Any] = {}
    active_label = ""
    correction_used = False
    for prompt_index, (label, template) in enumerate(prompt_templates):
        candidate_rows: list[dict[str, Any]] = []
        for record in manifest:
            prompt = template.replace("<expression>", str(record["text"]))
            candidate_rows.append(
                _run_case(
                    model,
                    processor,
                    record=record,
                    image_path=repository_root / str(record["original_image_path"]),
                    prompt=prompt,
                    condition="bf16_original",
                    config=config,
                    reference=None,
                )
            )
        repeat_count = int(config["integrity"]["repeatability_case_count"])
        for record, first in zip(
            manifest[:repeat_count], candidate_rows[:repeat_count], strict=True
        ):
            prompt = template.replace("<expression>", str(record["text"]))
            repeated = _run_case(
                model,
                processor,
                record=record,
                image_path=repository_root / str(record["original_image_path"]),
                prompt=prompt,
                condition="bf16_original",
                config=config,
                reference=None,
            )
            _assert_result_repeat(first, repeated)
            if first["reference_distribution"] != repeated["reference_distribution"]:
                raise AssertionError(
                    "BF16 fixed-support distribution is not repeatable"
                )
        candidate_stats = _preflight_stats(candidate_rows, config)
        active_rows = candidate_rows
        active_stats = candidate_stats
        active_label = label
        correction_used = prompt_index == 1
        if candidate_stats["valid"]:
            break
    torch.cuda.synchronize()
    gpu_seconds = time.perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated())
    write_jsonl(paths["raw_output"] / "bf16_original.jsonl", active_rows)
    outcome = "PASS" if active_stats["valid"] else "TASK FORMULATION INVALID"
    summary = {
        **active_stats,
        "correction_used": correction_used,
        "environment": environment,
        "gpu_hours": gpu_seconds / 3600.0,
        "manifest_sha256": manifest_hash,
        "outcome": outcome,
        "peak_memory_bytes": peak_memory,
        "repeatability_pass": True,
        "task_formulation": active_label,
    }
    write_json(paths["processed_output"] / "bf16_preflight_summary.json", summary)
    del model
    torch.cuda.empty_cache()
    return summary


def _active_prompt(
    record: dict[str, Any], config: dict[str, Any], preflight: dict[str, Any]
) -> str:
    key = (
        "corrected_prompt_template"
        if preflight["correction_used"]
        else "prompt_template"
    )
    return str(config["task"][key]).replace("<expression>", str(record["text"]))


def _cache_isolation_precheck(
    model: torch.nn.Module,
    processor: Any,
    record: dict[str, Any],
    prompt: str,
    repository_root: Path,
) -> tuple[dict[str, Any], VisionFeatureCache, VisionCacheKey]:
    prepared: dict[str, dict[str, torch.Tensor]] = {}
    file_hashes: dict[str, str] = {}
    for condition, key in (
        ("original", "original_image_path"),
        ("relevant_cf", "relevant_cf_path"),
        ("irrelevant_cf", "irrelevant_cf_path"),
    ):
        path = repository_root / str(record[key])
        file_hashes[condition] = sha256_file(path)
        with Image.open(path) as source:
            prepared[condition] = prepare_text_prompt_inputs(
                processor,
                image=source.convert("RGB"),
                prompt=prompt,
                device=torch.device("cuda:0"),
            )
    features = extract_visual_features(model, prepared["original"])
    repeated = extract_visual_features(model, prepared["original"])
    if not torch.equal(features, repeated):
        raise AssertionError("BF16 original visual features are not repeatable")
    preprocessing_hashes = {
        condition: sha256_json(
            {
                "grid": tensor_value_sha256(values["image_grid_thw"]),
                "pixels": tensor_value_sha256(values["pixel_values"]),
            }
        )
        for condition, values in prepared.items()
    }
    if len(set(preprocessing_hashes.values())) != 3:
        raise AssertionError("counterfactual preprocessing hashes unexpectedly collide")
    cache = VisionFeatureCache()
    original_key = VisionCacheKey(
        file_hashes["original"], preprocessing_hashes["original"], "bf16"
    )
    cache.put(original_key, features)
    condition_misses = {}
    for condition in ("relevant_cf", "irrelevant_cf"):
        key = VisionCacheKey(
            file_hashes[condition], preprocessing_hashes[condition], "bf16"
        )
        condition_misses[condition] = cache.get(key) is None
    return (
        {
            "bf16_direct_repeat_equal": True,
            "condition_cache_misses": condition_misses,
            "file_hashes_distinct": len(set(file_hashes.values())) == 3,
            "inference_cache_enabled": False,
            "preprocessing_hashes_distinct": True,
        },
        cache,
        original_key,
    )


def run_remaining_conditions(
    config: dict[str, Any], repository_root: Path
) -> dict[str, Any]:
    """After BF16 validity, run two BF16 CFs and all three locked W4 cases."""
    if not torch.cuda.is_available():
        raise RuntimeError("C02-S0 remaining conditions require CUDA")
    paths = _paths(config, repository_root)
    preflight = read_json(paths["processed_output"] / "bf16_preflight_summary.json")
    if preflight["outcome"] != "PASS":
        raise RuntimeError(
            "BF16 task formulation is invalid; remaining conditions forbidden"
        )
    manifest_hash = sha256_file(paths["manifest"])
    if manifest_hash != preflight["manifest_sha256"]:
        raise RuntimeError("manifest changed after BF16 preflight")
    manifest = _load_manifest(paths["manifest"], manifest_hash)
    original_rows = _load_manifest(paths["raw_output"] / "bf16_original.jsonl")
    original_by_expression = {int(row["expression_id"]): row for row in original_rows}
    configure_determinism(int(config["seed"]))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    model, processor = _load_model_and_processor(config, repository_root)
    first_record = manifest[0]
    first_prompt = _active_prompt(first_record, config, preflight)
    cache_audit, cache, original_cache_key = _cache_isolation_precheck(
        model, processor, first_record, first_prompt, repository_root
    )
    rows: list[dict[str, Any]] = []
    repeatability_pass = True
    js_repeatability_pass = True
    for record in manifest:
        reference = _reference(
            original_by_expression[int(record["expression_id"])][
                "reference_distribution"
            ]
        )
        prompt = _active_prompt(record, config, preflight)
        for condition, key in (
            ("bf16_relevant_cf", "relevant_cf_path"),
            ("bf16_irrelevant_cf", "irrelevant_cf_path"),
        ):
            row = _run_case(
                model,
                processor,
                record=record,
                image_path=repository_root / str(record[key]),
                prompt=prompt,
                condition=condition,
                config=config,
                reference=reference,
            )
            rows.append(row)
            if record is first_record:
                repeated = _run_case(
                    model,
                    processor,
                    record=record,
                    image_path=repository_root / str(record[key]),
                    prompt=prompt,
                    condition=condition,
                    config=config,
                    reference=reference,
                )
                _assert_result_repeat(row, repeated)

    module_manifest = build_module_manifest(model)
    quantizer = QuantizerSpec(**config["quantizer"])
    quantizer.validate()
    quantization_audits, _ = quantize_parameters_in_place(
        model,
        all_quantizable_tensor_names(module_manifest),
        quantizer,
        capture_original=False,
    )
    all_w4_changed = all(audit.changed for audit in quantization_audits)
    w4_cache_key = VisionCacheKey(
        original_cache_key.image_sha256,
        original_cache_key.preprocessing_sha256,
        f"w4:{module_manifest['manifest_sha256']}:{sha256_json(quantizer.to_dict())}",
    )
    cross_precision_cache_miss = cache.get(w4_cache_key) is None
    cache_audit["cross_precision_cache_miss"] = cross_precision_cache_miss
    cache_audit["pass"] = bool(
        cache_audit["bf16_direct_repeat_equal"]
        and all(cache_audit["condition_cache_misses"].values())
        and cache_audit["file_hashes_distinct"]
        and cache_audit["preprocessing_hashes_distinct"]
        and cross_precision_cache_miss
        and not cache_audit["inference_cache_enabled"]
    )
    for record in manifest:
        reference = _reference(
            original_by_expression[int(record["expression_id"])][
                "reference_distribution"
            ]
        )
        prompt = _active_prompt(record, config, preflight)
        for condition, key in (
            ("w4_original", "original_image_path"),
            ("w4_relevant_cf", "relevant_cf_path"),
            ("w4_irrelevant_cf", "irrelevant_cf_path"),
        ):
            row = _run_case(
                model,
                processor,
                record=record,
                image_path=repository_root / str(record[key]),
                prompt=prompt,
                condition=condition,
                config=config,
                reference=reference,
            )
            rows.append(row)
            if record is first_record:
                repeated = _run_case(
                    model,
                    processor,
                    record=record,
                    image_path=repository_root / str(record[key]),
                    prompt=prompt,
                    condition=condition,
                    config=config,
                    reference=reference,
                )
                _assert_result_repeat(row, repeated)
    torch.cuda.synchronize()
    gpu_seconds = time.perf_counter() - started
    total_gpu_hours = gpu_seconds / 3600.0 + float(preflight["gpu_hours"])
    if total_gpu_hours > float(config["budget"]["maximum_gpu_hours"]):
        raise RuntimeError("C02-S0 exceeded the 0.5 A800 GPU-hour cap")
    peak_memory = int(torch.cuda.max_memory_allocated())
    write_jsonl(paths["raw_output"] / "remaining_conditions.jsonl", rows)
    write_json(paths["raw_output"] / "module_manifest.json", module_manifest)
    critical_bugs: list[str] = []
    if not all_w4_changed:
        critical_bugs.append("at least one locked W4 tensor was unchanged")
    if not cache_audit["pass"]:
        critical_bugs.append("cache isolation failed")
    summary = {
        "all_w4_tensors_changed": all_w4_changed,
        "cache_isolation": cache_audit,
        "critical_bugs": critical_bugs,
        "gpu_hours": gpu_seconds / 3600.0,
        "js_repeatability_pass": js_repeatability_pass,
        "module_manifest_sha256": module_manifest["manifest_sha256"],
        "peak_memory_bytes": peak_memory,
        "quantized_tensor_count": len(quantization_audits),
        "repeatability_pass": repeatability_pass,
        "w4_definition": {
            "activation_dtype": "bfloat16",
            "kv_dtype": "bfloat16",
            "quantizer": quantizer.to_dict(),
            "weight_only": True,
        },
    }
    write_json(paths["processed_output"] / "remaining_integrity.json", summary)
    final = load_and_analyze(
        raw_directory=paths["raw_output"],
        processed_directory=paths["processed_output"],
        figure_directory=paths["figure_output"],
        manifest_hash=manifest_hash,
    )
    del model
    torch.cuda.empty_cache()
    return final


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate the locked C02-S0 configuration."""
    config = read_json(config_path)
    if (
        config.get("protocol") != "C02-S0"
        or config.get("stage") != "integrity_smoke_test"
    ):
        raise ValueError("configuration is not locked to C02-S0")
    if int(config["selection"]["image_count"]) not in range(12, 21):
        raise ValueError("C02-S0 requires 12--20 images")
    expression_count = int(config["selection"]["image_count"]) * int(
        config["selection"]["expressions_per_image"]
    )
    if expression_count not in range(24, 41):
        raise ValueError("C02-S0 requires 24--40 expressions")
    if float(config["budget"]["maximum_gpu_hours"]) > 0.5:
        raise ValueError("C02-S0 GPU budget cannot exceed 0.5 A800 hours")
    return config


def run_phase(config_path: Path, repository_root: Path, phase: str) -> dict[str, Any]:
    """Execute exactly one authorized C02-S0 phase."""
    config = load_config(config_path)
    if phase == "freeze-manifest":
        return freeze_s0_manifest(config, repository_root)
    if phase == "prepare-counterfactuals":
        return prepare_counterfactuals(config, repository_root)
    if phase == "bf16-original":
        return run_bf16_original(config, repository_root)
    if phase == "remaining":
        return run_remaining_conditions(config, repository_root)
    raise ValueError(f"unknown C02-S0 phase: {phase}")
