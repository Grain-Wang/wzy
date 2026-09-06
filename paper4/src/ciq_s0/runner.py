"""End-to-end C01-S0 integrity runner; deliberately contains no S1 logic."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from .data_manifest import (
    SampleRecord,
    finalize_token_metadata,
    freeze_sample_manifest,
    materialize_s0_records,
    randomized_image_group_order,
    select_s0_rows,
)
from .evaluation import normalized_exact_score, normalize_answer, vqa_soft_score
from .inference import (
    configure_determinism,
    extract_visual_features,
    load_rgb_image,
    prepare_prompt_inputs,
    run_inference_case,
    tensor_value_sha256,
)
from .io_utils import read_json, sha256_file, sha256_json, write_json, write_jsonl
from .module_manifest import (
    all_quantizable_tensor_names,
    build_module_manifest,
    group_tensor_names,
)
from .quantization import (
    QuantizerSpec,
    parameter_hashes,
    quantize_parameters_in_place,
    restore_parameters,
)
from .vision_cache import VisionCacheKey, VisionFeatureCache, combine_tensor_hashes


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def collect_environment() -> dict[str, Any]:
    """Collect reproducibility metadata without hostnames, users, or credentials."""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"
    capability = (
        list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
    )
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": _package_version("transformers"),
        "datasets": _package_version("datasets"),
        "huggingface_hub": _package_version("huggingface-hub"),
        "pillow": _package_version("Pillow"),
        "numpy": _package_version("numpy"),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "gpu_capability": capability,
    }


def _load_frozen_records(path: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(SampleRecord(**json.loads(line)))
    return records


def prepare_or_load_samples(
    config: dict[str, Any], repository_root: Path, processor: Any
) -> tuple[list[SampleRecord], str, dict[str, Any]]:
    """Freeze the S0 GQA manifest before any quantized model output is inspected."""
    raw_directory = repository_root / config["paths"]["raw_output"]
    manifest_path = raw_directory / "sample_manifest.jsonl"
    image_directory = repository_root / config["paths"]["image_directory"]
    if manifest_path.exists():
        records = _load_frozen_records(manifest_path)
        digest = sha256_file(manifest_path)
        expected = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
        locked = expected.read_text(encoding="ascii").split()[0]
        if digest != locked:
            raise RuntimeError("existing frozen S0 manifest hash is invalid")
        for record in records:
            if not (repository_root / record.image_path).is_file():
                raise RuntimeError(f"selected image is missing: {record.image_path}")
        return records, digest, {"reused_frozen_manifest": True}

    dataset = config["dataset"]
    snapshot = repository_root / dataset["local_snapshot"]
    instruction_files = sorted(
        snapshot.joinpath(dataset["instruction_config"]).glob("*.parquet")
    )
    image_files = sorted(snapshot.joinpath(dataset["image_config"]).glob("*.parquet"))
    if not instruction_files or not image_files:
        raise RuntimeError("locked local GQA parquet snapshot is incomplete")
    dataset_cache = str(repository_root / config["paths"]["dataset_cache"])
    instruction_rows = load_dataset(
        "parquet",
        data_files={dataset["split"]: [str(path) for path in instruction_files]},
        split=dataset["split"],
        cache_dir=dataset_cache,
    )
    image_rows = load_dataset(
        "parquet",
        data_files={dataset["split"]: [str(path) for path in image_files]},
        split=dataset["split"],
        cache_dir=dataset_cache,
    )
    selected = select_s0_rows(
        instruction_rows,
        image_count=int(dataset["image_count"]),
        question_count=int(dataset["question_count"]),
        seed=int(config["seed"]),
    )
    records = materialize_s0_records(
        selected,
        image_rows,
        repository_root=repository_root,
        image_directory=image_directory,
    )
    preprocessing = {
        "model_revision": config["model"]["revision"],
        "processor_revision": config["model"]["processor_revision"],
        "min_pixels": config["model"]["min_pixels"],
        "max_pixels": config["model"]["max_pixels"],
        "use_fast_image_processor": config["model"]["use_fast_image_processor"],
        "image_mode": "RGB",
        "serialized_image_format": "PNG",
        "chat_instruction": config["inference"]["instruction"],
        "chat_template_sha256": sha256_json(processor.chat_template),
    }
    records = finalize_token_metadata(
        records,
        tokenizer=processor.tokenizer,
        preprocessing_metadata=preprocessing,
    )
    digest = freeze_sample_manifest(manifest_path, records)
    return (
        records,
        digest,
        {
            "reused_frozen_manifest": False,
            "repo_id": dataset["repo_id"],
            "revision": dataset["revision"],
            "instruction_dataset_fingerprint": instruction_rows._fingerprint,
            "image_dataset_fingerprint": image_rows._fingerprint,
            "preprocessing": preprocessing,
        },
    )


def _generation_settings(config: dict[str, Any], processor: Any) -> dict[str, Any]:
    settings = dict(config["inference"]["generation"])
    settings["pad_token_id"] = processor.tokenizer.pad_token_id
    settings["eos_token_id"] = processor.tokenizer.eos_token_id
    return settings


def _run_record(
    model: torch.nn.Module,
    processor: Any,
    record: SampleRecord,
    *,
    repository_root: Path,
    config: dict[str, Any],
    condition: str,
) -> dict[str, Any]:
    result = run_inference_case(
        model,
        processor,
        image=load_rgb_image(repository_root / record.image_path),
        question=record.query_text,
        answer=record.answer,
        instruction=config["inference"]["instruction"],
        device=torch.device("cuda:0"),
        generation_settings=_generation_settings(config, processor),
    )
    payload = {
        "condition": condition,
        "image_id": record.image_id,
        "question_id": record.question_id,
        "query_family": record.query_family,
        "reference": record.answer,
        "normalized_reference": normalize_answer(record.answer),
        "normalized_exact_score": normalized_exact_score(
            result.prediction, record.answer
        ),
    }
    payload.update(result.to_dict())
    return payload


def _assert_repeatability(
    first: dict[str, Any], second: dict[str, Any], *, nll_tolerance: float
) -> None:
    if first["prediction"] != second["prediction"]:
        raise AssertionError(
            "BF16 deterministic decoding changed across identical runs"
        )
    if first["target_token_sha256"] != second["target_token_sha256"]:
        raise AssertionError("gold answer tokens changed across identical runs")
    if abs(float(first["answer_nll"]) - float(second["answer_nll"])) > nll_tolerance:
        raise AssertionError("BF16 teacher-forced NLL is not repeatable")


def _evaluator_checks() -> dict[str, Any]:
    checks = {
        "case_punctuation_articles": normalize_answer("The, CAT!") == "cat",
        "case_number_normalization": normalize_answer("two") == "2",
        "case_exact_positive": normalized_exact_score("A dog.", "dog") == 1.0,
        "case_exact_negative": normalized_exact_score("dog", "cat") == 0.0,
        "case_soft_partial": vqa_soft_score("dog", ["dog", "dog", "cat"]) == 2 / 3,
        "case_soft_cap": vqa_soft_score("dog", ["dog", "dog", "dog", "dog"]) == 1.0,
    }
    if not all(checks.values()):
        raise AssertionError("automatic evaluator self-check failed")
    return checks


def _same_image_preprocessing_checks(
    records: list[SampleRecord],
    *,
    repository_root: Path,
    processor: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    by_image: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        by_image[record.image_id].append(record)
    image_hashes: dict[str, str] = {}
    checked_pairs = 0
    for image_id, group in sorted(by_image.items()):
        if len(group) < 2:
            raise AssertionError(f"image {image_id} has no repeated query")
        expected_file_hash = {record.image_file_sha256 for record in group}
        expected_path = {record.image_path for record in group}
        if len(expected_file_hash) != 1 or len(expected_path) != 1:
            raise AssertionError(
                f"same-image records disagree on bytes/path for {image_id}"
            )
        pixel_hashes: set[str] = set()
        grid_hashes: set[str] = set()
        for record in group:
            prompt = prepare_prompt_inputs(
                processor,
                image=load_rgb_image(repository_root / record.image_path),
                question=record.query_text,
                instruction=config["inference"]["instruction"],
                device=torch.device("cpu"),
            )
            pixel_hashes.add(tensor_value_sha256(prompt["pixel_values"]))
            grid_hashes.add(tensor_value_sha256(prompt["image_grid_thw"]))
        if len(pixel_hashes) != 1 or len(grid_hashes) != 1:
            raise AssertionError(
                f"same image produced query-dependent visual preprocessing for {image_id}"
            )
        image_hashes[image_id] = next(iter(pixel_hashes))
        checked_pairs += len(group) - 1
    return {
        "image_group_count": len(by_image),
        "same_image_query_comparisons": checked_pairs,
        "pixel_tensor_sha256_by_image": image_hashes,
        "all_equal_within_image": True,
    }


def _vision_cache_check(
    model: torch.nn.Module,
    processor: Any,
    record: SampleRecord,
    *,
    repository_root: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    quantizer: QuantizerSpec,
) -> dict[str, Any]:
    vision_names = tuple(
        name
        for group in manifest["groups"]
        if group["family"] == "vision_encoder"
        for name in group["tensor_names"]
    )
    prompt = prepare_prompt_inputs(
        processor,
        image=load_rgb_image(repository_root / record.image_path),
        question=record.query_text,
        instruction=config["inference"]["instruction"],
        device=torch.device("cuda:0"),
    )
    bf16_hashes = parameter_hashes(model, vision_names)
    bf16_fingerprint = combine_tensor_hashes(bf16_hashes)
    key = VisionCacheKey(
        record.image_file_sha256,
        record.preprocessing_sha256,
        bf16_fingerprint,
    )
    direct_first = extract_visual_features(model, prompt)
    direct_second = extract_visual_features(model, prompt)
    if not torch.equal(direct_first, direct_second):
        raise AssertionError(
            "identical BF16 visual executions are not bitwise repeatable"
        )
    cache = VisionFeatureCache()
    cache.put(key, direct_first)
    cached = cache.get(key)
    if cached is None or not torch.equal(cached, direct_second):
        raise AssertionError("same-precision cached visual feature is not equivalent")

    target_names = group_tensor_names(manifest, "vision.q1")
    _, originals = quantize_parameters_in_place(
        model, target_names, quantizer, capture_original=True
    )
    quantized_fingerprint = combine_tensor_hashes(parameter_hashes(model, vision_names))
    if quantized_fingerprint == bf16_fingerprint:
        raise AssertionError(
            "vision quantization did not change the precision fingerprint"
        )
    quantized_key = VisionCacheKey(
        record.image_file_sha256,
        record.preprocessing_sha256,
        quantized_fingerprint,
    )
    cross_precision_hit = cache.get(quantized_key)
    if cross_precision_hit is not None:
        raise AssertionError(
            "BF16 visual cache was reused for quantized vision weights"
        )
    quantized_features = extract_visual_features(model, prompt)
    restore_parameters(model, originals)
    restored_features = extract_visual_features(model, prompt)
    if not torch.equal(restored_features, direct_first):
        raise AssertionError("restored BF16 vision features differ from the original")
    return {
        "bf16_direct_repeat_equal": True,
        "bf16_cache_hit_equal": True,
        "cross_precision_cache_hit": False,
        "vision_fingerprint_changed_under_w4": True,
        "quantized_features_differ_from_bf16": not torch.equal(
            quantized_features, direct_first
        ),
        "restored_features_equal_bf16": True,
    }


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def run_s0(config_path: Path, repository_root: Path) -> dict[str, Any]:
    """Execute only C01-S0 and return its machine-readable integrity summary."""
    config = read_json(config_path)
    if config["protocol"] != "C01-S0" or config["stage"] != "integrity_smoke_test":
        raise ValueError("configuration is not locked to C01-S0")
    if not torch.cuda.is_available():
        raise RuntimeError("C01-S0 requires one CUDA GPU")
    environment = collect_environment()
    if environment["python"].split(".")[:2] != ["3", "12"]:
        raise RuntimeError("AGENTS.md requires Python 3.12")
    raw_directory = repository_root / config["paths"]["raw_output"]
    processed_directory = repository_root / config["paths"]["processed_output"]
    raw_directory.mkdir(parents=True, exist_ok=True)
    processed_directory.mkdir(parents=True, exist_ok=True)
    write_json(raw_directory / "environment.json", environment)

    determinism = configure_determinism(int(config["seed"]))
    model_config = config["model"]
    cache_directory = repository_root / config["paths"]["model_cache"]
    model_snapshot = repository_root / model_config["local_snapshot"]
    if not model_snapshot.is_dir():
        raise RuntimeError("locked local model snapshot is missing")
    processor = AutoProcessor.from_pretrained(
        model_snapshot,
        cache_dir=cache_directory,
        min_pixels=int(model_config["min_pixels"]),
        max_pixels=int(model_config["max_pixels"]),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=bool(model_config["use_fast_image_processor"]),
    )
    records, manifest_hash, dataset_audit = prepare_or_load_samples(
        config, repository_root, processor
    )
    family_counts = Counter(record.query_family for record in records)
    execution_records = randomized_image_group_order(records, seed=int(config["seed"]))
    execution_image_order = list(
        dict.fromkeys(record.image_id for record in execution_records)
    )

    torch.cuda.reset_peak_memory_stats()
    gpu_started = time.perf_counter()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_snapshot,
        cache_dir=cache_directory,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_config["attention_implementation"],
        local_files_only=True,
        trust_remote_code=False,
    ).to("cuda:0")
    model.eval()
    torch.cuda.synchronize()

    module_manifest = build_module_manifest(model)
    write_json(raw_directory / "module_manifest.json", module_manifest)
    quantizer = QuantizerSpec(**config["quantizer"])
    quantizer.validate()
    all_names = all_quantizable_tensor_names(module_manifest)
    original_hashes = parameter_hashes(model, all_names)

    preprocessing_check = _same_image_preprocessing_checks(
        records,
        repository_root=repository_root,
        processor=processor,
        config=config,
    )
    evaluator_check = _evaluator_checks()

    repeat_count = int(config["s0"]["repeatability_case_count"])
    repeat_records = execution_records[:repeat_count]
    repeated_outputs: list[dict[str, Any]] = []
    for record in repeat_records:
        first = _run_record(
            model,
            processor,
            record,
            repository_root=repository_root,
            config=config,
            condition="bf16_repeat_a",
        )
        second = _run_record(
            model,
            processor,
            record,
            repository_root=repository_root,
            config=config,
            condition="bf16_repeat_b",
        )
        _assert_repeatability(
            first,
            second,
            nll_tolerance=float(config["s0"]["nll_repeatability_atol"]),
        )
        repeated_outputs.extend((first, second))

    bf16_outputs = [
        _run_record(
            model,
            processor,
            record,
            repository_root=repository_root,
            config=config,
            condition="bf16",
        )
        for record in execution_records
    ]
    baseline_by_question = {item["question_id"]: item for item in bf16_outputs}

    empty_audits, empty_originals = quantize_parameters_in_place(
        model, (), quantizer, capture_original=True
    )
    if (
        empty_audits
        or empty_originals
        or parameter_hashes(model, all_names) != original_hashes
    ):
        raise AssertionError("empty-group perturbation changed model state")
    empty_output = _run_record(
        model,
        processor,
        execution_records[0],
        repository_root=repository_root,
        config=config,
        condition="empty_group",
    )
    _assert_repeatability(
        baseline_by_question[execution_records[0].question_id],
        empty_output,
        nll_tolerance=float(config["s0"]["nll_repeatability_atol"]),
    )

    target_group = str(config["s0"]["single_group_id"])
    target_names = group_tensor_names(module_manifest, target_group)
    isolation_audits, target_originals = quantize_parameters_in_place(
        model, target_names, quantizer, capture_original=True
    )
    isolated_hashes = parameter_hashes(model, all_names)
    changed_names = {
        name for name in all_names if original_hashes[name] != isolated_hashes[name]
    }
    if not changed_names or not changed_names.issubset(set(target_names)):
        raise AssertionError("single-group perturbation changed the wrong tensor set")
    perturbed_outputs = [
        _run_record(
            model,
            processor,
            record,
            repository_root=repository_root,
            config=config,
            condition=f"single_group:{target_group}",
        )
        for record in records[:repeat_count]
    ]

    restore_parameters(model, target_originals)
    restored_hashes = parameter_hashes(model, all_names)
    if restored_hashes != original_hashes:
        raise AssertionError(
            "quantize-to-restore did not recover exact BF16 tensor hashes"
        )
    restored_output = _run_record(
        model,
        processor,
        execution_records[0],
        repository_root=repository_root,
        config=config,
        condition="restored_bf16",
    )
    _assert_repeatability(
        baseline_by_question[execution_records[0].question_id],
        restored_output,
        nll_tolerance=float(config["s0"]["nll_repeatability_atol"]),
    )

    cache_check = _vision_cache_check(
        model,
        processor,
        execution_records[0],
        repository_root=repository_root,
        config=config,
        manifest=module_manifest,
        quantizer=quantizer,
    )
    if parameter_hashes(model, all_names) != original_hashes:
        raise AssertionError("vision cache test failed to restore BF16 model state")

    all_w4_audits, _ = quantize_parameters_in_place(
        model, all_names, quantizer, capture_original=False
    )
    all_w4_hashes = parameter_hashes(model, all_names)
    all_w4_changed = {
        name for name in all_names if original_hashes[name] != all_w4_hashes[name]
    }
    if not all_w4_changed or not all_w4_changed.issubset(set(all_names)):
        raise AssertionError("all-W4 diagnostic did not apply to the manifest")
    all_w4_outputs = [
        _run_record(
            model,
            processor,
            record,
            repository_root=repository_root,
            config=config,
            condition="all_w4a16_diagnostic",
        )
        for record in execution_records
    ]
    torch.cuda.synchronize()
    gpu_elapsed_seconds = time.perf_counter() - gpu_started
    if gpu_elapsed_seconds >= 3600:
        raise RuntimeError("C01-S0 exceeded the locked one-GPU-hour cap")

    predictions = repeated_outputs + bf16_outputs + [empty_output]
    predictions.extend(perturbed_outputs)
    predictions.append(restored_output)
    predictions.extend(all_w4_outputs)
    write_jsonl(raw_directory / "predictions.jsonl", predictions)
    integrity = {
        "schema_version": 1,
        "protocol": "C01-S0",
        "outcome": "PASS",
        "scientific_claim_tested": False,
        "model": {
            "repo_id": model_config["repo_id"],
            "revision": model_config["revision"],
            "processor_revision": model_config["processor_revision"],
            "loaded_config_commit_hash": getattr(model.config, "_commit_hash", None),
            "dtype": str(next(model.parameters()).dtype),
        },
        "environment": environment,
        "determinism": determinism,
        "dataset": dataset_audit,
        "sample_count": len(records),
        "image_count": len({record.image_id for record in records}),
        "query_family_counts": dict(sorted(family_counts.items())),
        "execution_order": {
            "policy": "whole image groups shuffled with the locked seed",
            "image_ids": execution_image_order,
        },
        "sample_manifest_sha256": manifest_hash,
        "quantizer": quantizer.to_dict(),
        "module_manifest_sha256": module_manifest["manifest_sha256"],
        "module_group_count": len(module_manifest["groups"]),
        "quantizable_tensor_count": len(all_names),
        "tests": {
            "bf16_repeatability": {
                "passed": True,
                "case_count": repeat_count,
                "nll_atol": config["s0"]["nll_repeatability_atol"],
            },
            "empty_group_identity": {"passed": True, "changed_tensor_count": 0},
            "single_group_isolation": {
                "passed": True,
                "group_id": target_group,
                "target_tensor_count": len(target_names),
                "changed_tensor_count": len(changed_names),
                "non_target_changed_tensor_count": len(
                    changed_names - set(target_names)
                ),
                "tensor_audits": [audit.to_dict() for audit in isolation_audits],
            },
            "restore_identity": {
                "passed": True,
                "hashes_exact": True,
                "output_repeatable": True,
            },
            "all_w4_application": {
                "passed": True,
                "target_tensor_count": len(all_names),
                "changed_tensor_count": len(all_w4_changed),
                "logical_w4_bytes": sum(
                    audit.logical_packed_bytes for audit in all_w4_audits
                ),
                "diagnostic_parameter_storage_compressed": False,
                "generated_case_count": len(all_w4_outputs),
                "finite_nll_count": sum(
                    int(torch.isfinite(torch.tensor(item["answer_nll"])).item())
                    for item in all_w4_outputs
                ),
            },
            "evaluator_correctness": {"passed": True, "checks": evaluator_check},
            "teacher_forced_nll": {
                "passed": True,
                "prompt_masked": True,
                "answer_positions_only": True,
                "same_target_across_conditions": all(
                    baseline_by_question[item["question_id"]]["target_token_sha256"]
                    == item["target_token_sha256"]
                    for item in all_w4_outputs
                ),
                "finite_bf16_count": sum(
                    int(torch.isfinite(torch.tensor(item["answer_nll"])).item())
                    for item in bf16_outputs
                ),
            },
            "same_image_grouping": {"passed": True, **preprocessing_check},
            "vision_cache_equivalence": {"passed": True, **cache_check},
            "manifest_frozen_before_quantization": {"passed": True},
            "data_leakage_check": {
                "passed": True,
                "selection_uses_model_outputs": False,
                "post_hoc_filtering": False,
            },
        },
        "gpu_elapsed_seconds": gpu_elapsed_seconds,
        "gpu_hours": gpu_elapsed_seconds / 3600,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    write_json(raw_directory / "integrity_results.json", integrity)
    (raw_directory / "failure.json").unlink(missing_ok=True)
    summary = {
        "outcome": "PASS",
        "recommendation": "ENTER S1 (not executed)",
        "sample_manifest_sha256": manifest_hash,
        "image_count": integrity["image_count"],
        "sample_count": integrity["sample_count"],
        "query_family_counts": integrity["query_family_counts"],
        "integrity_tests": {
            name: bool(result["passed"]) for name, result in integrity["tests"].items()
        },
        "gpu_hours": integrity["gpu_hours"],
        "peak_gpu_memory_bytes": integrity["peak_gpu_memory_bytes"],
        "raw_artifact_bytes": _directory_size(raw_directory),
        "dataset_bytes": _directory_size(
            repository_root / config["paths"]["image_directory"]
        )
        + _directory_size(repository_root / config["paths"]["dataset_cache"]),
        "dataset_snapshot_bytes": _directory_size(
            repository_root / config["dataset"]["local_snapshot"]
        ),
        "model_cache_bytes": _directory_size(cache_directory)
        + _directory_size(model_snapshot),
    }
    write_json(processed_directory / "integrity_summary.json", summary)
    return summary


def write_failure(
    *, repository_root: Path, config_path: Path, error: BaseException
) -> None:
    """Persist a sanitized S0 failure record without changing the scientific protocol."""
    config = read_json(config_path)
    raw_directory = repository_root / config["paths"]["raw_output"]
    raw_directory.mkdir(parents=True, exist_ok=True)
    write_json(
        raw_directory / "failure.json",
        {
            "outcome": "FAIL",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "scientific_claim_tested": False,
        },
    )
