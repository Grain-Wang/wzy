"""Strictly staged execution for C01-S1 prepare, A, B, and coarse sweep."""

from __future__ import annotations

import json
import math
import shutil
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from ..ciq_s0.inference import configure_determinism, load_rgb_image
from ..ciq_s0.io_utils import read_json, sha256_file, write_json, write_jsonl
from ..ciq_s0.module_manifest import (
    all_quantizable_tensor_names,
    build_module_manifest,
    group_tensor_names,
)
from ..ciq_s0.quantization import (
    QuantizerSpec,
    parameter_hashes,
    quantize_parameters_in_place,
    restore_parameters,
)
from ..ciq_s0.runner import collect_environment
from .distributions import FixedSupportDistribution
from .evaluation import aggregate_condition_metrics, attach_scores
from .inference import run_s1_inference_case
from .precision import (
    W8RestorationSpec,
    apply_restoration_profile,
    profile_logical_bytes,
    reset_profile_to_w4,
)
from .profile_controls import (
    aggregate_executed_profiles,
    build_profile_definitions,
    select_execution_images,
)
from .sampling import (
    S1SampleRecord,
    finalize_s1_metadata,
    freeze_s1_manifest,
    materialize_s1_records,
    select_s1_rows,
    validate_sampling_constraints,
    whole_image_execution_order,
)


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol") != "C01-S1":
        raise ValueError("configuration is not locked to C01-S1")
    QuantizerSpec(**config["quantizer"]["w4"]).validate()


def _raw_directory(config: Mapping[str, Any], repository_root: Path) -> Path:
    return repository_root / str(config["paths"]["raw_output"])


def _processed_directory(config: Mapping[str, Any], repository_root: Path) -> Path:
    return repository_root / str(config["paths"]["processed_output"])


def _load_processor(config: Mapping[str, Any], repository_root: Path) -> Any:
    model_config = config["model"]
    snapshot = repository_root / str(model_config["local_snapshot"])
    if not snapshot.is_dir():
        raise RuntimeError("locked local model snapshot is missing")
    return AutoProcessor.from_pretrained(
        snapshot,
        cache_dir=repository_root / str(config["paths"]["model_cache"]),
        min_pixels=int(model_config["min_pixels"]),
        max_pixels=int(model_config["max_pixels"]),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=bool(model_config["use_fast_image_processor"]),
    )


def _load_model(config: Mapping[str, Any], repository_root: Path) -> torch.nn.Module:
    model_config = config["model"]
    snapshot = repository_root / str(model_config["local_snapshot"])
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        snapshot,
        cache_dir=repository_root / str(config["paths"]["model_cache"]),
        torch_dtype=torch.bfloat16,
        attn_implementation=str(model_config["attention_implementation"]),
        local_files_only=True,
        trust_remote_code=False,
    ).to("cuda:0")
    model.eval()
    return model


def _load_records(path: Path) -> list[S1SampleRecord]:
    records = [
        S1SampleRecord(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_hash = (
        path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").split()[0]
    )
    if sha256_file(path) != expected_hash:
        raise RuntimeError(f"frozen manifest hash mismatch: {path.name}")
    return records


def _generation_settings(config: Mapping[str, Any], processor: Any) -> dict[str, Any]:
    settings = dict(config["inference"]["generation"])
    settings["pad_token_id"] = processor.tokenizer.pad_token_id
    settings["eos_token_id"] = processor.tokenizer.eos_token_id
    return settings


def _compact_distribution(payload: Mapping[str, Any]) -> FixedSupportDistribution:
    value = payload.get("reference_distribution")
    if not isinstance(value, Mapping):
        raise ValueError("BF16 row lacks a compact reference distribution")
    return FixedSupportDistribution(
        token_ids=[[int(token) for token in row] for row in value["token_ids"]],
        probabilities=[
            [float(probability) for probability in row]
            for row in value["probabilities"]
        ],
    )


def _run_record(
    model: torch.nn.Module,
    processor: Any,
    record: S1SampleRecord,
    *,
    repository_root: Path,
    config: Mapping[str, Any],
    condition: str,
    bf16_row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    reference_distribution = (
        _compact_distribution(bf16_row) if bf16_row is not None else None
    )
    result = run_s1_inference_case(
        model,
        processor,
        image=load_rgb_image(repository_root / record.image_path),
        question=record.query_text,
        answer=record.answer,
        instruction=str(config["inference"]["instruction"]),
        device=torch.device("cuda:0"),
        generation_settings=_generation_settings(config, processor),
        distribution_top_k=int(config["distribution"]["top_k"]),
        bf16_reference=reference_distribution,
    )
    payload: dict[str, Any] = {
        "condition": condition,
        "image_id": record.image_id,
        "question_id": record.question_id,
        "query_family": record.query_family,
        "reference": record.answer,
        "question_token_length": record.question_token_length,
        "answer_token_length": record.answer_token_length,
        "visual_token_count": record.visual_token_count,
        "prompt_template_sha256": record.prompt_template_sha256,
    }
    payload.update(result.to_dict())
    payload.update(
        attach_scores(
            result.prediction,
            record.answer,
            maximum_words=int(config["inference"]["maximum_well_formed_words"]),
        )
    )
    if bf16_row is not None:
        if result.target_token_sha256 != str(bf16_row["target_token_sha256"]):
            raise AssertionError(
                "gold-answer target changed across precision conditions"
            )
        payload["delta_nll"] = result.answer_nll - float(bf16_row["answer_nll"])
        payload["answer_flip"] = float(
            payload["official_prediction"] != bf16_row["official_prediction"]
        )
        payload["bf16_correct_to_condition_wrong"] = float(
            bool(bf16_row["official_score"]) and not bool(payload["official_score"])
        )
    return payload


def _summaries_by_family(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["query_family"])].append(row)
    return {
        family: aggregate_condition_metrics(values)
        for family, values in sorted(by_family.items())
    }


def _gpu_environment() -> dict[str, Any]:
    environment = collect_environment()
    if environment["python"].split(".")[:2] != ["3", "12"]:
        raise RuntimeError("AGENTS.md requires Python 3.12")
    if not torch.cuda.is_available():
        raise RuntimeError("C01-S1 requires one CUDA GPU")
    return environment


def prepare_s1(config_path: Path, repository_root: Path) -> dict[str, Any]:
    """Freeze the candidate pool before any S1 model output exists."""
    config = read_json(config_path)
    _validate_config(config)
    raw_directory = _raw_directory(config, repository_root)
    processed_directory = _processed_directory(config, repository_root)
    raw_directory.mkdir(parents=True, exist_ok=True)
    processed_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_directory / "candidate_manifest.jsonl"
    if manifest_path.exists():
        records = _load_records(manifest_path)
        sampling_summary = validate_sampling_constraints(
            records,
            minimum_images_per_family=int(
                config["dataset"]["minimum_images_per_family"]
            ),
        )
        result = {
            "outcome": "CANDIDATE MANIFEST REUSED",
            "manifest_sha256": sha256_file(manifest_path),
            **sampling_summary,
        }
        write_json(processed_directory / "candidate_manifest_summary.json", result)
        return result

    processor = _load_processor(config, repository_root)
    dataset_config = config["dataset"]
    snapshot = repository_root / str(dataset_config["local_snapshot"])
    instruction_files = sorted(
        snapshot.joinpath(str(dataset_config["instruction_config"])).glob("*.parquet")
    )
    image_files = sorted(
        snapshot.joinpath(str(dataset_config["image_config"])).glob("*.parquet")
    )
    if not instruction_files or not image_files:
        raise RuntimeError("locked local GQA parquet snapshot is incomplete")
    dataset_cache = str(repository_root / str(config["paths"]["dataset_cache"]))
    instruction_rows = load_dataset(
        "parquet",
        data_files={
            str(dataset_config["split"]): [str(path) for path in instruction_files]
        },
        split=str(dataset_config["split"]),
        cache_dir=dataset_cache,
    )
    image_rows = load_dataset(
        "parquet",
        data_files={str(dataset_config["split"]): [str(path) for path in image_files]},
        split=str(dataset_config["split"]),
        cache_dir=dataset_cache,
    )
    selected = select_s1_rows(
        instruction_rows,
        image_count=int(dataset_config["image_count"]),
        questions_per_image=int(dataset_config["questions_per_image"]),
        families_per_image=int(dataset_config["families_per_image"]),
        questions_per_cell=int(dataset_config["questions_per_cell"]),
        minimum_images_per_family=int(dataset_config["minimum_images_per_family"]),
        seed=int(config["seed"]),
    )
    records = materialize_s1_records(
        selected,
        image_rows,
        repository_root=repository_root,
        image_directory=repository_root / str(config["paths"]["image_directory"]),
    )
    preprocessing = {
        "model_revision": config["model"]["revision"],
        "processor_revision": config["model"]["processor_revision"],
        "min_pixels": config["model"]["min_pixels"],
        "max_pixels": config["model"]["max_pixels"],
        "use_fast_image_processor": config["model"]["use_fast_image_processor"],
        "image_mode": "RGB",
        "serialized_image_format": "PNG",
    }
    records = finalize_s1_metadata(
        records,
        processor=processor,
        instruction=str(config["inference"]["instruction"]),
        preprocessing_metadata=preprocessing,
        repository_root=repository_root,
    )
    sampling_summary = validate_sampling_constraints(
        records,
        minimum_images_per_family=int(dataset_config["minimum_images_per_family"]),
    )
    digest = freeze_s1_manifest(manifest_path, records)
    result = {
        "outcome": "CANDIDATE MANIFEST FROZEN",
        "manifest_sha256": digest,
        "selection_uses_model_outputs": False,
        "post_hoc_filtering": False,
        "instruction_dataset_fingerprint": instruction_rows._fingerprint,
        "image_dataset_fingerprint": image_rows._fingerprint,
        "preprocessing": preprocessing,
        **sampling_summary,
    }
    write_json(processed_directory / "candidate_manifest_summary.json", result)
    return result


def _load_candidate(
    config: Mapping[str, Any], repository_root: Path
) -> tuple[list[S1SampleRecord], str]:
    path = _raw_directory(config, repository_root) / "candidate_manifest.jsonl"
    records = _load_records(path)
    validate_sampling_constraints(
        records,
        minimum_images_per_family=int(config["dataset"]["minimum_images_per_family"]),
    )
    return records, sha256_file(path)


def run_s1_a(config_path: Path, repository_root: Path) -> dict[str, Any]:
    """Run BF16 only and decide the preregistered baseline validity gate."""
    config = read_json(config_path)
    _validate_config(config)
    environment = _gpu_environment()
    configure_determinism(int(config["seed"]))
    records, manifest_hash = _load_candidate(config, repository_root)
    processor = _load_processor(config, repository_root)
    raw_directory = _raw_directory(config, repository_root)
    processed_directory = _processed_directory(config, repository_root)
    environment_path = raw_directory / "environment.json"
    output_path = raw_directory / "bf16_predictions.jsonl"
    reused_completed_output = _validate_completed_group(output_path, records, "bf16")
    if reused_completed_output:
        outputs = _read_jsonl(output_path)
        module_manifest = read_json(raw_directory / "module_manifest.json")
        elapsed = max(
            0.0, output_path.stat().st_mtime - environment_path.stat().st_mtime
        )
        peak_memory: int | None = None
        loaded_commit_hash = config["model"]["revision"]
    else:
        write_json(environment_path, environment)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = _load_model(config, repository_root)
        module_manifest = build_module_manifest(model)
        write_json(raw_directory / "module_manifest.json", module_manifest)
        ordered = whole_image_execution_order(records, seed=int(config["seed"]))
        outputs = []
        previous_image = None
        for index, record in enumerate(ordered, start=1):
            if record.image_id != previous_image:
                print(
                    f"S1-A image {len({row['image_id'] for row in outputs}) + 1}/90",
                    flush=True,
                )
                previous_image = record.image_id
            outputs.append(
                _run_record(
                    model,
                    processor,
                    record,
                    repository_root=repository_root,
                    config=config,
                    condition="bf16",
                    bf16_row=None,
                )
            )
            if index % 60 == 0:
                print(f"S1-A questions {index}/{len(ordered)}", flush=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        write_jsonl(output_path, outputs)
        peak_memory = torch.cuda.max_memory_allocated()
        loaded_commit_hash = getattr(model.config, "_commit_hash", None)
    overall = aggregate_condition_metrics(outputs)
    by_family = _summaries_by_family(outputs)
    gate = config["gates"]["baseline"]
    adequate = [
        summary
        for summary in by_family.values()
        if int(summary["sample_count"]) >= 30 and int(summary["image_count"]) >= 15
    ]
    checks = {
        "finite_nll_rate": sum(
            math.isfinite(float(row["answer_nll"])) for row in outputs
        )
        / len(outputs)
        >= float(gate["minimum_finite_nll_rate"]),
        "empty_rate": float(overall["empty_generation_rate"])
        <= float(gate["maximum_empty_rate"]),
        "malformed_rate": float(overall["malformed_generation_rate"])
        <= float(gate["maximum_malformed_rate"]),
        "overall_official_score": float(overall["official_score"])
        >= float(gate["minimum_overall_official_score"]),
        "family_dynamic_range": sum(
            float(summary["official_score"])
            >= float(gate["minimum_family_official_score"])
            for summary in adequate
        )
        >= int(gate["minimum_adequate_families"]),
    }
    outcome = "PASS" if all(checks.values()) else "BASELINE INVALID"
    summary = {
        "stage": "S1-A",
        "outcome": outcome,
        "candidate_manifest_sha256": manifest_hash,
        "model": {
            "repo_id": config["model"]["repo_id"],
            "revision": config["model"]["revision"],
            "processor_revision": config["model"]["processor_revision"],
            "loaded_config_commit_hash": loaded_commit_hash,
        },
        "module_manifest_sha256": module_manifest["manifest_sha256"],
        "overall": overall,
        "by_family": by_family,
        "checks": checks,
        "reused_completed_output_after_postprocessing_failure": reused_completed_output,
        "gpu_elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600,
        "peak_gpu_memory_bytes": peak_memory,
    }
    write_json(processed_directory / "s1_a_baseline_summary.json", summary)
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _baseline_map(
    config: Mapping[str, Any], repository_root: Path
) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(
        _raw_directory(config, repository_root) / "bf16_predictions.jsonl"
    )
    result = {str(row["question_id"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("BF16 output contains duplicate question IDs")
    return result


def run_s1_b(config_path: Path, repository_root: Path) -> dict[str, Any]:
    """Run all-W4 only after S1-A and decide the proxy dynamic-range gate."""
    config = read_json(config_path)
    _validate_config(config)
    processed_directory = _processed_directory(config, repository_root)
    a_summary = read_json(processed_directory / "s1_a_baseline_summary.json")
    if a_summary["outcome"] != "PASS":
        raise RuntimeError("S1-B is forbidden because S1-A did not pass")
    environment = _gpu_environment()
    configure_determinism(int(config["seed"]))
    records, manifest_hash = _load_candidate(config, repository_root)
    baseline = _baseline_map(config, repository_root)
    processor = _load_processor(config, repository_root)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = _load_model(config, repository_root)
    module_manifest = build_module_manifest(model)
    if module_manifest["manifest_sha256"] != a_summary["module_manifest_sha256"]:
        raise RuntimeError("runtime module manifest changed between S1-A and S1-B")
    all_names = all_quantizable_tensor_names(module_manifest)
    audits, _ = quantize_parameters_in_place(
        model,
        all_names,
        QuantizerSpec(**config["quantizer"]["w4"]),
        capture_original=False,
    )
    ordered = whole_image_execution_order(records, seed=int(config["seed"]))
    outputs: list[dict[str, Any]] = []
    previous_image = None
    for index, record in enumerate(ordered, start=1):
        if record.image_id != previous_image:
            print(
                f"S1-B image {len({row['image_id'] for row in outputs}) + 1}/90",
                flush=True,
            )
            previous_image = record.image_id
        outputs.append(
            _run_record(
                model,
                processor,
                record,
                repository_root=repository_root,
                config=config,
                condition="all_w4",
                bf16_row=baseline[record.question_id],
            )
        )
        if index % 60 == 0:
            print(f"S1-B questions {index}/{len(ordered)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    raw_directory = _raw_directory(config, repository_root)
    write_jsonl(raw_directory / "all_w4_predictions.jsonl", outputs)
    overall = aggregate_condition_metrics(outputs)
    by_family = _summaries_by_family(outputs)
    bf16 = a_summary["overall"]
    gate = config["gates"]["proxy"]
    official_point_change = (
        abs(float(overall["official_score"]) - float(bf16["official_score"])) * 100.0
    )
    mean_nll_change = float(overall["mean_delta_nll"])
    no_range = (
        official_point_change < float(gate["no_range_maximum_official_point_change"])
        and abs(mean_nll_change) < float(gate["no_range_maximum_absolute_nll_change"])
        and float(overall["mean_js_divergence"]) < float(gate["no_range_maximum_js"])
        and float(overall["mean_answer_flip"])
        < float(gate["no_range_maximum_answer_flip_rate"])
    )
    bf16_score = float(bf16["official_score"])
    retention = float(overall["official_score"]) / bf16_score
    catastrophic = float(overall["official_score"]) < float(
        gate["collapse_minimum_absolute_official_score"]
    ) or retention < float(gate["collapse_minimum_bf16_retention"])
    if no_range:
        outcome = "PROXY NO DYNAMIC RANGE"
    elif catastrophic:
        outcome = "PROXY TOO AGGRESSIVE"
    else:
        outcome = "PASS"
    formal_hash = None
    if outcome == "PASS":
        candidate_path = raw_directory / "candidate_manifest.jsonl"
        formal_path = raw_directory / "formal_manifest.jsonl"
        if formal_path.exists():
            if sha256_file(formal_path) != manifest_hash:
                raise RuntimeError("existing formal manifest differs from candidate")
        else:
            shutil.copyfile(candidate_path, formal_path)
            shutil.copyfile(
                candidate_path.with_suffix(candidate_path.suffix + ".sha256"),
                formal_path.with_suffix(formal_path.suffix + ".sha256"),
            )
        formal_hash = sha256_file(formal_path)
    summary = {
        "stage": "S1-B",
        "outcome": outcome,
        "candidate_manifest_sha256": manifest_hash,
        "formal_manifest_sha256": formal_hash,
        "environment": environment,
        "quantizer": QuantizerSpec(**config["quantizer"]["w4"]).to_dict(),
        "quantized_tensor_count": len(audits),
        "logical_w4_bytes": sum(audit.logical_packed_bytes for audit in audits),
        "overall": overall,
        "by_family": by_family,
        "bf16_official_score": bf16_score,
        "official_score_drop": bf16_score - float(overall["official_score"]),
        "official_point_change": official_point_change,
        "bf16_score_retention": retention,
        "mean_delta_nll": mean_nll_change,
        "mean_js_divergence": overall["mean_js_divergence"],
        "answer_flip_rate": overall["mean_answer_flip"],
        "bf16_correct_to_w4_wrong_rate": overall[
            "mean_bf16_correct_to_condition_wrong"
        ],
        "no_dynamic_range": no_range,
        "catastrophic": catastrophic,
        "gpu_elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    write_json(processed_directory / "s1_b_proxy_summary.json", summary)
    return summary


def _validate_completed_group(
    path: Path, records: Sequence[S1SampleRecord], condition: str
) -> bool:
    if not path.exists():
        return False
    rows = _read_jsonl(path)
    return (
        len(rows) == len(records)
        and {str(row["question_id"]) for row in rows}
        == {record.question_id for record in records}
        and {str(row["condition"]) for row in rows} == {condition}
    )


def run_s1_sweep(config_path: Path, repository_root: Path) -> dict[str, Any]:
    """Run the coarse 13-group W4-from-BF16 perturbation after both gates pass."""
    config = read_json(config_path)
    _validate_config(config)
    processed_directory = _processed_directory(config, repository_root)
    b_summary = read_json(processed_directory / "s1_b_proxy_summary.json")
    if b_summary["outcome"] != "PASS":
        raise RuntimeError("formal S1 sweep is forbidden because S1-B did not pass")
    raw_directory = _raw_directory(config, repository_root)
    formal_path = raw_directory / "formal_manifest.jsonl"
    records = _load_records(formal_path)
    if sha256_file(formal_path) != b_summary["formal_manifest_sha256"]:
        raise RuntimeError("formal S1 manifest changed after S1-B")
    baseline = _baseline_map(config, repository_root)
    _gpu_environment()
    configure_determinism(int(config["seed"]))
    processor = _load_processor(config, repository_root)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = _load_model(config, repository_root)
    module_manifest = build_module_manifest(model)
    a_summary = read_json(processed_directory / "s1_a_baseline_summary.json")
    if module_manifest["manifest_sha256"] != a_summary["module_manifest_sha256"]:
        raise RuntimeError("runtime module manifest changed before formal sweep")
    ordered = whole_image_execution_order(records, seed=int(config["seed"]))
    completed: list[dict[str, Any]] = []
    for group_index, group in enumerate(module_manifest["groups"], start=1):
        group_id = str(group["group_id"])
        condition = f"single_group_w4:{group_id}"
        output_path = (
            raw_directory / f"single_group__{group_id.replace('.', '__')}.jsonl"
        )
        if _validate_completed_group(output_path, records, condition):
            outputs = _read_jsonl(output_path)
            completed.append(
                {
                    "group_id": group_id,
                    "reused": True,
                    "summary": aggregate_condition_metrics(outputs),
                }
            )
            print(f"S1 sweep reused {group_index}/13 {group_id}", flush=True)
            continue
        tensor_names = group_tensor_names(module_manifest, group_id)
        before_hashes = parameter_hashes(model, tensor_names)
        _, originals = quantize_parameters_in_place(
            model,
            tensor_names,
            QuantizerSpec(**config["quantizer"]["w4"]),
            capture_original=True,
        )
        outputs = []
        previous_image = None
        for index, record in enumerate(ordered, start=1):
            if record.image_id != previous_image:
                print(
                    f"S1 sweep {group_index}/13 {group_id}: image "
                    f"{len({row['image_id'] for row in outputs}) + 1}/90",
                    flush=True,
                )
                previous_image = record.image_id
            outputs.append(
                _run_record(
                    model,
                    processor,
                    record,
                    repository_root=repository_root,
                    config=config,
                    condition=condition,
                    bf16_row=baseline[record.question_id],
                )
            )
            if index % 120 == 0:
                print(
                    f"S1 sweep {group_index}/13 questions {index}/{len(ordered)}",
                    flush=True,
                )
        write_jsonl(output_path, outputs)
        restore_parameters(model, originals)
        if parameter_hashes(model, tensor_names) != before_hashes:
            raise AssertionError(f"BF16 restore failed after group {group_id}")
        completed.append(
            {
                "group_id": group_id,
                "reused": False,
                "summary": aggregate_condition_metrics(outputs),
            }
        )
        prior_hours = float(
            read_json(processed_directory / "s1_a_baseline_summary.json")["gpu_hours"]
        ) + float(b_summary["gpu_hours"])
        current_hours = (time.perf_counter() - started) / 3600
        if prior_hours + current_hours >= 14:
            raise RuntimeError("C01-S1 reached the 14 A800 GPU-hour cap")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    summary = {
        "stage": "S1 coarse 13-group sweep",
        "outcome": "PASS",
        "formal_manifest_sha256": sha256_file(formal_path),
        "module_manifest_sha256": module_manifest["manifest_sha256"],
        "group_count": len(completed),
        "groups": completed,
        "gpu_elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    write_json(processed_directory / "s1_sweep_summary.json", summary)
    return summary


def _sensitivity_matrix(
    raw_directory: Path, module_manifest: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, float]] = defaultdict(dict)
    for group in module_manifest["groups"]:
        group_id = str(group["group_id"])
        path = raw_directory / f"single_group__{group_id.replace('.', '__')}.jsonl"
        for row in _read_jsonl(path):
            matrix[str(row["question_id"])][group_id] = float(row["delta_nll"])
    expected_groups = {str(group["group_id"]) for group in module_manifest["groups"]}
    if any(set(values) != expected_groups for values in matrix.values()):
        raise RuntimeError("coarse sensitivity matrix has incomplete group rows")
    return dict(matrix)


def _profile_output_valid(path: Path, question_ids: set[str], condition: str) -> bool:
    if not path.exists():
        return False
    rows = _read_jsonl(path)
    return (
        len(rows) == len(question_ids)
        and {str(row["question_id"]) for row in rows} == question_ids
        and {str(row["condition"]) for row in rows} == {condition}
    )


def run_s1_profiles(config_path: Path, repository_root: Path) -> dict[str, Any]:
    """Execute held-out equal-budget multi-group profiles from the all-W4 state."""
    config = read_json(config_path)
    _validate_config(config)
    raw_directory = _raw_directory(config, repository_root)
    processed_directory = _processed_directory(config, repository_root)
    sweep_summary = read_json(processed_directory / "s1_sweep_summary.json")
    if sweep_summary["outcome"] != "PASS" or int(sweep_summary["group_count"]) != 13:
        raise RuntimeError("profile execution is forbidden before the 13-group sweep")
    formal_path = raw_directory / "formal_manifest.jsonl"
    records = _load_records(formal_path)
    module_manifest = read_json(raw_directory / "module_manifest.json")
    sensitivities = _sensitivity_matrix(raw_directory, module_manifest)
    execution_images = select_execution_images(
        [record.image_id for record in records],
        image_count=int(config["profiles"]["execution_image_count"]),
        seed=int(config["profiles"]["seed"]),
    )
    definitions = build_profile_definitions(
        [record.to_dict() for record in records],
        sensitivities,
        module_manifest,
        execution_image_ids=execution_images,
        budget_fractions=[
            float(item) for item in config["profiles"]["budget_fractions"]
        ],
        seed=int(config["profiles"]["seed"]),
    )
    write_jsonl(raw_directory / "profile_definitions.jsonl", definitions)
    signature_assignments: dict[str, dict[str, str]] = {}
    signature_questions: dict[str, set[str]] = defaultdict(set)
    for definition in definitions:
        signature = str(definition["profile_signature"])
        assignments = {
            str(group_id): str(precision)
            for group_id, precision in definition["assignments"].items()
        }
        if (
            signature in signature_assignments
            and signature_assignments[signature] != assignments
        ):
            raise RuntimeError("profile signature collision")
        signature_assignments[signature] = assignments
        signature_questions[signature].add(str(definition["question_id"]))

    baseline = _baseline_map(config, repository_root)
    all_w4_rows = {
        str(row["question_id"]): row
        for row in _read_jsonl(raw_directory / "all_w4_predictions.jsonl")
    }
    record_map = {record.question_id: record for record in records}
    profile_directory = raw_directory / "profiles"
    profile_directory.mkdir(parents=True, exist_ok=True)
    pending = [
        signature
        for signature in sorted(signature_assignments)
        if not _profile_output_valid(
            profile_directory / f"profile__{signature}.jsonl",
            signature_questions[signature],
            f"restoration_profile:{signature}",
        )
    ]
    runtime_path = raw_directory / "profile_runtime.json"
    runtime_state = (
        read_json(runtime_path)
        if runtime_path.exists()
        else {"accumulated_gpu_seconds": 0.0, "peak_gpu_memory_bytes": 0}
    )
    elapsed = float(runtime_state["accumulated_gpu_seconds"])
    peak_memory = int(runtime_state["peak_gpu_memory_bytes"])
    quantized_tensor_count = len(all_quantizable_tensor_names(module_manifest))
    if pending:
        _gpu_environment()
        configure_determinism(int(config["seed"]))
        processor = _load_processor(config, repository_root)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = _load_model(config, repository_root)
        runtime_manifest = build_module_manifest(model)
        if runtime_manifest["manifest_sha256"] != module_manifest["manifest_sha256"]:
            raise RuntimeError(
                "runtime module manifest changed before profile execution"
            )
        all_names = all_quantizable_tensor_names(runtime_manifest)
        _, originals = quantize_parameters_in_place(
            model,
            all_names,
            QuantizerSpec(**config["quantizer"]["w4"]),
            capture_original=True,
        )
        w8_spec = W8RestorationSpec(**config["quantizer"]["w8"])
        w8_spec.validate()
        for index, signature in enumerate(pending, start=1):
            assignments = signature_assignments[signature]
            logical_bytes = profile_logical_bytes(runtime_manifest, assignments)
            apply_restoration_profile(
                model,
                originals,
                runtime_manifest,
                assignments,
                w8_spec=w8_spec,
            )
            outputs: list[dict[str, Any]] = []
            for question_id in sorted(signature_questions[signature]):
                record = record_map[question_id]
                output = _run_record(
                    model,
                    processor,
                    record,
                    repository_root=repository_root,
                    config=config,
                    condition=f"restoration_profile:{signature}",
                    bf16_row=baseline[question_id],
                )
                output["profile_signature"] = signature
                output["profile_logical_bytes"] = logical_bytes
                outputs.append(output)
            write_jsonl(profile_directory / f"profile__{signature}.jsonl", outputs)
            reset_profile_to_w4(model, originals, runtime_manifest, assignments)
            print(
                f"S1 profiles {index}/{len(pending)} signature={signature} "
                f"questions={len(outputs)}",
                flush=True,
            )
            elapsed = float(runtime_state["accumulated_gpu_seconds"]) + (
                time.perf_counter() - started
            )
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())
            write_json(
                runtime_path,
                {
                    "accumulated_gpu_seconds": elapsed,
                    "peak_gpu_memory_bytes": peak_memory,
                    "completed_profile_signatures": index,
                },
            )
            prior_hours = sum(
                float(read_json(processed_directory / name)["gpu_hours"])
                for name in (
                    "s1_a_baseline_summary.json",
                    "s1_b_proxy_summary.json",
                    "s1_sweep_summary.json",
                )
            )
            profile_hours = (
                float(runtime_state["accumulated_gpu_seconds"])
                + time.perf_counter()
                - started
            ) / 3600
            if prior_hours + profile_hours >= 14:
                raise RuntimeError("C01-S1 reached the 14 A800 GPU-hour cap")
        torch.cuda.synchronize()
        elapsed = float(runtime_state["accumulated_gpu_seconds"]) + (
            time.perf_counter() - started
        )
        peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())

    execution_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for signature in sorted(signature_assignments):
        path = profile_directory / f"profile__{signature}.jsonl"
        if not _profile_output_valid(
            path,
            signature_questions[signature],
            f"restoration_profile:{signature}",
        ):
            raise RuntimeError(f"profile output is incomplete: {signature}")
        for row in _read_jsonl(path):
            execution_rows[(str(row["question_id"]), signature)] = row
    aggregates = aggregate_executed_profiles(
        definitions, execution_rows, baseline, all_w4_rows
    )
    summary = {
        "stage": "S1 held-out executed profiles",
        "outcome": "PASS",
        "execution_image_ids": list(execution_images),
        "execution_image_count": len(execution_images),
        "execution_question_count": len(
            {str(definition["question_id"]) for definition in definitions}
        ),
        "definition_count": len(definitions),
        "unique_profile_count": len(signature_assignments),
        "unique_question_profile_execution_count": len(execution_rows),
        "quantized_tensor_count": quantized_tensor_count,
        "aggregates": aggregates,
        "gpu_elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600,
        "peak_gpu_memory_bytes": peak_memory,
    }
    write_json(processed_directory / "s1_profile_summary.json", summary)
    return summary
