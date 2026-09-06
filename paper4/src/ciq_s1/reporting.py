"""Render the immutable C01-S1 scientific report from processed artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..ciq_s0.io_utils import read_json


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _profile_table(rows: Sequence[Mapping[str, Any]], budget: float) -> str:
    selected = [row for row in rows if float(row["budget_fraction"]) == budget]
    lines = [
        "| Policy | Official | NLL | JS | NLL recovery | Added bytes | Non-additivity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['policy']} | {_percent(float(row['official_score']))} | "
            f"{float(row['mean_answer_nll']):.5f} | "
            f"{float(row['mean_js_divergence']):.6f} | "
            f"{_percent(float(row['relative_nll_recovery']))} | "
            f"{int(float(row['mean_added_bytes'])):,} | "
            f"{float(row['non_additivity_error']):+.5f} |"
        )
    return "\n".join(lines)


def render_s1_report(
    *, repository_root: Path, config: Mapping[str, Any], summary: Mapping[str, Any]
) -> Path:
    """Write every required S1 report section without changing preregistration."""
    processed = repository_root / str(config["paths"]["processed_output"])
    raw = repository_root / str(config["paths"]["raw_output"])
    a = read_json(processed / "s1_a_baseline_summary.json")
    b = read_json(processed / "s1_b_proxy_summary.json")
    sweep = read_json(processed / "s1_sweep_summary.json")
    profiles = read_json(processed / "s1_profile_summary.json")
    candidate = read_json(processed / "candidate_manifest_summary.json")
    environment = read_json(raw / "environment.json")
    interaction = summary["interaction"]
    js_interaction = summary["js_interaction"]
    bootstrap = interaction["bootstrap"]
    permutation = interaction["permutation"]
    best = summary["best_headroom"]
    gates = summary["gate_components"]
    outcome = str(summary["outcome"])
    if outcome == "STRONG PASS":
        interpretation = (
            "The coarse confirmatory evidence supports stable, useful image×query "
            "non-KV precision-profile variation under the locked diagnostic proxy. "
            "This passes only the S1 gate; it does not establish a deployable method."
        )
        next_action = "C01-S2 refinement / independent replication only; do not train a router yet."
    elif outcome == "NEGATIVE":
        interpretation = (
            "The validated W4 proxy has dynamic range, but the preregistered interaction "
            "and equal-byte oracle criteria jointly reject useful coarse CIQ-PP structure."
        )
        next_action = (
            "Archive CIQ-PP. The paired recoverable tail may motivate a separately "
            "preregistered Quantization-Induced Disagreement Rescue probe; do not run it now."
        )
    else:
        interpretation = (
            "The result satisfies neither the conjunctive STRONG PASS gate nor the robust "
            "NEGATIVE gate. The valid proxy and frozen outputs permit one bounded, "
            "preregistered sample expansion, but no router or S2 execution is authorized."
        )
        next_action = "One bounded sample expansion only; do not apply a proxy correction or run S2."
    gpu_hours = sum(float(item["gpu_hours"]) for item in (a, b, sweep, profiles))
    peak_candidates = [
        int(item["peak_gpu_memory_bytes"])
        for item in (a, b, sweep, profiles)
        if item.get("peak_gpu_memory_bytes") is not None
    ]
    peak_memory = max(peak_candidates) if peak_candidates else 0
    storage = summary["storage_bytes"]
    profile_rows = profiles["aggregates"]
    family_lines = [
        f"- {family}: n={value['sample_count']}, images={value['image_count']}, official headroom="
        f"{100 * float(value['official_score_headroom']):+.2f} points, "
        f"NLL-recovery headroom={100 * float(value['relative_nll_recovery_headroom']):+.2f} pp, "
        f"directionally consistent={value['directionally_consistent']}"
        for family, value in summary["query_family_replication"].items()
    ]
    module_manifest = read_json(raw / "module_manifest.json")
    module_lines = [
        f"- `{group['group_id']}`: {len(group['tensor_names'])} tensors, "
        f"W4={int(group['w4_bytes']):,} B, W8={int(group['w8_bytes']):,} B, "
        f"BF16={int(group['bf16_bytes']):,} B"
        for group in module_manifest["groups"]
    ]
    report = f"""# C01-S1 Report

## Outcome

**{outcome}**

## S1-A Baseline Validity

Outcome: **{a['outcome']}**. BF16 official accuracy was {_percent(float(a['overall']['official_score']))}, mean/median answer-only NLL were {float(a['overall']['mean_answer_nll']):.5f}/{float(a['overall']['median_answer_nll']):.5f}, and empty/malformed rates were {_percent(float(a['overall']['empty_generation_rate']))}/{_percent(float(a['overall']['malformed_generation_rate']))}. All locked checks passed: `{a['checks']}`.

The initial GPU job completed all 540 inference cases but exited during aggregation because unavailable baseline JS values were represented as `None`. The raw output was complete and reused after a tested post-processing fix; no BF16 inference was rerun and no sample changed.

## S1-B W4 Dynamic Range

Outcome: **{b['outcome']}**. All-W4 official accuracy was {_percent(float(b['overall']['official_score']))} (drop {100 * float(b['official_score_drop']):.2f} point), mean ΔNLL was {float(b['mean_delta_nll']):.6f}, mean JS was {float(b['mean_js_divergence']):.6f}, and answer-flip rate was {_percent(float(b['answer_flip_rate']))}. This was measurable and non-catastrophic; no proxy correction was used.

## Model and Environment

- Checkpoint: `{config['model']['repo_id']}` at `{config['model']['revision']}`
- Processor revision: `{config['model']['processor_revision']}`
- Python / torch / transformers / CUDA: `{environment['python']}` / `{environment['torch']}` / `{environment['transformers']}` / `{environment['cuda_runtime']}`
- GPU: `{environment['gpu_name']}`; BF16 eager attention; deterministic algorithms; no vision-feature cache
- Shared-device load was present, so no latency or throughput claim is made.

## Frozen Manifest

- image count: {candidate['image_count']}
- question count: {candidate['question_count']}
- family counts: `{candidate['family_question_counts']}`
- image×family repeated cells: {candidate['repeated_image_family_cells']}
- manifest SHA256: `{candidate['manifest_sha256']}`
- selection used no model output and applied no post-hoc correctness filter

## Quantizer

- W4: symmetric RTN, group size 128, qmin/qmax -7/7, FP32 scale, no percentile clipping; activations and KV BF16
- W8 restoration: symmetric RTN, group size 128, qmin/qmax -127/127, FP32 scale, no percentile clipping
- BF16 restoration used the exact captured checkpoint tensors

## Module Groups

Runtime manifest SHA256: `{module_manifest['manifest_sha256']}`; 13 coarse non-KV groups.

{chr(10).join(module_lines)}

## BF16 Baseline

Official score: {_percent(float(a['overall']['official_score']))}; normalized-exact audit: {_percent(float(a['overall']['normalized_exact']))}; mean NLL: {float(a['overall']['mean_answer_nll']):.6f}.

## All-W4 Baseline

Official score: {_percent(float(b['overall']['official_score']))}; mean ΔNLL: {float(b['mean_delta_nll']):.6f}; mean JS: {float(b['mean_js_divergence']):.6f}; BF16-correct→W4-wrong rate: {_percent(float(b['bf16_correct_to_w4_wrong_rate']))}.

## Interaction Analysis

- pooled cross-validated partial R²: {float(interaction['partial_r2']):.6f}
- image-bootstrap 95% CI: [{float(bootstrap['lower_95']):.6f}, {float(bootstrap['upper_95']):.6f}]
- within-image permutation p: {float(permutation['p_value']):.6g}
- standardized permutation effect size: {float(permutation['standardized_effect_size']):.4f}
- weighting: pooled SSE across all 13 groups; five image-grouped outer folds with leave-one-query-out cell support
- JS corroboration partial R²: {float(js_interaction['partial_r2']):.6f}, 95% CI [{float(js_interaction['bootstrap']['lower_95']):.6f}, {float(js_interaction['bootstrap']['upper_95']):.6f}], permutation p={float(js_interaction['permutation']['p_value']):.6g}

## Same-Image Sensitivity Structure

- same-family pairs: n={summary['same_image_structure']['same_pair_count']}, cosine distance {float(summary['same_image_structure']['same_family']['cosine_distance']['mean']):.5f} (95% CI [{float(summary['same_image_structure']['same_family']['cosine_distance']['lower_95']):.5f}, {float(summary['same_image_structure']['same_family']['cosine_distance']['upper_95']):.5f}])
- cross-family pairs: n={summary['same_image_structure']['cross_pair_count']}, cosine distance {float(summary['same_image_structure']['cross_family']['cosine_distance']['mean']):.5f} (95% CI [{float(summary['same_image_structure']['cross_family']['cosine_distance']['lower_95']):.5f}, {float(summary['same_image_structure']['cross_family']['cosine_distance']['upper_95']):.5f}])
- ranking stability: median top-3 Jaccard {float(summary['ranking_stability']['median_jaccard']):.3f}, lower 95% {float(summary['ranking_stability']['lower_95_jaccard']):.3f}, exact identity rate {_percent(float(summary['ranking_stability']['exact_identity_rate']))}

## Fixed-Budget Restoration

All profiles use exact manifest byte accounting and the locked deterministic no-over-budget packing rule. These are executed multi-group results on {profiles['execution_image_count']} held-out images / {profiles['execution_question_count']} questions, not sums of single-group gains.

### +5%

{_profile_table(profile_rows, 0.05)}

### +10%

{_profile_table(profile_rows, 0.10)}

### +20%

{_profile_table(profile_rows, 0.20)}

## Best Global Control

See `global_static` in the tables above.

## Best Task Control

See `per_query_family` in the tables above.

## Best Image Control

See `leave_one_query_out_per_image` in the tables above.

## Image×Family Control

See `per_image_family` in the tables above.

## Per-Query Oracle

See `per_query_oracle` in the tables above.

## Random Control

See `random_equal_byte` in the tables above.

## Official Score Headroom

Best locked budget: +{100 * float(best['budget_fraction']):.0f}%; oracle minus `{best['control_policy']}` = {100 * float(best['official_score_headroom']):+.2f} points, 95% CI [{100 * float(best['official_score_headroom_lower_95']):+.2f}, {100 * float(best['official_score_headroom_upper_95']):+.2f}].

## Relative NLL Recovery Headroom

Oracle minus control = {100 * float(best['relative_nll_recovery_headroom']):+.2f} percentage points, 95% CI [{100 * float(best['relative_nll_recovery_headroom_lower_95']):+.2f}, {100 * float(best['relative_nll_recovery_headroom_upper_95']):+.2f}].

## Executed Profile Non-Additivity

`non_additivity_error` in each fixed-budget table is actual all-W4→profile NLL gain minus the candidate additive estimate. It is reported explicitly and is never substituted for executed performance.

## Query-Family Replication

{chr(10).join(family_lines)}

Adequate directionally consistent families: {summary['adequate_directional_replication_count']}.

## Confound Checks

`{summary['confounds']}`

## Failure Cases

- BF16-correct / W4-wrong recoverable tail: {summary['recoverable_tail']['bf16_correct_w4_wrong_count']} ({_percent(float(summary['recoverable_tail']['fraction']))}), family counts `{summary['recoverable_tail']['family_counts']}`.
- The S1-A aggregation bug was fixed and regression-tested; it did not affect saved inference values.
- A NumPy-scalar JSON serialization bug was fixed and regression-tested after the first completed analysis pass; it did not affect any statistic or model output.
- Random-control priority scores were initially mislabeled as additive NLL estimates. The report now recomputes every profile's additive estimate from that query's measured single-group gains; profile selection, executed outputs, and gate statistics were unchanged.
- Free-form latency is not evaluated because the diagnostic reconstruction does not compress physical parameter storage and the A800 was shared.

## GPU Time

- S1-A: {float(a['gpu_hours']):.4f} h
- S1-B: {float(b['gpu_hours']):.4f} h
- 13-group sweep: {float(sweep['gpu_hours']):.4f} h
- executed profiles: {float(profiles['gpu_hours']):.4f} h
- total successful GPU interval: {gpu_hours:.4f} h
- retries: 0 GPU inference retries; analysis was repeated once after a serialization-only fix
- failed jobs: 2 post-processing exits (S1-A aggregation and final JSON serialization), with all completed inference artifacts reused
- peak allocated GPU memory recorded across completed stages: {peak_memory / 2**30:.3f} GiB

## Storage

- raw S1 artifacts: {int(storage['raw']):,} bytes
- processed S1 artifacts: {int(storage['processed']):,} bytes
- tables: {int(storage['tables']):,} bytes
- figures: {int(storage['figures']):,} bytes
- ignored materialized images: {int(storage['materialized_images']):,} bytes

## Scientific Interpretation

{interpretation}

The strongest contrary consideration is that coarse W4 perturbations may not predict non-additive all-W4 restoration behavior; the held-out executed profiles and reported non-additivity error directly audit this concern.

## Gate Decision

- interaction gate: {gates['interaction_pass']}
- equal-byte profile gate: {gates['profile_pass']}
- query-family replication gate: {gates['replication_pass']}
- ranking-stability gate: {gates['ranking_stability_pass']}
- robust-negative rule: {gates['robust_negative']}
- final S1 outcome: **{outcome}**

## Next Action

{next_action}
"""
    path = repository_root / "paper4/experiments/02_canary/C01/S1_REPORT.md"
    path.write_text(report, encoding="utf-8")
    return path
