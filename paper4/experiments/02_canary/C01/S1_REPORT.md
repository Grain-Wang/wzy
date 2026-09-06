# C01-S1 Report

## Outcome

**INCONCLUSIVE**

## S1-A Baseline Validity

Outcome: **PASS**. BF16 official accuracy was 60.00%, mean/median answer-only NLL were 0.62331/0.25513, and empty/malformed rates were 0.00%/0.00%. All locked checks passed: `{'empty_rate': True, 'family_dynamic_range': True, 'finite_nll_rate': True, 'malformed_rate': True, 'overall_official_score': True}`.

The initial GPU job completed all 540 inference cases but exited during aggregation because unavailable baseline JS values were represented as `None`. The raw output was complete and reused after a tested post-processing fix; no BF16 inference was rerun and no sample changed.

## S1-B W4 Dynamic Range

Outcome: **PASS**. All-W4 official accuracy was 59.26% (drop 0.74 point), mean ΔNLL was 0.032510, mean JS was 0.012857, and answer-flip rate was 14.07%. This was measurable and non-catastrophic; no proxy correction was used.

## Model and Environment

- Checkpoint: `Qwen/Qwen2-VL-2B-Instruct` at `895c3a49bc3fa70a340399125c650a463535e71c`
- Processor revision: `895c3a49bc3fa70a340399125c650a463535e71c`
- Python / torch / transformers / CUDA: `3.12.13` / `2.5.0+cu121` / `4.55.2` / `12.1`
- GPU: `NVIDIA A800 80GB PCIe`; BF16 eager attention; deterministic algorithms; no vision-feature cache
- Shared-device load was present, so no latency or throughput claim is made.

## Frozen Manifest

- image count: 90
- question count: 540
- family counts: `{'fine_grained_recognition': 170, 'global_perception': 30, 'reasoning': 170, 'spatial_relation': 170}`
- image×family repeated cells: 270
- manifest SHA256: `6a7d8237438bd1764cb8a9341fdafad0e5d43dbbba448962bdd0f532993a63b0`
- selection used no model output and applied no post-hoc correctness filter

## Quantizer

- W4: symmetric RTN, group size 128, qmin/qmax -7/7, FP32 scale, no percentile clipping; activations and KV BF16
- W8 restoration: symmetric RTN, group size 128, qmin/qmax -127/127, FP32 scale, no percentile clipping
- BF16 restoration used the exact captured checkpoint tensors

## Module Groups

Runtime manifest SHA256: `db080e473fb6f9fa4204b1372ace0a5793644a71f41566a99b1847f696e705f6`; 13 coarse non-KV groups.

- `bridge.merger`: 2 tensors, W4=18,104,320 B, W8=35,143,680 B, BF16=68,157,440 B
- `decoder.q1.attention`: 28 tensors, W4=20,471,808 B, W8=39,739,392 B, BF16=77,070,336 B
- `decoder.q1.mlp`: 21 tensors, W4=153,538,560 B, W8=298,045,440 B, BF16=578,027,520 B
- `decoder.q2.attention`: 28 tensors, W4=20,471,808 B, W8=39,739,392 B, BF16=77,070,336 B
- `decoder.q2.mlp`: 21 tensors, W4=153,538,560 B, W8=298,045,440 B, BF16=578,027,520 B
- `decoder.q3.attention`: 28 tensors, W4=20,471,808 B, W8=39,739,392 B, BF16=77,070,336 B
- `decoder.q3.mlp`: 21 tensors, W4=153,538,560 B, W8=298,045,440 B, BF16=578,027,520 B
- `decoder.q4.attention`: 28 tensors, W4=20,471,808 B, W8=39,739,392 B, BF16=77,070,336 B
- `decoder.q4.mlp`: 21 tensors, W4=153,538,560 B, W8=298,045,440 B, BF16=578,027,520 B
- `vision.q1`: 32 tensors, W4=83,558,400 B, W8=162,201,600 B, BF16=314,572,800 B
- `vision.q2`: 32 tensors, W4=83,558,400 B, W8=162,201,600 B, BF16=314,572,800 B
- `vision.q3`: 32 tensors, W4=83,558,400 B, W8=162,201,600 B, BF16=314,572,800 B
- `vision.q4`: 32 tensors, W4=83,558,400 B, W8=162,201,600 B, BF16=314,572,800 B

## BF16 Baseline

Official score: 60.00%; normalized-exact audit: 60.00%; mean NLL: 0.623309.

## All-W4 Baseline

Official score: 59.26%; mean ΔNLL: 0.032510; mean JS: 0.012857; BF16-correct→W4-wrong rate: 4.81%.

## Interaction Analysis

- pooled cross-validated partial R²: -0.603177
- image-bootstrap 95% CI: [-0.671533, -0.523433]
- within-image permutation p: 0.051
- standardized permutation effect size: 1.5682
- weighting: pooled SSE across all 13 groups; five image-grouped outer folds with leave-one-query-out cell support
- JS corroboration partial R²: -0.540628, 95% CI [-0.679444, -0.352256], permutation p=0.016

## Same-Image Sensitivity Structure

- same-family pairs: n=270, cosine distance 0.90654 (95% CI [0.86378, 0.94875])
- cross-family pairs: n=1080, cosine distance 0.93513 (95% CI [0.91065, 0.95888])
- ranking stability: median top-3 Jaccard 0.500, lower 95% 0.200, exact identity rate 38.85%

## Fixed-Budget Restoration

All profiles use exact manifest byte accounting and the locked deterministic no-over-budget packing rule. These are executed multi-group results on 20 held-out images / 120 questions, not sums of single-group gains.

### +5%

| Policy | Official | NLL | JS | NLL recovery | Added bytes | Non-additivity |
|---|---:|---:|---:|---:|---:|---:|
| global_static | 66.67% | 0.51914 | 0.010627 | 51.46% | 38,535,168 | -0.00747 |
| leave_one_query_out_per_image | 66.67% | 0.53098 | 0.013056 | 12.93% | 35,918,097 | -0.01367 |
| per_image_family | 67.50% | 0.52730 | 0.012791 | 24.92% | 34,563,413 | -0.01145 |
| per_query_family | 66.67% | 0.52796 | 0.012212 | 22.77% | 37,829,563 | -0.01657 |
| per_query_oracle | 69.17% | 0.48709 | 0.013103 | 155.73% | 34,563,413 | +0.00955 |
| random_equal_byte | 67.50% | 0.52630 | 0.013580 | 28.15% | 38,012,381 | -0.00897 |

### +10%

| Policy | Official | NLL | JS | NLL recovery | Added bytes | Non-additivity |
|---|---:|---:|---:|---:|---:|---:|
| global_static | 65.00% | 0.52007 | 0.010598 | 48.42% | 92,905,472 | -0.02847 |
| leave_one_query_out_per_image | 66.67% | 0.51872 | 0.011098 | 52.83% | 83,021,755 | -0.01857 |
| per_image_family | 68.33% | 0.52233 | 0.011659 | 41.09% | 82,323,114 | -0.02265 |
| per_query_family | 65.83% | 0.52598 | 0.011976 | 29.20% | 87,509,674 | -0.03099 |
| per_query_oracle | 68.33% | 0.47656 | 0.012215 | 190.01% | 82,323,114 | -0.01434 |
| random_equal_byte | 67.50% | 0.52880 | 0.011579 | 20.02% | 92,403,848 | -0.03205 |

### +20%

| Policy | Official | NLL | JS | NLL recovery | Added bytes | Non-additivity |
|---|---:|---:|---:|---:|---:|---:|
| global_static | 65.00% | 0.53247 | 0.011039 | 8.08% | 182,517,760 | -0.06492 |
| leave_one_query_out_per_image | 68.33% | 0.52425 | 0.010974 | 34.84% | 182,011,016 | -0.03725 |
| per_image_family | 67.50% | 0.51399 | 0.010314 | 68.20% | 182,531,822 | -0.03017 |
| per_query_family | 67.50% | 0.51481 | 0.009635 | 65.53% | 188,516,215 | -0.03816 |
| per_query_oracle | 72.50% | 0.44293 | 0.010881 | 299.40% | 182,531,822 | -0.01447 |
| random_equal_byte | 67.50% | 0.52882 | 0.009351 | 19.95% | 203,299,089 | -0.06243 |

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

Best locked budget: +20%; oracle minus `leave_one_query_out_per_image` = +4.17 points, 95% CI [+0.83, +8.33].

## Relative NLL Recovery Headroom

Oracle minus control = +264.57 percentage points, 95% CI [+101.81, +2442.28].

## Executed Profile Non-Additivity

`non_additivity_error` in each fixed-budget table is actual all-W4→profile NLL gain minus the candidate additive estimate. It is reported explicitly and is never substituted for executed performance.

## Query-Family Replication

- fine_grained_recognition: n=38, images=19, official headroom=+5.26 points, NLL-recovery headroom=+162.26 pp, directionally consistent=True
- global_perception: n=8, images=4, official headroom=+0.00 points, NLL-recovery headroom=+129.52 pp, directionally consistent=True
- reasoning: n=38, images=19, official headroom=+2.63 points, NLL-recovery headroom=+149.64 pp, directionally consistent=True
- spatial_relation: n=36, images=18, official headroom=+5.56 points, NLL-recovery headroom=+1803.87 pp, directionally consistent=True

Adequate directionally consistent families: 3.

## Confound Checks

`{'absolute_sensitivity_pearson_correlations': {'answer_token_length': 0.11826603595876242, 'question_token_length': -0.16831673230369976, 'visual_token_count': 0.0037850438565866502}, 'prompt_hash_count': 1, 'samples_filtered_by_correctness': False, 'taxonomy_uses_model_output': False, 'vision_cache_enabled': False}`

## Failure Cases

- BF16-correct / W4-wrong recoverable tail: 26 (4.81%), family counts `{'fine_grained_recognition': 6, 'reasoning': 10, 'spatial_relation': 10}`.
- The S1-A aggregation bug was fixed and regression-tested; it did not affect saved inference values.
- A NumPy-scalar JSON serialization bug was fixed and regression-tested after the first completed analysis pass; it did not affect any statistic or model output.
- Random-control priority scores were initially mislabeled as additive NLL estimates. The report now recomputes every profile's additive estimate from that query's measured single-group gains; profile selection, executed outputs, and gate statistics were unchanged.
- Free-form latency is not evaluated because the diagnostic reconstruction does not compress physical parameter storage and the A800 was shared.

## GPU Time

- S1-A: 0.1019 h
- S1-B: 0.1032 h
- 13-group sweep: 1.2399 h
- executed profiles: 0.3640 h
- total successful GPU interval: 1.8090 h
- retries: 0 GPU inference retries; analysis was repeated once after a serialization-only fix
- failed jobs: 2 post-processing exits (S1-A aggregation and final JSON serialization), with all completed inference artifacts reused
- peak allocated GPU memory recorded across completed stages: 8.451 GiB

## Storage

- raw S1 artifacts: 12,252,348 bytes
- processed S1 artifacts: 53,537 bytes
- tables: 5,847 bytes
- figures: 202,225 bytes
- ignored materialized images: 43,597,971 bytes

## Scientific Interpretation

The result satisfies neither the conjunctive STRONG PASS gate nor the robust NEGATIVE gate. The valid proxy and frozen outputs permit one bounded, preregistered sample expansion, but no router or S2 execution is authorized.

The strongest contrary consideration is that coarse W4 perturbations may not predict non-additive all-W4 restoration behavior; the held-out executed profiles and reported non-additivity error directly audit this concern.

## Gate Decision

- interaction gate: False
- equal-byte profile gate: True
- query-family replication gate: True
- ranking-stability gate: True
- robust-negative rule: False
- final S1 outcome: **INCONCLUSIVE**

## Next Action

One bounded sample expansion only; do not apply a proxy correction or run S2.
