# Current Research Status

Last updated: 2026-09-06

## Main Direction

**CIQ-PP: Counterfactual Image–Query Non-KV Precision Profiles for Hardware-Realistic VLM Inference.**

The broad phrase “query-aware dynamic quantization” is not the contribution. The selected opportunity asks whether changing only the query for a fixed image changes the useful non-KV VLM sensitivity profile, and—only if that fact is validated—whether the interaction can support at most 3–4 resident, prepacked, hardware-native profiles.

Decision analysis: [`analysis/idea_selection.md`](analysis/idea_selection.md)
Main-direction specification: [`ideas/main_direction.md`](ideas/main_direction.md)
Literature audit: [`literature/surveys/query_aware_vlm_quantization_literature_review.md`](literature/surveys/query_aware_vlm_quantization_literature_review.md)

## Current Stage

**Canary Validation.** Literature, gap, collision, and idea-selection gates are complete. C01-S0 has been implemented and passed all registered integrity checks on 8 GQA images / 32 questions. C01-S1 has **not** been run. The project remains a Research Opportunity and has **not** reached the Paper Candidate gate.

`paper4/results/` now contains only S0 infrastructure evidence. S0 does not test the image-query sensitivity hypothesis, oracle headroom, quality gain, or hardware benefit; current scientific support for the hypothesis remains literature-based only.

## Core Hypothesis

For a fixed image and model, changing the query creates a stable image-query interaction residual in the W4A16 sensitivity of the vision encoder, multimodal projector, language-model attention, and language-model MLP. The effect is useful only if it changes the best fixed-budget protected-module profile beyond global-static, per-task, per-image, query-only, and sequence-length controls.

## Strongest Supporting Evidence

- Query-guided visual-token methods such as QuietPrune, IVTP, LVPruning, SparseVLM, and MADTP show that visual relevance depends on the current language query.
- Q-VLM, MBQ, VLMQ, TLQ, VLM-PTQ, MABA, and VVSQ show that VLM quantization sensitivity is heterogeneous across modalities, tokens, layers, and modules.
- QAQ, TAQ, DP-LLM, PMPD, MixKVQ, and related dynamic-precision work show that query-, task-, input-, phase-, and token-conditioned precision decisions are feasible in adjacent settings.

These findings motivate the hypothesis but do not prove same-image query-dependent non-KV quantization sensitivity.

## Biggest Collision

- QAQ already routes LLM weight bit-planes from query representations.
- PAQ routes prompts among whole-model precision variants; its publication status is **UNVERIFIED** in the current survey.
- TAQ assigns task-conditioned layer precision.
- DP-LLM selects layer precision from runtime inputs across decoding iterations.
- MixKVQ performs query-aware mixed-precision KV-cache quantization.
- WindowQuant uses text-prompt/visual-window similarity to assign visual-window KV precision.
- PMPD already studies prefill/decode and prompt-adaptive precision schedules.
- QUOTA, QAPruner, Bi-VLM, and ZipVL block a simple token-pruning-plus-quantization fallback.

The project must not claim to be the first query-aware, task-aware, or runtime mixed-precision quantization method.

## Open Questions

1. Does the controlled same-image interaction explain at least a meaningful fraction of block-sensitivity variance?
2. Is a per-query oracle materially better than global, per-task, and leave-one-query-out per-image profiles at identical resident-byte budgets?
3. Does the result survive task family, prompt, question/answer length, visual-token count, and quantization-proxy controls?
4. If the phenomenon passes, can 3–4 profiles retain the headroom and can interaction features predict the profile on unseen images?
5. If prediction passes, do real A800 TTFT, TPOT, throughput, memory, batching, and graph costs preserve a Pareto gain?

## Current Canary

**C01 — Same-Image Query × Non-KV Quantization Sensitivity**, specified at [`experiments/02_canary/C01/README.md`](experiments/02_canary/C01/README.md).

- S0 status: **PASS**. All integrity checks passed; frozen manifest SHA-256 is `cfea83045278c17d8f1428b381ef1bee181fc69bf3a128ef41165b38ec3bdc05`. See [`experiments/02_canary/C01/S0_REPORT.md`](experiments/02_canary/C01/S0_REPORT.md).
- S0 scope: Qwen2-VL-2B-Instruct revision `895c3a49bc3fa70a340399125c650a463535e71c`, 8 GQA images / 32 questions, 13 runtime-discovered groups, diagnostic W4A16, and 0.0681 measured A800 GPU-hour for the successful run.
- S0 interpretation: infrastructure is credible enough to request S1; no CIQ-PP scientific claim has been tested.

- Model: Qwen2-VL-2B-Instruct.
- Confirmatory data: 80–100 GQA images with replicated image×query-family cells / 480–600 pairs; VQAv2 and TextVQA only as bounded replication slices.
- Conditions: BF16, static W4A16, single-group W4 perturbation, and equal-byte W8/BF16 restoration from W4.
- Modules: vision quartiles, projector/merger, and decoder attention/MLP depth groups; conditionally refine only the two most sensitive regions to blocks.
- Metrics: teacher-forced NLL, JS divergence, official score, interaction variance, profile ranking, and equal-byte oracle headroom with image-grouped uncertainty.
- Main visualization: Query Type × Layer/Block Quantization Sensitivity Heatmap.
- Cost cap: one A800 80GB, 8–20 GPU-hours; approximately 20–35 GB minimal workspace and up to 80 GB with replication data.

C01 uses resident bytes as a diagnostic resource proxy and makes no latency claim. Real hardware cost is a later gate.

## Current Decision

**CONDITIONAL GO — C01-S1 only, after explicit authorization.**

C01-S0 passed the infrastructure gate. This does not change the preregistered C01 positive, negative, or gray-zone thresholds and does not satisfy the Paper Candidate gate.

- Strong pass requires a controlled interaction partial \(R^2\) of at least 10% with 95% lower bound above 5%, plus at least 1.5 score percentage points and 15 percentage points of relative NLL recovery over the best non-query equal-byte control, replicated across two adequate query families or an independent same-image slice.
- Negative requires, after integrity/proxy checks, an interaction 95% upper bound below 5% and oracle advantage below both 0.5 point and 5% relative NLL recovery, or complete explanation by task/length/prompt controls.
- Results between those thresholds are INCONCLUSIVE and permit one bounded replication or proxy correction, not method development.

Primary: CIQ-PP.
Backup: Quantization-Induced Disagreement Rescue.
Paper Candidate Gate: **FAIL / UNVERIFIED**.

## Next Action

Wait for explicit authorization before running C01-S1. When authorized, begin with the locked BF16 baseline-validity and W4 proxy-dynamic-range checks; the tiny S0 slice had only 4/32 BF16 normalized-exact matches, which is not an S0 failure but is a prespecified S1 validity concern. Do not train a router, write custom kernels, expand to MoE, run a full benchmark matrix, execute S2, or begin Paper Build.
