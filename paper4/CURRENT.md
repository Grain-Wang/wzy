# Current Research Status

Last updated: 2026-09-06

## Main Direction

**CIQ-PP: Counterfactual Image–Query Non-KV Precision Profiles for Hardware-Realistic VLM Inference.**

The broad phrase “query-aware dynamic quantization” is not the contribution. The selected opportunity asks whether changing only the query for a fixed image changes the useful non-KV VLM sensitivity profile, and—only if that fact is validated—whether the interaction can support at most 3–4 resident, prepacked, hardware-native profiles.

Decision analysis: [`analysis/idea_selection.md`](analysis/idea_selection.md)
Main-direction specification: [`ideas/main_direction.md`](ideas/main_direction.md)
Literature audit: [`literature/surveys/query_aware_vlm_quantization_literature_review.md`](literature/surveys/query_aware_vlm_quantization_literature_review.md)

## Current Stage

**Canary Validation — S1 INCONCLUSIVE.** Literature, gap, collision, and idea-selection gates are complete. C01-S0 passed all integrity checks, and the authorized C01-S1 coarse confirmatory probe completed on 90 GQA images / 540 questions. S1-A baseline validity and S1-B W4 proxy dynamic range passed, but the preregistered controlled interaction gate failed. The project remains a Research Opportunity and has **not** reached the Paper Candidate gate.

S1 found executed equal-byte per-query oracle headroom, but did not validate a stable repeated-cell image×query-family interaction. No latency, hardware benefit, deployable routing, or method claim has been established.

## Core Hypothesis

For a fixed image and model, changing the query creates a stable image-query interaction residual in the W4A16 sensitivity of the vision encoder, multimodal projector, language-model attention, and language-model MLP. The effect is useful only if it changes the best fixed-budget protected-module profile beyond global-static, per-task, per-image, query-only, and sequence-length controls.

## Strongest Supporting Evidence

- Query-guided visual-token methods such as QuietPrune, IVTP, LVPruning, SparseVLM, and MADTP show that visual relevance depends on the current language query.
- Q-VLM, MBQ, VLMQ, TLQ, VLM-PTQ, MABA, and VVSQ show that VLM quantization sensitivity is heterogeneous across modalities, tokens, layers, and modules.
- QAQ, TAQ, DP-LLM, PMPD, MixKVQ, and related dynamic-precision work show that query-, task-, input-, phase-, and token-conditioned precision decisions are feasible in adjacent settings.

These findings motivate the hypothesis but do not prove same-image query-dependent non-KV quantization sensitivity.

C01-S1 adds mixed empirical evidence: BF16 accuracy was 60.00%, the all-W4 proxy produced a non-catastrophic 0.74-point score drop, 0.032510 mean ΔNLL, and 0.012857 mean gold-position JS. The executed +20% per-query oracle exceeded the strongest non-query control by 4.17 score points (95% CI [+0.83, +8.33]), but the controlled interaction partial R² was -0.603177 (95% CI [-0.671533, -0.523433], permutation p=0.051). Oracle headroom therefore cannot yet be attributed to stable image×query-family structure.

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

1. Is the negative repeated-cell interaction estimate a stable rejection, or does one bounded image-grouped expansion change it without altering the valid proxy?
2. Why does the executed per-query oracle show headroom when image×family sensitivity does not predict held-out cell observations: query-identity structure, oracle selection noise, or non-additivity?
3. Does the small held-out all-W4-to-BF16 NLL denominator make relative-recovery magnitude too unstable for method decisions?
4. If one bounded expansion still fails the interaction gate, should CIQ-PP be downgraded and the 26-case recoverable tail used only to preregister the backup disagreement-rescue probe?

## Current Canary

**C01 — Same-Image Query × Non-KV Quantization Sensitivity**, specified at [`experiments/02_canary/C01/README.md`](experiments/02_canary/C01/README.md).

- S0 status: **PASS**. All integrity checks passed; frozen manifest SHA-256 is `cfea83045278c17d8f1428b381ef1bee181fc69bf3a128ef41165b38ec3bdc05`. See [`experiments/02_canary/C01/S0_REPORT.md`](experiments/02_canary/C01/S0_REPORT.md).
- S0 scope: Qwen2-VL-2B-Instruct revision `895c3a49bc3fa70a340399125c650a463535e71c`, 8 GQA images / 32 questions, 13 runtime-discovered groups, diagnostic W4A16, and 0.0681 measured A800 GPU-hour for the successful run.
- S0 interpretation: infrastructure is credible enough to request S1; no CIQ-PP scientific claim has been tested.
- S1 status: **INCONCLUSIVE**. Formal manifest SHA-256 is `6a7d8237438bd1764cb8a9341fdafad0e5d43dbbba448962bdd0f532993a63b0`; see [`experiments/02_canary/C01/S1_REPORT.md`](experiments/02_canary/C01/S1_REPORT.md).
- S1 scope: 90 images / 540 questions / 270 repeated image×family cells; BF16, all-W4, 13 single-group interventions, and 2,160 fixed-budget policy definitions evaluated through 1,878 unique question-profile executions on a held-out 20-image / 120-question subset.
- S1 interpretation: the equal-byte profile gate, family-direction gate, and descriptive ranking-stability gate passed, but the primary interaction gate failed. The result is neither STRONG PASS nor robust NEGATIVE under the locked criteria.

- Model: Qwen2-VL-2B-Instruct.
- Confirmatory data: 80–100 GQA images with replicated image×query-family cells / 480–600 pairs; VQAv2 and TextVQA only as bounded replication slices.
- Conditions: BF16, static W4A16, single-group W4 perturbation, and equal-byte W8/BF16 restoration from W4.
- Modules: vision quartiles, projector/merger, and decoder attention/MLP depth groups; conditionally refine only the two most sensitive regions to blocks.
- Metrics: teacher-forced NLL, JS divergence, official score, interaction variance, profile ranking, and equal-byte oracle headroom with image-grouped uncertainty.
- Main visualization: Query Type × Layer/Block Quantization Sensitivity Heatmap.
- Cost cap: one A800 80GB, 8–20 GPU-hours; approximately 20–35 GB minimal workspace and up to 80 GB with replication data.

C01 uses resident bytes as a diagnostic resource proxy and makes no latency claim. Real hardware cost is a later gate.

## Current Decision

**HOLD — C01-S1 INCONCLUSIVE; S2 NOT AUTHORIZED.**

C01-S0 passed the infrastructure gate and C01-S1 completed without changing the preregistered criteria. S1 failed the conjunctive STRONG PASS gate because controlled interaction partial R² was negative. It does not meet robust NEGATIVE because the executed per-query oracle retained substantial equal-byte headroom. This does not satisfy the Paper Candidate gate.

- Strong pass requires a controlled interaction partial \(R^2\) of at least 10% with 95% lower bound above 5%, plus at least 1.5 score percentage points and 15 percentage points of relative NLL recovery over the best non-query equal-byte control, replicated across two adequate query families or an independent same-image slice.
- Negative requires, after integrity/proxy checks, an interaction 95% upper bound below 5% and oracle advantage below both 0.5 point and 5% relative NLL recovery, or complete explanation by task/length/prompt controls.
- Results between those thresholds are INCONCLUSIVE and permit one bounded replication or proxy correction, not method development.

Primary: CIQ-PP.
Backup: Quantization-Induced Disagreement Rescue.
Paper Candidate Gate: **FAIL / UNVERIFIED**.

## Next Action

Do not execute C01-S2. The only scientifically allowed recommendation is **one bounded, preregistered sample expansion** using the already valid W4 proxy and frozen definitions; it requires a new explicit authorization. If that expansion remains in the gray zone or confirms the absent interaction, downgrade CIQ-PP rather than entering method development. The 26/540 BF16-correct/W4-wrong tail supports only recording that Quantization-Induced Disagreement Rescue is testable; do not start it. Do not train a router, write custom kernels, run a full benchmark matrix, or begin Paper Build.
