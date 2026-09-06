# Main Direction: CIQ-PP

## Decision

The primary `paper4` direction is **CIQ-PP: Counterfactual Image–Query Non-KV Precision Profiles for Hardware-Realistic VLM Inference**.

Status: **Research Opportunity / Canary Validation / S1 INCONCLUSIVE.** It is not a Paper Candidate. C01-S2 and method implementation are not authorized.

Backup: **Quantization-Induced Disagreement Rescue**, retained only if the primary direction is falsified or proves operationally unusable.

## Problem

Current VLM PTQ policies are predominantly calibrated offline and fixed during deployment. The missing scientific fact is whether, for a fixed image, changing only the query creates a stable and useful change in the non-KV module quantization sensitivity vector:

\[
s(I,q) = [s_1(I,q), \ldots, s_G(I,q)],
\]

where groups span the vision encoder, multimodal projector, language-model attention, and language-model MLP. “Useful” means that the change alters the best precision profile at a fixed resource budget by enough to improve end-task quality, not merely that two floating-point diagnostics differ.

## Claim Boundary

The project must not claim any of the following as novel:

- query-aware, prompt-aware, task-aware, input-aware, or sample-wise quantization in general;
- a query or precision router by itself;
- query-dependent visual-token importance;
- query-aware KV-cache precision;
- prefill/decode asymmetric precision;
- token pruning plus quantization.

QAQ, PAQ, TAQ, DP-LLM, MixKVQ, WindowQuant, PMPD, QUOTA, QAPruner, and related work already occupy those claims. The literature status and URLs are maintained in the [`literature survey`](../literature/surveys/query_aware_vlm_quantization_literature_review.md).

## Core Hypothesis

For the same image, query changes induce an interaction residual in non-KV module sensitivity that:

1. remains after controlling for task family, prompt template, question/answer length, and image preprocessing;
2. changes the optimal fixed-budget set of protected modules;
3. creates material oracle headroom over global-static, per-task, and per-image profiles; and
4. can later be compressed into at most 3–4 resident hardware-native profiles without losing most of that headroom.

C01 tests only points 1–3. Point 4 and learned routing are conditional later gates.

## Proposed Algorithmic Object, Conditional on C01

If C01 passes, sensitivity will be modeled as

\[
s_{i q g} = \mu_g + a_{i g} + b_{t(q)g} + c_{i q g} + \epsilon_{i q g},
\]

where \(a\) is image-specific, \(b\) is task/query-family-specific, and \(c\) is the image-query interaction residual. The later method would jointly select a small profile codebook \(\mathcal{P}=\{P_1,\ldots,P_K\}, K\leq4\), and request assignment \(r(I,q)\) to minimize quality risk plus measured serving cost:

\[
\min_{\mathcal{P},r}\; \mathbb{E}_{I,q}[\mathcal{L}(P_{r(I,q)};I,q)]
+ \lambda C_{\mathrm{A800}}(P_{r(I,q)}).
\]

Profiles must use backend-supported formats, be packed before serving, and switch only at request boundaries. Arbitrary per-token or per-layer bit changes are outside the main design. A lightweight router is a means of estimating assignment, not the core novelty.

## Why This Is a Research Opportunity

- **Problem evidence:** query-guided token work establishes query-dependent visual relevance; VLM PTQ establishes strong module/modality sensitivity heterogeneity; adjacent LLM/KV work establishes that runtime precision adaptation is feasible.
- **Nearest-work gap:** no located study performs the fixed-image, changed-query non-KV intervention and equal-budget profile test.
- **Algorithmic path:** interaction-residual estimation plus joint finite-profile codebook optimization is more specific than attaching a query encoder to a static allocator.
- **Falsifiability:** one small model and public multi-question datasets can reject the premise before router or systems work.
- **Resource fit:** C01 requires one A800, no training, and no new human annotation.

These are motivations, not evidence that the hypothesis is true.

## Required Baselines After a Canary Pass

No full comparison is in scope yet. Before Paper Candidate promotion, the method must eventually beat:

- BF16, uniform W8A16/W4A16, and the best static profile;
- MBQ/MABA-style static modality-aware allocation;
- per-task and per-image oracle/profile controls;
- QAQ-style query-only routing and TAQ-style task-conditioned allocation;
- DP-LLM-style runtime local-error selection;
- random and unconstrained oracle routing;
- relevant PMPD and MixKVQ/WindowQuant baselines for phase/KV scope separation.

## Canary Gate

Run [`C01`](../experiments/02_canary/C01/README.md): a Qwen2-VL-2B same-image multi-query, non-KV W4A16 block perturbation and fixed-budget restoration probe.

- **Strong pass:** a nontrivial interaction effect with image-grouped uncertainty, plus at least 1.5 score percentage points and 15 percentage points of relative quantization-loss recovery over the best static/per-task/per-image control at the same resident-byte budget, replicated in at least two query families or an independent dataset slice.
- **Negative:** after implementation/proxy checks, the interaction upper confidence bound is below 5% and the per-query oracle advantage is below 0.5 point and 5% relative loss recovery; or the apparent effect is fully explained by task/length/prompt controls.
- **Inconclusive:** anything between those thresholds. Permit one prespecified sample expansion or one justified quantization-proxy correction. Do not train a router.

C01-S1 completed on 2026-09-06 with outcome **INCONCLUSIVE**. The exact-byte per-query oracle profile gate passed, but the primary controlled interaction gate failed: partial R² -0.603177, 95% CI [-0.671533, -0.523433], permutation p=0.051. The result authorizes neither S2 nor method work. Only one bounded, preregistered sample expansion may be proposed in a later explicitly authorized round.

## Backup Direction

If the primary interaction is absent but paired BF16/W4 outputs contain a recoverable error tail, test **Quantization-Induced Disagreement Rescue**: predict “W4 wrong and BF16 correct” and compare with generic entropy/margin at an identical rescue rate. The backup is discarded if it cannot outperform generic confidence after accounting for recomputation.

## Next Action

Stop after C01-S1. Do not execute S2 or the backup experiment. The only allowed recommendation is one bounded, preregistered sample expansion using the already valid W4 proxy, subject to new explicit authorization. Do not begin model-scale router training, custom kernels, full benchmark matrices, MoE extensions, or Paper Build.
