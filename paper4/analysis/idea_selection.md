# paper4 Research Direction Selection

Date: 2026-09-05

Status: **Research Opportunity decision complete; C01-S1 INCONCLUSIVE.** S0 and S1 results now exist, but the controlled interaction gate did not pass; none of the directions below has passed the Paper Candidate gate. See [`S1_REPORT.md`](../experiments/02_canary/C01/S1_REPORT.md).

## 1. Evidence Basis and Repository Reconciliation

This decision uses, in priority order, the repository rules, current files and result directories, [`CURRENT.md`](../CURRENT.md), and the handoff. The main evidence source is the 100-paper [`Query-Aware Dynamic Precision for VLMs — Literature Review`](../literature/surveys/query_aware_vlm_quantization_literature_review.md), supported by the existing [`literature_gap.md`](literature_gap.md), [`novelty_matrix.md`](novelty_matrix.md), and [`CIQ-PP candidate`](../ideas/candidates/ciq_pp.md).

The repository facts are internally consistent on the scientific status:

- The broad phrase and mechanism of query-aware dynamic mixed-precision quantization are already occupied by close work.
- No located work provides a controlled same-image/different-query intervention across non-KV VLM modules.
- At direction-selection time, `paper4/results/` contained no scientific result artifact. The later C01-S1 result is mixed: executed oracle headroom exists, while the preregistered controlled interaction partial R² is negative. It does not retrospectively turn the direction-selection rationale into experimental evidence.
- Existing paper files are pre-candidate scaffolds. They do not authorize Paper Build.
- The earlier canary description mixed a scientific sensitivity test with “measured-cost profile regret.” C01 below deliberately uses an exact resident-weight-byte budget as a diagnostic proxy. It makes no latency claim. A real A800 backend gate is required later, but only if C01 passes.
- The literature documents a gray zone between a strong continuation threshold (1.5 absolute score points) and a stop threshold (0.5 points). This decision makes that zone explicit as **INCONCLUSIVE**, rather than selecting whichever threshold favors the direction.

## 2. Direct Review of the Query-Aware Dynamic Precision Claim

### Q1. Has the fixed-image, changed-query sensitivity question already been studied?

**No direct controlled study was found in the completed survey.** In particular, no located paper fixes an image, varies only its query, perturbs precision across the vision encoder, multimodal projector, language-model attention, and language-model MLP, and then tests whether the best non-KV precision profile changes.

The nearest evidence is:

- [QuietPrune](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_QuietPrune_Query-Guided_Early_Token_Pruning_for_Vision-Language_Models_CVPR_2026_paper.html), [IVTP](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02577.pdf), and [LVPruning](https://aclanthology.org/anthology-files/pdf/naacl/2025.naacl-findings.242.pdf): the language query changes visual-token relevance, but they do not intervene on quantization sensitivity.
- [QAQ](https://neurips.cc/virtual/2025/129098), [TAQ](https://arxiv.org/abs/2511.06516), and [DP-LLM](https://arxiv.org/abs/2508.06041): query-, task-, or runtime-input-conditioned weight precision in text LLMs, without a visual path or a same-image control.
- [MixKVQ](https://aclanthology.org/2026.acl-long.326/) and [WindowQuant](https://arxiv.org/abs/2605.02262): query-aware KV channels or prompt-aware visual-window KV precision, not cross-module non-KV VLM profiles. WindowQuant is a 2026 preprint in the survey.
- [MBQ](https://openaccess.thecvf.com/content/CVPR2025/html/Li_MBQ_Modality-Balanced_Quantization_for_Large_Vision-Language_Models_CVPR_2025_paper.html) and [MABA](https://openaccess.thecvf.com/content/CVPR2026F/html/Zhang_Modality-Aware_Bit_Allocation_for_Mixed-Precision_Quantization_of_Vision-Language_Models_CVPRF_2026_paper.html): strong static modality-aware VLM quantization baselines, but their deployed allocation does not change per image-query request.

Therefore the unverified claim is narrower than “query-aware quantization”: **the image-query interaction may alter the useful non-KV module sensitivity vector and the best budget-matched profile.**

### Q2. Direct collision audit

| Search concept | Located collision | Consequence |
|---|---|---|
| query-aware / query-adaptive quantization | QAQ; MixKVQ; A2ATS | Direct phrase collision; “first query-aware quantization” is invalid. |
| query-conditioned / prompt-aware quantization | QAQ; PAQ; PMPD | Query or prompt representation already routes precision. PAQ publication status remains **UNVERIFIED** in the repository survey. |
| task-aware quantization / bit allocation | TAQ; MABA-like task/modality calibration | Task-conditioned layer policies are not new. |
| input-/instance-aware and sample-wise precision | Instance-Aware Dynamic NN Quantization; USDN; IGQ-ViT | Mature adjacent-field mechanism; moving it to a VLM is insufficient. |
| dynamic/adaptive precision VLM | WindowQuant; Dynamic KV Quantization for MLLMs; Quant Experts | Direct for KV, windows, or correction; partial for a complete VLM request. |
| precision routing / quantization router | QAQ; PAQ; MoQAE; Dynamic Mixed-Precision Routing | A router alone cannot be the novelty. |
| runtime mixed precision | DP-LLM; PMPD; Any-Precision; Bit-Mixer | Layer-, phase-, step-, and model-level switching already exist. |
| prompt-conditioned precision | PMPD; PAQ | Direct adjacent collision. |
| task-conditioned bit allocation | TAQ | Direct collision. |

The collision rating of a broad Direction A is therefore **High**. It becomes defensible only through the conjunction of a same-image counterfactual estimand, non-KV VLM scope, interaction-specific controls, a small jointly optimized profile codebook, and a later real-hardware demonstration.

### Q3. What drives existing mixed-precision allocation?

Existing work uses Hessian or second-order reconstruction error (for example VLMQ/GPTQ-style methods), gradients or loss sensitivity (MBQ, TLQ, MABA), activations and outliers (AWQ/SmoothQuant-family work), layer-wise reconstruction or logit KL (Q-VLM, TAQ, DP-LLM), token saliency and attention persistence (MiKV, ZipCache, QAPruner), expert frequency/routing statistics (VEQ/MODE and MoE quantization), and explicit hardware cost, packing, or latency constraints (MABA, ScaleBITS, QuantX, QServe).

Query/task conditions are also already used: QAQ consumes a query hidden representation; PAQ uses prompt encoding; TAQ uses task calibration prompts; MixKVQ uses query relevance; WindowQuant uses text-prompt/visual-window similarity. What remains unverified is whether an **image-query interaction term** predicts the best budget-constrained precision allocation across non-KV VLM modules better than image-only, query-only, task-only, or additive controls.

### Q4. Do sample-wise, token-wise, and runtime switching already exist?

**Yes.** Sample-wise policies appear in Instance-Aware Dynamic NN Quantization and USDN; token- or decode-step-wise decisions appear in MoBiQuant, DP-LLM, KVmix, and ZipCache; runtime switching appears in Bit-Mixer, Any-Precision, QuEPT, QAQ, and PMPD.

They do not settle the present question because most omit the visual path, do not use a fixed-image counterfactual design, operate only on KV/depth, or require a trained elastic weight representation. They nevertheless occupy the generic algorithmic territory and must later be adapted as killer baselines. VLM domain transfer by itself is not a contribution.

### Q5. Does query-guided token pruning establish query-dependent visual importance?

**Yes, as motivation.** QuietPrune, IVTP, LVPruning, SparseVLM, and MADTP provide strong evidence that a query changes which visual tokens matter. This does not imply that the same modules require different precision: relevance, error propagation, and quantization sensitivity are different quantities.

It also makes a token-plus-precision headline more collided, not automatically stronger. [QUOTA](https://arxiv.org/abs/2604.17320), [QAPruner](https://arxiv.org/abs/2604.02816), Bi-VLM, and ZipVL already combine or jointly reason about token sparsity and quantization. Direction C should replace A only if a factorial canary proves a non-additive pruning–quantization interaction that cannot be captured by either ordering. No such repository evidence exists.

## 3. Research Opportunities

### Direction A — Counterfactual Image–Query Non-KV Precision Profiles (CIQ-PP)

| Field | Assessment |
|---|---|
| Problem | Static VLM PTQ protects the same modules for every request, even though the query may change how a fixed visual representation is used. It is unknown whether this produces a useful request-specific precision mismatch. |
| Existing Strongest Baseline | MBQ/MABA-style static modality-aware allocation, plus VLM adaptations of QAQ, TAQ, and DP-LLM as dynamic killer baselines. |
| Nearest Papers | MBQ, MABA, VLMQ, TLQ; QAQ, PAQ, TAQ, DP-LLM; MixKVQ and WindowQuant; QuietPrune, IVTP, and LVPruning. |
| What They Already Solve | Static VLM sensitivity, modality/token calibration, query/task/runtime precision routing in LLMs, query-aware KV precision, and query-aware visual-token relevance. |
| Remaining Gap | A fixed-image counterfactual measurement of non-KV VLM sensitivity; evidence that it changes the best equal-budget profile beyond task/query/image-only controls; and a small deployable profile codebook. |
| Proposed Algorithmic Difference | If C01 passes, decompose sensitivity into global, image, task/query, and image-query residuals; jointly learn at most 3–4 resident hardware-native profiles and a request-boundary assignment under measured cost, rather than predicting arbitrary per-layer bits. |
| Why This Is Not Simple Module Combination | The candidate algorithmic object is the joint codebook/assignment optimization over counterfactual interaction residuals and real profile cost. A query router attached to MABA is explicitly insufficient. The distinction must be demonstrated by additive-vs-interaction controls and by a finite-profile regret analysis. |
| Direct Collision Risk | **High** for the broad wording; **Medium–High** for the narrowed conjunction, pending C01. |
| Scientific Value | Tests whether quantization sensitivity is partly an input interaction rather than only a static model property, and whether that property is actionable under a discrete deployment constraint. |
| Expected Experimental Signal | Different query families on the same image produce different block sensitivity rankings; a per-query oracle rescues more W4 loss than global, per-task, or per-image profiles at identical resident-byte budget. |
| Required Compute | C01: 8–20 A800 GPU-hours. A full two-model validation would be roughly 300–600 GPU-hours only after all gates pass. |
| Implementation Difficulty | Medium for C01; high for a complete codebook/router method. |
| Hardware Difficulty | Medium–High. Request-boundary selection among resident profiles is feasible; arbitrary bit switching and repacking are not. |
| Minimum Canary Experiment | C01 same-image multi-query non-KV block perturbation and fixed-budget restoration on Qwen2-VL-2B. |
| Failure Criterion | With validated perturbations, the 95% upper bound of the interaction variance ratio is below 5% and per-query oracle headroom is below both 0.5 score point and 5% relative loss recovery, or all variation is explained by task/query length/prompt controls. |
| Single-A800 Feasibility | Yes. The model and all perturbations fit easily on one A800 80GB; no training or human labels are needed. |
| Potential Publication Level | Conditional strong CCF-C / weak CCF-B only if C01, prediction, killer baselines, and real A800 Pareto all pass. Currently not a Paper Candidate. |

### Direction B — Visual-Prefix Prefill/Decode Asymmetric Quantization

| Field | Assessment |
|---|---|
| Problem | Visual prefill and autoregressive decode have different kernels, reuse patterns, and error propagation, so a single VLM precision policy may be inefficient. |
| Existing Strongest Baseline | PMPD for high-precision prefill/progressive low-precision decode, LLM-PQ for phase-aware deployment, MBQ for prefill/decode quantization paths, and modern KV quantization. |
| Nearest Papers | PMPD, LLM-PQ, MBQ, KIVI, KVQuant, PM-KVQ, SparseVILA, and ZipVL. |
| What They Already Solve | Phase-specific precision, placement, KV accumulation, and visual-prefix sparsity/compression. |
| Remaining Gap | Whether a visual prefix has a quantitatively distinct precision/error-propagation requirement from text prefill at matched TTFT/TPOT, not merely a different phase label. |
| Proposed Algorithmic Difference | A visual-prefix-specific error budget that allocates precision jointly across vision/projector/prefill and decode under TTFT, TPOT, and memory constraints. |
| Why This Is Not Simple Module Combination | It would require a measured visual-prefix propagation model and a constrained optimizer that beats a PMPD schedule. Merely assigning W8 to prefill and W4 to decode is rejected. |
| Direct Collision Risk | **High.** PMPD already occupies the central phase-asymmetry claim. |
| Scientific Value | Moderate: potentially useful deployment insight, but the uniquely multimodal mechanism is currently hypothetical. |
| Expected Experimental Signal | Visual-prefix perturbation causes a different downstream error curve from length-matched text prefill, and an optimized schedule improves TTFT/TPOT at fixed accuracy. |
| Required Compute | Canary 8–16 A800 GPU-hours; full backend evaluation 150–350 GPU-hours. |
| Implementation Difficulty | Medium. |
| Hardware Difficulty | Medium; phase boundaries are kernel- and graph-friendly. |
| Minimum Canary Experiment | Compare matched W4/W8 interventions in visual-prefix prefill, text prefill, and decode while controlling token count and generation length. |
| Failure Criterion | Visual-prefix effects are explained by token count/activation scale, or a PMPD-style two-phase schedule matches the proposed allocation. |
| Single-A800 Feasibility | Yes. |
| Potential Publication Level | Strong CCF-C at best unless a distinct visual-prefix mechanism is established; weak-B potential is lower than A or D. |

### Direction C — Query-Aware Visual Token + Precision Co-Design

| Field | Assessment |
|---|---|
| Problem | Token pruning changes activation distributions and quantization-error propagation, so independently selected sparsity and precision policies may be suboptimal. |
| Existing Strongest Baseline | QuietPrune/IVTP for query-guided pruning and QUOTA/QAPruner/ZipVL/Bi-VLM for pruning–quantization combinations. |
| Nearest Papers | QuietPrune, IVTP, LVPruning, QUOTA, QAPruner, Bi-VLM, and ZipVL. |
| What They Already Solve | Query-aware token selection, quantization-aware pruning, joint or scheduled token/bit compression, and dynamic KV/token sparsity. |
| Remaining Gap | A demonstrably non-separable interaction where every sequential pruning→quantization or quantization→pruning baseline is suboptimal under real kernels. |
| Proposed Algorithmic Difference | A joint constrained selection objective over query-conditioned token retention and a small precision codebook, with explicit cross-term modeling and kernel-compatible groups. |
| Why This Is Not Simple Module Combination | It is defensible only if factorial intervention shows a sizable cross-term and the joint optimizer wins against both orderings. Concatenating QuietPrune and W4 is explicitly rejected. |
| Direct Collision Risk | **Very High.** |
| Scientific Value | Potentially high if non-separability is real; otherwise it is a crowded compression composition. |
| Expected Experimental Signal | The quality loss of combined pruning and W4 differs materially from the sum of individual losses, and the sign/pattern depends on the query. |
| Required Compute | Canary 12–30 A800 GPU-hours; full system 300–700 GPU-hours. |
| Implementation Difficulty | High. |
| Hardware Difficulty | High because mixed token counts and precision fragment batching and kernels. |
| Minimum Canary Experiment | A 2×2 query-stratified factorial test: unpruned/pruned × BF16/W4, compared with QUOTA/QAPruner-style controls. |
| Failure Criterion | The interaction term is negligible or a sequential baseline accounts for all benefit. |
| Single-A800 Feasibility | Canary yes; a convincing full study is possible but expensive. |
| Potential Publication Level | Strong CCF-C only if the non-additive mechanism is large and hardware-real; otherwise no-go. |

### Direction D — Quantization-Induced Disagreement Rescue

| Field | Assessment |
|---|---|
| Problem | A uniformly low-precision model has a tail of requests where quantization, rather than intrinsic task uncertainty, changes the answer. Generic entropy cannot identify that cause. |
| Existing Strongest Baseline | CALM-style calibrated confidence, TPT/C-TPT/ZERO uncertainty adaptation, PMPD prompt scheduling, and Dynamic Mixed-Precision Routing. |
| Nearest Papers | CALM, TPT/C-TPT/ZERO, PMPD, Dynamic Mixed-Precision Routing, QAQ, and quality-adaptive KV quantization. |
| What They Already Solve | Confidence-based compute escalation, prompt/task adaptive compute, two-tier precision routing, and generic test-time uncertainty handling. |
| Remaining Gap | A cheap, calibrated estimate of **quantization-induced disagreement** that separates recoverable low-bit errors from requests that are already wrong in BF16. |
| Proposed Algorithmic Difference | A sequential decision/optimal-stopping policy trained or calibrated on paired BF16–low-bit counterfactual disagreement, minimizing expected quality risk plus measured recomputation cost. |
| Why This Is Not Simple Module Combination | The target variable is causal precision disagreement, not entropy; the policy must be calibrated against false rescue and compare against generic confidence at the same escalation rate. |
| Direct Collision Risk | **Medium–High.** Two-tier precision routing is collided, but causal disagreement calibration remains narrower. |
| Scientific Value | High if it selectively protects low-bit failures and supplies risk–cost control; it also provides a practical safety mechanism. |
| Expected Experimental Signal | At a 5–10% rescue rate, a disagreement score captures substantially more BF16-recoverable W4 errors than entropy, margin, or random rescue. |
| Required Compute | Canary 10–24 A800 GPU-hours; it can reuse paired outputs from C01. |
| Implementation Difficulty | Medium. |
| Hardware Difficulty | Medium. Whole-request or first-token escalation is more regular than per-layer switching but recomputation can erase savings. |
| Minimum Canary Experiment | On paired BF16/W4 outputs, predict the event “W4 wrong and BF16 correct”; compare AUROC/AUPRC and risk–coverage against entropy, margin, answer consistency, and random rescue. |
| Failure Criterion | The disagreement detector does not beat generic confidence at matched rescue rate, or required recomputation removes the cost benefit. |
| Single-A800 Feasibility | Yes. |
| Potential Publication Level | Strong CCF-C / possible weak CCF-B if it gives calibrated guarantees and real end-to-end savings across models; otherwise an auxiliary component only. |

## 4. Direction Comparison

Ratings are relative within `paper4`; “publication potential” assumes later completion of the relevant gates, not current readiness.

| Direction | Novelty | Collision Risk | Scientific Value | Algorithmic Depth | Expected Signal | Implementation Difficulty | Hardware Difficulty | Single-A800 Cost | Hardware Friendliness | Publication Potential |
|---|---|---|---|---|---|---|---|---|---|---|
| **A. CIQ-PP** | Medium–High only in narrowed conjunction | High | High | High after phenomenon gate | Medium; unverified | Medium canary / High method | Medium–High | 8–20 h canary | Medium if K≤4 resident profiles | **Rank 1: strong C / weak B conditional** |
| **D. Disagreement Rescue** | Medium | Medium–High | High | Medium–High | Medium–High | Medium | Medium | 10–24 h canary | Medium–High at request boundary | **Rank 2: strong C / weak B conditional** |
| **B. Phase Asymmetry** | Low–Medium | High | Medium | Medium | Medium–High | Medium | Medium | 8–16 h canary | High | **Rank 3: strong C ceiling without new mechanism** |
| **C. Token + Precision** | Low–Medium | Very High | Medium–High | High if non-separable | Medium | High | High | 12–30 h canary | Low–Medium | **Rank 4: strong C only after hard interaction proof** |

## 5. Final Selection

### Primary Direction

**CIQ-PP: Counterfactual Image–Query Non-KV Precision Profiles for Hardware-Realistic VLM Inference.**

This is not selected because “query-aware” is new; it is not. It is selected because the completed survey leaves one precise, falsifiable intersection unanswered, and that intersection can be tested without implementing a router. The prospective algorithmic novelty is:

1. an estimand that separates global, image, task/query, and image-query interaction components of module sensitivity;
2. a jointly optimized codebook of at most 3–4 backend-native resident profiles under measured serving cost; and
3. request-boundary assignment based on interaction features, evaluated against image-only, query-only, task-only, additive, static, random, and oracle controls.

Only item 1 and the oracle headroom are in scope for C01. Items 2–3 are conditional future work.

### Backup Direction

**Quantization-Induced Disagreement Rescue.**

This is retained as the sole backup because it targets a different decision object—whether low-bit execution caused a recoverable error—while reusing paired BF16/W4 observations. It must beat generic uncertainty at a matched rescue rate and include recomputation cost. Generic “entropy-triggered high precision” is not an acceptable contribution.

### Rejected as Current Main Line

- Direction B is too close to PMPD/LLM-PQ unless a visual-prefix-specific mechanism is first proven.
- Direction C has the strongest motivation but the worst collision and hardware regularity; QUOTA/QAPruner/ZipVL prevent a simple co-design claim.

## 6. Gate Decision

- **Decision:** CONDITIONAL GO, only for C01.
- **Confidence:** 0.78 that C01 is the highest-value next action; low confidence that the core phenomenon will pass because no project experiment exists.
- **Research Opportunity Gate:** PASS. The defect is precise, nearest work does not directly test it, the candidate has an algorithmic path, and the probe is affordable and falsifiable.
- **Paper Candidate Gate:** FAIL / UNVERIFIED. There is no phenomenon result, no killer-baseline comparison, no stable improvement, and no real-hardware Pareto.
- **Stop rule:** A robust C01 negative result archives Direction A. A gray-zone result permits one bounded replication or proxy correction, not router development or threshold tuning.

The executable specification is [`../experiments/02_canary/C01/README.md`](../experiments/02_canary/C01/README.md). The selected direction is maintained in [`../ideas/main_direction.md`](../ideas/main_direction.md).
