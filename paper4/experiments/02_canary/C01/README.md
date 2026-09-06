# C01 — Same-Image Query × Non-KV Quantization Sensitivity

Status: **Designed, not implemented, not run.**

Decision owner: [`../../../CURRENT.md`](../../../CURRENT.md)
Direction: [`../../../ideas/main_direction.md`](../../../ideas/main_direction.md)
Selection analysis: [`../../../analysis/idea_selection.md`](../../../analysis/idea_selection.md)

## Hypothesis

For a fixed image and fixed VLM, changing only the query changes the relative W4A16 sensitivity of non-KV module groups. The variation remains after controlling for query family and sequence-length effects and is large enough that a per-query oracle selects a better fixed-budget protection profile than global-static, per-task, or per-image controls.

This experiment does **not** test whether a learned router works, and it makes no wall-clock speed claim.

## Why This Canary Matters

The broad concept of query-aware quantization is already collided by QAQ, PAQ, TAQ, DP-LLM, MixKVQ, WindowQuant, and PMPD. The only defensible primary direction begins with a narrower unanswered fact: same image, changed query, non-KV precision intervention.

If the effect or equal-budget oracle headroom is absent, there is no reason to train a router or build a dynamic backend. If it is present, the experiment identifies the modules, query families, effect size, and upper-bound benefit needed to design a finite profile codebook.

## Scope

### In scope

- deterministic BF16 and weight-only W4A16 inference;
- controlled perturbation of vision, projector, decoder-attention, and decoder-MLP groups;
- same-image multi-query comparisons;
- teacher-forced and generated-answer metrics;
- fixed resident-weight-byte profile budgets;
- image-grouped bootstrap and prespecified confound checks.

### Out of scope

- router training;
- activation or KV-cache quantization;
- arbitrary runtime bit switching;
- custom kernels or weight repacking during serving;
- wall-clock speedup claims;
- full benchmark matrices or Paper Build.

## Model

**Primary:** `Qwen2-VL-2B-Instruct`.

Rationale:

- it is the model already selected by the repository literature review and current state;
- 2B scale makes repeated block intervention practical on one A800 80GB;
- it includes a real vision encoder, projector/merger path, and decoder blocks, so all non-KV module families in the hypothesis are observable;
- deterministic Hugging Face-style weight replacement is sufficient for the diagnostic and does not require a dynamic kernel.

The model revision, tokenizer revision, Transformers version, image processor settings, and exact parameter-group manifest must be locked before execution. A different model may be used only if an environment smoke test proves this checkpoint unavailable or incompatible; such a substitution requires updating this protocol before examining sensitivity outcomes.

## Dataset and Sampling

### Confirmatory set: GQA

Use public GQA questions with functional programs because multiple questions can be grouped by image and categorized without new human labels.

Prespecify a deterministic sample of approximately 80–100 images, each with at least six valid questions and at least three represented non-OCR query families, with at least two questions in each retained image×family cell. Target 480–600 image-question pairs. The cell replication is required to distinguish an image×family interaction from question-level residual variation. Sample images and query slots before any quantized result is inspected. Keep every selected pair unless it fails a logged, model-independent decoding/data-integrity rule.

### Replication slices

- **VQAv2:** same-image multiple-question groups for an independently worded general-VQA slice.
- **TextVQA:** an OCR slice, preferentially using images with multiple annotated questions. If TextVQA cannot provide a valid same-image cross-family comparison, report OCR only as a secondary sensitivity slice; do not use cross-dataset OCR-vs-GQA differences as evidence for the same-image causal claim.

No generated questions, new manual labels, or selective post-hoc example filtering are allowed in C01.

## Query Taxonomy

The mapping rules must be implemented from existing dataset metadata or prespecified text/program rules and frozen before quantized evaluation.

| Query family | Source and intended rule | Role in C01 |
|---|---|---|
| Global perception | GQA/VQAv2 questions about scene, existence, count, dominant property, or broad category | Confirmatory where same-image pairs exist |
| Fine-grained recognition | Object identity, attribute, color, material, or local detail | Confirmatory |
| OCR | TextVQA reading/transcription and text-grounded questions | Secondary; confirmatory only for valid same-image pairs |
| Spatial relation | GQA relation/location programs and spatial wording | Confirmatory |
| Reasoning | GQA compare, logical composition, relational chain, or multi-step functional program | Confirmatory |

Report category counts, images per category, questions per image, question-token length, answer-token length, and BF16 accuracy before interpreting sensitivity. Any family with fewer than 30 valid questions or fewer than 15 images is descriptive only.

## Quantization Conditions

All quantization is non-KV and weight-only; activations and KV cache remain BF16 to isolate the candidate mechanism.

1. **BF16 reference:** unchanged model, deterministic decoding.
2. **Static W4A16 baseline:** fixed group-wise symmetric round-to-nearest W4 weights and BF16 activations. The exact grouping (initial target: 128 weights/scale), clipping rule, excluded tensors, and scale dtype must be frozen and logged.
3. **Single-group perturbation:** begin from BF16 and quantize one predefined non-KV group to W4 while every other group remains BF16. This supplies the least-confounded marginal sensitivity vector.
4. **Fixed-budget restoration:** begin from static W4A16 and restore selected groups to W8 or BF16 under exactly matched resident-weight-byte budgets. This estimates actionable oracle headroom and checks whether the single-group ranking survives multi-group quantization interactions.

Embedding tables, normalization, rotary embeddings, output head/tied embeddings, and non-linearities remain BF16 unless a later protocol amendment explicitly justifies them. Quantization scales may not depend on the evaluation query.

### Module grouping

Use a two-stage grouping to keep the canary cheap:

- **Coarse stage:** vision encoder depth quartiles; the projector/merger; decoder depth quartiles split into attention and MLP. This yields roughly 13 groups, subject to the actual architecture manifest.
- **Conditional refinement:** only if the coarse stage shows a positive interaction signal, split the two most sensitive decoder regions into individual blocks and attention/MLP subgroups. The resulting Query Type × Layer/Block heatmap is the final C01 visualization.

Every group must list module names, parameter count, BF16 bytes, W8 bytes, W4 bytes, and tensors excluded from quantization.

## Controlled Variables

- image pixels, preprocessing, resolution policy, and visual-token count within each same-image comparison;
- model/tokenizer/checkpoint revision and quantizer recipe;
- prompt/chat template and system prompt;
- decoding: greedy, temperature 0, fixed maximum new tokens, fixed stopping rules;
- answer normalization and official evaluator;
- BF16 activation/KV dtype;
- group definitions and protection budget;
- execution order randomized by image group, with deterministic seeds;
- no per-query calibration or scale fitting;
- question and answer token lengths retained as covariates rather than silently filtered.

For vision perturbations, cache or reuse outputs only when mathematically identical for that precision condition. Never reuse a BF16 visual embedding for a quantized-vision condition.

## Independent Variables

1. Query identity and prespecified query family, varied within a fixed image.
2. Quantized/protected module group.
3. Precision condition: BF16, W4A16, or a W4A16 profile with W8/BF16 restoration.
4. Fixed restoration budget: initially +5%, +10%, and +20% resident weight bytes relative to all-W4.

## Dependent Variables

For image \(i\), query \(q\), and perturbed group \(g\):

- **Normalized teacher-forced loss change**

  \[
  S^{\mathrm{NLL}}_{iqg} = \mathrm{NLL}_{iqg} - \mathrm{NLL}^{\mathrm{BF16}}_{iq}.
  \]

- **Distribution change:** mean Jensen–Shannon divergence from BF16 over gold-answer positions. Use a fixed top-k-plus-other representation if full logits are not stored.
- **End-task change:** official normalized answer score, answer flip, and the event “BF16 correct, quantized condition wrong.”
- **Restoration gain:** W4 loss or score recovered when a group/profile is restored.
- **Sensitivity-vector structure:** rank correlation, cosine distance, top-k protected-group Jaccard overlap, and best-profile identity across queries sharing an image.
- **Budget-matched profile regret:** quality difference from the per-query oracle at the same exact resident-weight-byte budget.

For profile \(P\), define relative NLL recovery as

\[
R(P)=\frac{\Delta\mathrm{NLL}_{\mathrm{W4}}-\Delta\mathrm{NLL}_{P}}
{\max(\Delta\mathrm{NLL}_{\mathrm{W4}},\epsilon)}.
\]

Report the **percentage-point difference** in \(R\) between the interaction/per-query oracle and the strongest non-query control; do not describe a relative ratio as a percentage-point recovery.

NLL and JS are high-resolution mechanism metrics; official task score is required for usefulness. A logit-only difference cannot pass C01 by itself.

## Statistical Analysis

### Primary estimand

Let \(t\) denote query family and \(q\) index repeated questions within an image×family cell. Estimate a group-specific model of the form

\[
S_{itqg} = \mu_g + a_{ig} + b_{tg} + c_{itg} + \beta_g^\top z_{itq} + \epsilon_{itqg},
\]

where \(a_{ig}\) captures image-specific sensitivity, \(b_{tg}\) captures the query family, \(z\) contains question length, answer length, visual-token count, and prompt-template controls, and \(c_{itg}\) is the image×query-family interaction. Fit an additive model without \(c\) and an interaction model with \(c\), using repeated questions to evaluate held-out cell observations. Define the primary interaction ratio as the cross-validated partial \(R^2=(\mathrm{SSE}_{add}-\mathrm{SSE}_{int})/\mathrm{SSE}_{add}\), aggregated across groups with a prespecified weighting. Report its image-grouped 95% bootstrap interval. Query-identity variation within a family is reported separately and is not silently relabeled as the estimable image×family interaction.

### Required comparisons

At each byte budget compare:

1. global static profile;
2. per-query-family/task profile;
3. per-image profile based on the mean of that image’s other queries (leave-one-query-out);
4. per-image×family profile based on other questions in the same cell;
5. per-query oracle profile;
6. random profile with identical byte budget.

The oracle may use measured C01 sensitivity and is explicitly an upper bound, not a deployable method. Build candidate protected sets from the single-group restoration gains, then execute the selected multi-group profiles on a held-out subset to quantify non-additivity. Do not report an additive estimate as if it were an executed result.

### Uncertainty and leakage control

- Bootstrap whole images, never individual questions, for 95% confidence intervals.
- Use a within-image permutation test for query-family/block interaction and report effect size, not only a p-value.
- Lock the sample manifest, taxonomy, budgets, metrics, and thresholds before inspecting quantized outcomes.
- If a simple predictive sanity probe is run, split by image and compare task-only, image-only, query-only, additive image+query, and interaction features. It is exploratory and must not be called a router.

## Metrics

Primary metrics:

- interaction variance ratio for \(S^{\mathrm{NLL}}\) and JS;
- per-query oracle improvement over the best non-query control at matched bytes;
- relative recovery of all-W4 quantization-induced NLL;
- official score difference with image-grouped 95% CI;
- profile ranking/identity stability across bootstrap resamples.

Secondary metrics:

- top-k group overlap and sensitivity-vector distance within image;
- results by query family;
- BF16 score, all-W4 score, output-token length, and quantization failure rate;
- coarse-to-refined group consistency;
- executed-profile non-additivity error.

Latency, TTFT, TPOT, throughput, and CUDA-graph behavior are **not** C01 claims. Record gross runtime only for experiment accounting. If C01 passes, a separate hardware gate must replace byte proxy with real A800 costs.

## Visualization

Required outputs, once the experiment is authorized and run:

1. **Query Type × Layer/Block Quantization Sensitivity Heatmap:** image-grouped mean standardized NLL/JS sensitivity, with bootstrap uncertainty or a companion confidence panel.
2. Same-image pairwise sensitivity-vector distance distributions, separated into same-family and cross-family pairs.
3. Fixed-budget oracle headroom/regret curves for global, task, image, and per-query policies.
4. Top-k protected-block overlap matrix by query family.

Every plot must show sample counts and distinguish confirmatory GQA results from secondary VQAv2/TextVQA slices.

## Sequential Execution Plan

This is one canary with early stopping, not a full benchmark sweep.

### C01-S0: integrity smoke test

- 8–16 images and 30–60 questions;
- BF16 repeatability, W4 application, single-group restore identity, evaluator, and cached-vision equivalence tests;
- expected time: less than 1 A800 hour.

Stop for implementation repair if unchanged BF16 outputs are not deterministic, restored tensors do not hash/compare correctly, or quantizing an empty group changes outputs.

### C01-S1: coarse confirmatory probe

- 80–100 GQA images and 480–600 questions, with replicated image×family cells;
- BF16, all-W4, about 13 single-group perturbations, and fixed-budget restoration profiles;
- image-grouped statistics and prespecified taxonomy;
- expected time: 6–14 A800 hours.

### C01-S2: conditional refinement/replication

- run only if S1 is positive or genuinely inconclusive;
- refine the two most sensitive decoder regions to layer/block granularity;
- add the smallest valid VQAv2/TextVQA replication slices;
- expected time: 2–6 A800 hours.

Do not proceed to S2 after a robust negative S1.

## Positive Criterion

C01 is a **STRONG PASS** only if all of the following hold:

1. after controls, the image-query interaction explains at least 10% of sensitivity variance, its image-bootstrap 95% lower bound exceeds 5%, and the within-image permutation test gives \(p<0.01\);
2. at one prespecified byte budget, the executed per-query oracle improves official answer score by at least **1.5 percentage points on the 0–100 task scale** over the best global/task/image control and gains at least **15 percentage points of relative NLL recovery** over that control, with the image-bootstrap 95% CI excluding zero;
3. the direction of the result replicates in at least two query families with adequate counts or in one independent same-image dataset slice; and
4. sensitivity/profile rankings are stable across bootstrap resamples rather than driven by a few images.

A pass authorizes design of a small profile codebook and simple predictive controls. It does not authorize full Paper Build.

## Negative Criterion

C01 is a **NEGATIVE** for Direction A when, after the integrity checks and at most one justified quantization-proxy correction:

- the 95% upper confidence bound of the controlled interaction variance ratio is below 5%; **and**
- at every budget the per-query oracle advantage is below both **0.5 absolute score point** and **5% relative NLL recovery**; or
- the apparent advantage disappears after task family, question/answer length, visual-token count, and prompt-template controls, leaving no image-query interaction beyond query/task-only prior work.

A robust negative archives Direction A. It must not be rescued by tuning thresholds or training a router.

Results between the positive and negative criteria are **INCONCLUSIVE**. Permit one prespecified sample-size expansion or one proxy correction; record which was chosen before rerunning. If the result remains in the gray zone, downgrade Direction A rather than entering method development.

## Failure Interpretation

| Observation | Interpretation | Allowed response |
|---|---|---|
| BF16 has floor-level accuracy or cannot separate query families | Model/dataset validity failure | Replace the invalid slice or model once, before sensitivity inspection; update protocol. |
| W4 collapses nearly all answers or changes no outputs | Quantization proxy has no useful dynamic range | Verify tensors/scales; once only, move to a better W4 recipe or W8 perturbation. |
| Teacher-forced NLL changes but official answers and oracle headroom do not | Sensitivity is measurable but not useful | Do not proceed with CIQ-PP. |
| Query-family effect exists, but per-image/query interaction vanishes under controls | Task-aware rather than image-query-aware effect | Direction A loses novelty against TAQ/QAQ; downgrade or archive. |
| Per-query oracle is strong but rankings are unstable | Measurement/sample instability | One image-grouped replication is allowed; no router training. |
| Single-group estimates predict gains that executed multi-group profiles do not realize | Non-additive proxy failure | Report it; redesign the profile objective only if actual profiles still show headroom. |
| Same-image sensitivity and headroom are robustly absent | Core hypothesis is false or too weak at useful W4 budgets | Archive Direction A and evaluate the backup only if paired outputs support a recoverable error tail. |

## Estimated GPU Time

- Integrity smoke test: **<1 A800 GPU-hour**.
- Coarse confirmatory probe: **6–14 A800 GPU-hours**.
- Conditional refinement and replication: **2–6 A800 GPU-hours**.
- Total cap: **8–20 A800 GPU-hours**, approximately one to two natural days on one A800 80GB.

The cap excludes checkpoint/dataset download time. Exceeding 20 GPU-hours requires a recorded reason and a new decision; it must not silently expand into a benchmark matrix.

## Estimated Storage

- Minimal GQA confirmatory workspace, model cache, manifests, and compact outputs: approximately **20–35 GB**.
- With full VQAv2/TextVQA replication archives: up to approximately **80 GB**.
- Result artifacts should remain below **3 GB** by storing aggregate/top-k distribution statistics rather than full-vocabulary logits.

Raw outputs, processed statistics, tables, and figures must be written to their respective `paper4/results/` subdirectories when execution is later authorized. No data or model artifact belongs in Git.

## Decision Record Template

After execution, record without changing the prespecified thresholds:

- Outcome: STRONG PASS / INCONCLUSIVE / NEGATIVE
- Integrity checks:
- Sample manifest hash:
- Model and dependency revisions:
- Interaction variance ratio and 95% CI:
- Best equal-byte oracle gain and 95% CI:
- Query-family replication:
- Confound checks:
- Runtime and storage actually consumed:
- Scientific interpretation:
- Next gate:

Until those fields are populated from real artifacts, C01 provides no positive evidence.
