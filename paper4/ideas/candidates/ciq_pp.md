# CIQ-PP Candidate

## Working Name

Counterfactual Image--Query Precision Profiles for Hardware-Realistic VLM Inference.

## Research Question

For a fixed image, does changing only the query create a stable and useful change in the quantization sensitivity of the vision encoder, projector, language-model attention, and language-model MLP modules?

## Candidate Mechanism

1. Estimate a module sensitivity vector by restoring module groups from a W4A16 base to W8/BF16.
2. Remove the image-conditioned mean to isolate the image-query interaction residual.
3. Construct at most three or four profiles from hardware-supported module precisions and measured A800 costs.
4. Route each request using a lightweight interaction of existing pooled visual and query features.
5. Optionally upgrade high-risk requests at the request or first-token boundary using a calibrated quantization-disagreement signal.

## Falsification

The candidate fails if the per-query oracle does not materially improve over static/per-task profiles, if interaction features do not beat task- or query-only controls, or if runtime costs remove the Pareto gain.

## Current Status

Research Opportunity / Conditional GO. No experimental evidence yet. See [`../../CURRENT.md`](../../CURRENT.md).
