# Paper Outline

## Working Title

CIQ-PP: Counterfactual Image--Query Precision Profiles for Hardware-Realistic Vision-Language Model Inference.

## Status

This is a pre-candidate outline. Sections describe the intended evidence chain and must not be treated as completed claims.

## Planned Structure

1. Introduction: static VLM quantization policies versus heterogeneous image-query interactions.
2. Related Work: VLM PTQ, dynamic precision, adaptive inference, visual-token selection, KV quantization, and hardware-aware serving.
3. Counterfactual Sensitivity Study: same-image multi-query intervention, controls, and oracle headroom.
4. Method: hardware-constrained profile codebook and interaction router.
5. System: prepacked profiles, request-boundary selection, CUDA graphs, and profile-aware batching.
6. Experiments: accuracy--latency--memory--throughput Pareto, killer baselines, ablations, robustness, and failure cases.
7. Limitations and Threats to Validity.
8. Conclusion.

## Contribution Gate

The paper outline becomes active only if the current canary demonstrates significant same-image interaction and deployable oracle headroom.
