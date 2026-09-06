# Literature Gap

## Established Prior Art

The broad idea of adapting quantization precision to an input, query, prompt, task, token, or decoding phase is not open. QAQ, PAQ, TAQ, DP-LLM, PMPD, MixKVQ, and WindowQuant occupy substantial parts of that space. VLM-specific PTQ already includes modality-, token-, layer-, Hessian-, gradient-, and expert-aware methods. Query-guided visual-token selection is also well established.

## Remaining Testable Gap

The literature review did not identify a study that simultaneously:

1. fixes the image and changes only the query;
2. intervenes on quantization precision across the vision encoder, projector, language-model attention, and language-model MLP rather than only KV cache or visual tokens;
3. tests whether the sensitivity difference is larger than decoding and measurement noise;
4. demonstrates that the difference changes the best profile under a measured hardware budget; and
5. compares against static VLM allocation and query/task/input-adaptive LLM baselines.

This intersection is the current research gap. It is not yet an established scientific fact.

## Required Evidence

- Same-image, multi-query counterfactual sensitivity with image-grouped confidence intervals.
- Per-query oracle headroom over best static and per-task profiles.
- Interaction features that outperform image-only, query-only, and task-only controls.
- A small profile codebook that preserves the oracle benefit.
- End-to-end A800 results covering TTFT, TPOT, throughput, peak memory, routing, batching, and profile-switching costs.

## Stop Condition

Stop the main direction if per-query oracle improvement is below 0.5 absolute points and measured cost improvement is below 5%, or if task ID, answer length, image resolution, and prompt template fully explain the observed sensitivity variation.

See [`../literature/surveys/query_aware_vlm_quantization_literature_review.md`](../literature/surveys/query_aware_vlm_quantization_literature_review.md) for the evidence table and collision audit.
