# Experiments

## Current Evidence

No model experiment has been completed. The literature gate recommends one canary before any full experimental program.

## Canary

Run Qwen2-VL-2B on same-image multi-query groups from GQA, VQAv2, and TextVQA. Measure module restoration sensitivity from a W4A16 base to W8/BF16, control confounds, and estimate per-query oracle profile headroom with image-grouped bootstrap.

## Planned Main Evaluation if the Canary Passes

- Models: Qwen2-VL-2B plus Qwen2.5-VL-3B or a 7B LLaVA/Qwen model.
- Query slices: global perception, fine-grained object/attribute, OCR, spatial/diagram, knowledge-intensive, and multi-step reasoning.
- Metrics: official task score, teacher-forced NLL/KL, TTFT, TPOT, tokens/s, peak memory, router latency, packing/capture memory, and batching loss.
- Baselines: BF16, uniform W8/W4, MBQ/MABA-style static allocation, TAQ-style task profiles, QAQ-style query routing, DP-LLM-style local selection, PMPD phase schedules, and MixKVQ/WindowQuant where applicable.
- Ablations: profile count, module grouping, feature source, interaction form, cost term, rescue signal, and profile-aware batching.
- Robustness: unseen images/tasks, resolution and length shifts, corruption/OOD slices, and independent seeds.

## Stop Rule

Do not proceed to the main evaluation if the same-image per-query oracle gain is below the thresholds in [`../CURRENT.md`](../CURRENT.md).
