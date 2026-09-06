# Novelty Matrix

| Candidate claim | Closest prior work | What prior work already covers | Remaining distinction required | Risk |
|---|---|---|---|---|
| Query-aware dynamic weight precision | QAQ, PAQ, DP-LLM | Query/prompt routing and runtime layer or model precision | Demonstrate image-query interaction and VLM non-KV module profiles | High |
| Task-aware layer bit allocation | TAQ, MABA, VVSQ | Task-conditioned or modality-sensitive static allocation | Per-instance residual beyond task and modality | High |
| Query-aware KV precision | MixKVQ, WindowQuant, A2ATS | Query-relevant K channels, visual-window KV bits, query-aware key VQ | Not a viable headline for this project | Very High |
| Prefill/decode asymmetric precision | PMPD, LLM-PQ, PM-KVQ | Phase-aware weight, placement, and KV policies | A VLM-specific visual-prefix mechanism with non-additive benefit | High |
| Query-aware token and precision co-design | QUOTA, QAPruner, Bi-VLM, ZipVL | Joint or staged token pruning and quantization | Prove a non-separable interaction and hardware benefit | Very High |
| Uncertainty-triggered precision rescue | CALM, TPT/C-TPT/ZERO, dynamic precision routing | Confidence-based compute allocation and test-time adaptation | Predict quantization-induced disagreement, not generic uncertainty | Medium--High |
| Same-image query-dependent non-KV sensitivity | No direct controlled study found | Adjacent evidence from query-guided tokens and LLM dynamic precision | Controlled intervention, confound removal, oracle headroom | Medium until measured |
| Hardware-constrained profile codebook | Any-Precision, ScaleBITS, MABA, QServe | Elastic storage, hardware-aligned allocation, fused low-bit kernels | Joint profile-codebook/router objective with real VLM serving cost | Medium--High |

## Claims That Must Not Be Used

- First query-aware quantization.
- First task-aware mixed-precision allocation.
- First runtime precision switching.
- First query-aware visual-token importance method.
- First prefill/decode asymmetric quantization.
- First combination of token pruning and quantization.

The defensible candidate is the conjunction of same-image counterfactual evidence, non-KV VLM module allocation, a small deployable profile codebook, and measured serving gains.
