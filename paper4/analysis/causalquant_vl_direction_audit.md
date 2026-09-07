# CausalQuant-VL Direction Audit

## 1. Executive Decision

**CONDITIONAL GO — Research Opportunity / Canary Design only.** The problem is scientifically distinct from the internal hardware-adaptive roadmap, but the measurement idea is not collision-free: visual-reliance and counterfactual-causality evaluations already exist, and VVSQ/GRACE already target quantized VLM visual degradation. C02 must test the quantization-specific interaction before any method work.

## 2. Internal Non-Overlap Audit

| Dimension | PhD roadmap | CausalQuant-VL |
|---|---|---|
| Main problem | adaptive compute allocation | quantization-induced modality-dependence distortion |
| Decision variable | token K / precision p | static quantized-weight calibration/objective |
| Runtime adaptation | yes | no |
| Hardware budget | core | secondary, not a claim |
| Query-aware routing | core | not required |
| Token allocation | core | excluded |
| Mixed precision | core | excluded as main method |
| Main evidence | quality–cost Pareto | counterfactual visual-dependence preservation |
| Main method | controller/search | dependency-preserving quantization |

Internal overlap is low only under these exclusions; renaming mixed precision or routing would be an immediate NO-GO.

## 3. External Collision Search

The audit searched VLM PTQ, multimodal quantization, visual reliance, counterfactual VLM evaluation, hallucination/language-prior, and cross-modal alignment literature (2024–2026). Directly relevant evidence includes VVSQ's “Blind but Fluent” quantization failure framing [VVSQ](https://www.sciencedirect.com/science/article/pii/S0925231226022861), counterfactual causal hallucination intervention [Treble Counterfactual VLMs](https://arxiv.org/abs/2503.06169), and controlled image-intervention visual-reliance evaluation [Vision-language models for chest radiography do not always need the image](https://github.com/mahshadlotfinia/causal). Pixels Versus Priors explicitly studies visual counterfacts and prior/evidence transitions [paper](https://arxiv.org/abs/2505.17127).

## 4. Nearest Papers

The relevant comparison set is: Q-VLM, MBQ, VLMQ, TLQ, VLM-PTQ, MABA, VVSQ, GRACE, AWQ, GPTQ, QuaRot, QAQ, TAQ, DP-LLM, PMPD, MixKVQ, WindowQuant, Treble Counterfactual VLMs, Pixels Versus Priors, Vision-language models for chest radiography do not always need the image, CC-VQA, ViCrit, Visual Language Model Introspection, QuietPrune, and SparseVLM. The quantization papers mostly optimize reconstruction, Hessian/gradient sensitivity, modality/layer protection, or distillation; the counterfactual papers measure or alter visual reliance without studying low-bit weight quantization. VVSQ and GRACE are the strongest collision because they explicitly report visual degradation after quantization and train/align quantized VLMs, respectively.

## 5. What Is Already Occupied

Occupied: generic accuracy degradation; blind-but-fluent behavior as an observation; modality/token/layer sensitivity; confidence-based rescue; representation alignment/distillation; counterfactual visual-reliance evaluation in full precision; causal debiasing of VQA; mixed-precision protection of important layers; runtime precision routing.

## 6. Remaining Scientific Gap

No located work directly tests, under matched image–question pairs, whether *the change induced by low-bit quantization in the original-versus-relevant-counterfactual response* differs from its change under an irrelevant-region control, and treats preservation of the BF16 counterfactual response as a quantization objective. This is a provisional gap, not a priority claim of firstness.

## 7. Precise Core Hypothesis

For (x=(I,Q)), let (I_r) remove or occlude metadata/program-identified evidence and (I_u) perturb an irrelevant region. Define (E(M)=NLL(M,I_r,Q)-NLL(M,I,Q)) (primary), with score-drop and gold-position JS as secondary. The claim is that 
Δ = [E(W4,I_r)-E(W4,I_u)]-[E(BF16,I_r)-E(BF16,I_u)]
is consistently negative: W4 weakens dependence on relevant visual evidence beyond generic corruption sensitivity.

## 8. Why This Is Not VVSQ / GRACE / Reliability Selector

VVSQ diagnoses vision vulnerability and quantizes vulnerable components; GRACE uses gated relational alignment/distillation to improve low-bit quality; reliability selectors use confidence to trigger computation. CausalQuant-VL first measures a paired counterfactual dependency shift with an irrelevant-region control, and any later method would be a static dependency-response preservation objective—not a selector, router, or generic alignment loss.

## 9. Candidate Algorithmic Space

Only after a positive C02: (A) static counterfactual dependency-preserving calibration, adding a paired response-consistency loss to PTQ; (B) evidence-necessity-weighted reconstruction/Hessian calibration. Neither is authorized now.

## 10. C02 Canary Design

Use Qwen2-VL-2B-Instruct, BF16 versus locked diagnostic W4A16, on 50–100 automatically structured GQA/RefCOCO-style images and 2–4 queries per image spanning attributes, relations, recognition, and reasoning. Use original image, metadata/program-derived relevant evidence removal/occlusion, and equal-area irrelevant-region occlusion; optionally matched distractor images. Freeze image/query/evidence manifests before inference. Run deterministic answer-only NLL, normalized/exact score, gold-position JS, answer flips, and output length.

Primary metric is the paired NLL interaction Δ above; official answer-score drop and JS are secondary high-resolution checks. Analyze with image-grouped mixed effects and paired/image bootstrap plus within-image permutation.

## 11. Positive Criterion

Direction-consistent interaction, meaningful pre-registered effect, replication in at least two query families, no comparable irrelevant-control shift, and simultaneous support from one end-task metric and one high-resolution metric. p-values alone are insufficient.

## 12. Negative Criterion

W4 and BF16 relevant-evidence dependence are indistinguishable; the shift is fully explained by generic corruption; relevant and irrelevant controls are equivalent; or the effect occurs in only one family. Stop the direction.

## 13. Single-A800 Feasibility

Reuse the validated Qwen2-VL-2B, processor, W4, NLL/JS and deterministic infrastructure. A 50–100 image paired probe is estimated at 1–3 A800 GPU-hours, with under 10 GB processed artifacts and no full benchmark.

## 14. Publication Potential

Conditional weak CCF-B/strong CCF-C potential only if the quantization-specific interaction is robust and a static objective improves it against VVSQ/GRACE/PTQ baselines. A measurement-only positive result is not sufficient.

## 15. Risks

Evidence-region construction may be noisy; counterfactual edits can introduce corruption; existing visual-reliance papers may cover the exact metric; model-scale generalization is unknown; NLL shifts may not translate to official scores.

## 16. Final GO / NO-GO

**CONDITIONAL GO** for one C02 canary only. Method development: NO.
