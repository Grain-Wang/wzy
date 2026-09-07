# CausalQuant-VL External Collision Matrix

Status labels follow the source venue: published/accepted when an official proceedings page exists; otherwise preprint or under-review. “Why Not Same” records the methodological distinction, not a claim of priority.

| Paper | Year | Venue | Published Status | Quantization? | Counterfactual Intervention? | Visual Reliance? | Relevant vs Irrelevant Control? | BF16-vs-LowBit Comparison? | Dependency Preservation Objective? | Method Type | Direct Collision Risk | Why Not Same |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| Q-VLM | 2024 | arXiv | preprint | Yes | No | No | No | No | No | modality-aware PTQ | Medium | protects modality quantization, no behavioral intervention |
| MBQ | 2024 | NeurIPS/arXiv | published | Yes | No | No | No | No | No | modality-balanced PTQ | Medium | reconstruction/bit allocation |
| VLMQ | 2024 | arXiv | preprint | Yes | No | No | No | No | No | multimodal PTQ | Medium | accuracy/reconstruction |
| TLQ | 2024 | arXiv | preprint | Yes | No | No | No | No | No | token-aware PTQ | Medium | token sensitivity, no counterfactual response |
| VLM-PTQ | 2026 | CVPR | published | Yes | No | No | No | No | No | PTQ | Medium | feature/statistical calibration |
| QIG | 2026 | CVPR | published | Yes | No | No | No | No | No | integrated-gradient PTQ | Medium | gradient importance |
| MABA | 2024 | arXiv | preprint | Yes | No | No | No | No | No | modality-aware adaptation | Medium | no causal image intervention |
| VVSQ | 2026 | Neurocomputing | published | Yes | No | vision vulnerability | No | No | No | vulnerability-guided PTQ | High | blind-but-fluent diagnosis, not paired dependency preservation |
| GRACE | 2025/26 | arXiv | preprint | Yes | No | relational proxy | No | No | No | gated relational distillation | High | feature relation, not intervention response |
| AWQ | 2023 | MLSys/arXiv | published | Yes | No | No | No | No | No | activation-aware PTQ | Low | unimodal calibration |
| GPTQ | 2023 | ICLR | published | Yes | No | No | No | No | No | Hessian PTQ | Low | generic LLM PTQ |
| QuaRot | 2024 | ICLR | published | Yes | No | No | No | No | No | rotation PTQ | Low | generic quantization |
| QAQ | 2024 | arXiv | preprint | Yes | No | No | No | No | No | query-conditioned bits | High | runtime/query precision, excluded by internal rule |
| TAQ | 2024 | arXiv | preprint | Yes | No | No | No | No | No | task-conditioned bits | High | task precision routing |
| DP-LLM | 2024 | arXiv | preprint | Yes | No | No | No | No | No | input/runtime precision | High | dynamic precision |
| PMPD | 2025 | arXiv | preprint | Yes | No | No | No | No | No | prompt/phase precision | High | prompt-adaptive schedule |
| MixKVQ | 2025 | arXiv | preprint | Yes | No | No | No | No | No | KV quantization | Medium | KV-only and no causal dependency |
| WindowQuant | 2025 | arXiv | preprint | Yes | No | prompt relevance | No | No | No | visual-window KV precision | High | prompt-aware KV policy |
| Treble Counterfactual VLMs | 2025 | arXiv | preprint | No | Yes | Yes | Partial | No | No | causal hallucination intervention | High | counterfactual causal evaluation, not quantization |
| Pixels Versus Priors | 2025 | arXiv | preprint | No | Yes | Yes | Partial | No | No | visual-counterfact analysis | High | measures prior/evidence transition in FP models |
| Chest VLMs Do Not Always Need the Image | 2025 | arXiv/GitHub | preprint | No | Yes | Yes | Yes | No | No | image-ablation audit | High | paired reliance audit, no low-bit objective |
| CC-VQA | 2025 | Neural Networks | published | No | Yes | Yes | Partial | No | No | causal VQA debiasing | Medium | causal data/model intervention, no quantization |
| CELLO | 2024 | EMNLP | published | No | Yes | Causal | No | No | No | causal evaluation benchmark | Medium | general causal VLM benchmark |
| ViCrit | 2025 | NeurIPS | published | No | Synthetic edits | Visual perception | No | No | No | verifiable RL proxy | Low | trains perception critic, not quantization |
| CF-VLM | 2025 | OpenReview | under review/preprint | No | Yes | Yes | Partial | No | No | counterfactual fine-tuning | High | counterfactual training but not low-bit preservation |
| M2CQA | 2026 | arXiv | preprint | No | Yes | Hallucination | Yes | No | No | counterfactual hallucination benchmark | Medium | language/culture controls, no quantization |
| Vision-Language Introspection | 2025 | OpenReview | under review | No | Yes | Yes | Partial | No | No | introspection/intervention | Medium | model introspection, no PTQ |
| Enhancing Visual Reliance | 2025 | arXiv | preprint | No | Yes | Yes | Partial | No | No | Bayesian reliance training | Medium | reliance mitigation, no quantized model |

## Strongest-collision conclusions

VVSQ occupies the “quantization causes blind-but-fluent visual degradation” observation and vulnerable-component PTQ, but reports conventional accuracy/reliability and scale divergence rather than a matched relevant/irrelevant counterfactual estimand. GRACE occupies relational feature alignment/distillation, not equality of behavioral intervention responses. Counterfactual-VLM papers occupy visual-reliance estimands and interventions in full precision, while quantized-reliability work occupies quality, confidence, or modality sensitivity. The remaining gap is therefore narrow and conditional: quantization-specific change in a counterfactual visual-dependence response plus a static objective to preserve that response.
