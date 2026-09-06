# Related Work

## VLM Post-Training Quantization

Cover Q-VLM, MBQ, VLMQ, TLQ, VLM-PTQ, MABA, VVSQ, VEQ, and MODE. The key boundary is that most policies are calibrated offline and remain fixed at inference.

## Dynamic and Mixed-Precision Quantization

Cover instance-aware dynamic quantization, Bit-Mixer, USDN, Any-Precision LLM, QAQ, PAQ, TAQ, DP-LLM, PMPD, and ScaleBITS. These works prevent any broad first-use claim for query-, task-, sample-, token-, or phase-adaptive precision.

## Query-Adaptive VLM Computation

Cover QuietPrune, IVTP, LVPruning, SparseVLM, MADTP, SparseVILA, and query/model routing. These works support query-dependent visual relevance but do not directly establish non-KV quantization sensitivity.

## KV Cache and Visual-Prefix Compression

Cover KIVI, KVQuant, MiKV, ZipCache, MixKVQ, A2ATS, WindowQuant, visual KV quantization, ZipVL, and PM-KVQ. Query-aware KV precision is prior art and is not the proposed headline.

## Hardware-Aware Serving

Cover TVM, vLLM, SGLang, FlashInfer, Atom, QServe, MARLIN, QFactory, and FLUTE. The evaluation must include packing, dequantization, graph capture, batching, and routing overhead.

The annotated source table is in [`../literature/surveys/query_aware_vlm_quantization_literature_review.md`](../literature/surveys/query_aware_vlm_quantization_literature_review.md).
