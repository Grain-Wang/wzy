# C01-S0 Integrity Smoke Test Report

Execution date: 2026-09-06 (Asia/Shanghai)

## Outcome

**PASS. Recommendation: ENTER S1 only after explicit authorization. S1 was not run.**

This result establishes that the C01 experimental infrastructure is internally
consistent. It is not evidence that query-dependent sensitivity exists, that
CIQ-PP is effective, or that any quality, memory, or latency improvement has
been achieved.

## Locked Model and Environment

- Model: `Qwen/Qwen2-VL-2B-Instruct`
- Model revision: `895c3a49bc3fa70a340399125c650a463535e71c`
- Processor/tokenizer revision: `895c3a49bc3fa70a340399125c650a463535e71c`
- Model shard SHA-256:
  - `994ac2b03f97de8bc647d0fe5eba2e4b632b3e28dc03574c29bdfc36cf47e1b9`
  - `92540d8353c8d226a589a3b179bdb33851c970ee2cc2ac7ba035f79425e7b833`
- Model dtype: BF16; attention implementation: eager
- Python: 3.12.13
- PyTorch: 2.5.0+cu121; CUDA runtime: 12.1; cuDNN: 9.1.0
- Transformers: 4.55.2; Datasets: 4.0.0
- GPU: NVIDIA A800 80GB PCIe, one visible device
- Determinism: seed 20260906, deterministic algorithms enabled,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 disabled, greedy decoding
- Image processor: slow processor, RGB, `min_pixels=200704`,
  `max_pixels=401408`
- Generation: `do_sample=false`, one beam, `max_new_tokens=16`, KV cache BF16

The checkpoint was loaded from an exact local snapshot, so Transformers reports
no Hub `_commit_hash`; the source revision and both shard hashes above are the
binding identity.

## Dataset and Frozen Sample Manifest

- Dataset: `lmms-lab/GQA`, revision
  `a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8`
- Slice: balanced test-dev
- Images: 8
- Questions: 32, four questions per image
- Query-family counts: global perception 4; fine-grained recognition 10;
  spatial relation 12; reasoning 6
- Same-image preprocessing comparisons: 24; every comparison used identical
  image bytes, pixel tensors, grid metadata, and preprocessing hash
- Sample manifest SHA-256:
  `cfea83045278c17d8f1428b381ef1bee181fc69bf3a128ef41165b38ec3bdc05`
- Selection was frozen before any quantized output, did not use model outputs,
  and was not filtered after inference.
- Execution order shuffled whole image groups with the locked seed; it never
  split an image group to manufacture independent samples.

The frozen manifest and selected images remain local experimental artifacts.
They are excluded from Git; only the hash and small integrity summary are
committed.

## Quantizer Specification

- Diagnostic W4A16, weight-only; activations and KV cache remain BF16
- Symmetric per-group RTN over flattened contiguous groups of 128 weights
- Integer range: [-7, 7]
- Rounding: PyTorch round-to-nearest-even
- Scale: `max(abs(group))/7`, stored/calculated in FP32
- Clipping: no percentile clipping; integer saturation only
- Eligible tensors: floating matrix weights in the 13 locked non-KV groups
- Excluded: embedding tables, normalization, RoPE, output/tied head, biases,
  non-linearities, and tensors outside the locked groups
- Diagnostic implementation stores reconstructed values in BF16. The logical
  packed W4 byte count is reported, but S0 makes no physical compression or
  runtime-speed claim.

## Module Manifest Summary

- Manifest SHA-256:
  `db080e473fb6f9fa4204b1372ace0a5793644a71f41566a99b1847f696e705f6`
- Groups: 13
- Vision layers: 32, divided into four depth quartiles
- Bridge: one merger group
- Decoder layers: 28, divided into four attention and four MLP quartiles
- Quantizable tensors: 326
- Quantizable parameters: 1,973,420,032
- BF16 bytes for quantizable weights: 3,946,840,064
- Logical packed W4 bytes including FP32 scales: 1,048,379,392

The full runtime-discovered module/tensor manifest is retained in the ignored
raw artifact directory.

## Integrity Tests

| Test | Result | Recorded check |
|---|---|---|
| BF16 repeatability | PASS | 4 repeated cases; normalized/generated answers identical; maximum NLL difference 0 |
| Empty-group identity | PASS | 0 changed tensors; generated answer and NLL exactly unchanged |
| Single-group isolation | PASS | `decoder.q1.attention`; 28/28 target tensors changed; 0 non-target tensors changed |
| Restore identity | PASS | quantize -> restore recovered exact BF16 tensor hashes, answer, and NLL |
| All-W4 application | PASS | 326/326 eligible tensors changed; 32/32 generations completed; 32/32 NLL values finite |
| Evaluator correctness | PASS | 6 normalization, exact-score, and soft-score fixtures passed |
| Teacher-forced NLL | PASS | prompt positions masked; answer positions aligned; 32/32 BF16 NLL values finite; target token hashes identical across conditions |
| Same-image grouping | PASS | 8 image groups and 24 repeated-query comparisons; visual preprocessing identical within image |
| Vision-cache equivalence | PASS | identical BF16 cache hit equals direct result; quantized vision fingerprint misses BF16 cache; restored features equal BF16 |
| Manifest freeze/leakage | PASS | manifest frozen pre-quantization; no model-output selection or post-hoc filtering |

Machine-readable evidence is in
`paper4/results/raw/C01/S0/integrity_results.json` and
`paper4/results/processed/C01/S0/integrity_summary.json`.

## GPU Time and Storage

- Successful measured S0 GPU interval: 245.11 seconds = 0.0681 A800 GPU-hour
- Conservative cumulative total including two early integrity-gate failures:
  less than 0.10 A800 GPU-hour
- Peak memory allocated by the successful process: 5,193,116,160 bytes
- Remote S0 workspace: 11,692,940,671 bytes (11.69 GB decimal), including the
  isolated Python environment, exact model snapshot, minimal GQA snapshot,
  caches, selected images, and results
- Raw result artifacts: 264,028 bytes; processed summary: 971 bytes payload
- No model, dataset image, cache, credential, or SSH material is tracked by Git.

The selected GPU already had unrelated long-running compute. S0 stopped and
released its allocation after completion; pre/post checks showed the same
pre-existing GPU0 memory footprint. Therefore this run cannot support latency
or throughput claims.

## Bugs Found and Fixed

1. GQA exposes a `semantic` functional-program field as well as nested
   `types.semantic`. The initial classifier could consult the wrong field.
   Nested official type metadata now has priority, with a regression test.
2. Transformers 4.55 wraps decoder layers under
   `model.language_model.layers.*`, unlike the checkpoint index names. Runtime
   discovery now recognizes the actual wrapper and was validated against the
   loaded 32-layer vision / 28-layer decoder structure.
3. Deterministic CUDA matmul requires a cuBLAS workspace setting. S0 now refuses
   to start without one of the supported deterministic values and records the
   selected value.

Both failed attempts stopped before sensitivity output was generated. No
scientific threshold or quantization definition was changed.

## Automated Code Checks

- Black 25.1.0: 15 C01 source/test/entry files pass `--check`.
- Ruff 0.12.7: all C01 files pass; the local `paper4` tree also passes.
- Pytest 8.4.1: 16 tests passed in the isolated Python 3.12 environment.
- The repository-wide multi-paper test suite was not run; validation was
  intentionally scoped to `paper4` because other papers use independent legacy
  environments and are outside C01-S0.

## Remaining Concerns

- The tiny S0 slice produced only 4/32 BF16 normalized-exact matches. This is
  not an S0 gate or a scientific estimate, but S1 must treat baseline dataset /
  prompt validity as an early stop check before interpreting sensitivity.
- Transformers emits a future-version deprecation warning for the checkpoint's
  video-processor config filename. Image preprocessing completed correctly and
  remained identical within image groups.
- W4 bytes are a diagnostic logical proxy; a later hardware gate is still
  required before any deployability, memory, or latency claim.

## Recommendation

**ENTER S1 only when separately authorized.** Begin S1 with its registered
baseline-validity and proxy-dynamic-range checks. Do not train a router, build a
runtime switching backend, or infer support for CIQ-PP from S0.
