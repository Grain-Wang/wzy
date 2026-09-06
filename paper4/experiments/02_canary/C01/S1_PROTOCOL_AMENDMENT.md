# C01-S1 Protocol Amendment

Status: **Frozen before any C01-S1 W4 output.**

Date: 2026-09-06

## Scope

This is the single protocol-correctness amendment allowed before the S1 W4
dynamic-range gate. It does not change the C01 hypothesis, sample based on model
outputs, alter the locked W4A16 definition, change module groups, budgets,
metrics, or any positive/negative threshold in the C01 preregistration.

## Official GQA score

The primary end-task metric is now implemented as the GQA standard accuracy:
one point when the serialized prediction exactly matches the unique reference
answer and zero otherwise. The official GQA evaluation description specifies
this exact-match rule, and the released evaluation code compares
`predicted == gold`.

For the generative Qwen checkpoint, output serialization is fixed to stripping
leading/trailing whitespace and lowercasing. No punctuation, article, number,
stemming, substring, semantic, or sentence-answer heuristic is applied. The S0
VQA-style normalized exact metric remains a non-primary audit metric and is not
reported as official GQA accuracy.

Primary source: <https://cs.stanford.edu/people/dorarad/gqa/evaluate.html>

## Prompt correction

The S0 instruction requested a short phrase, but its frozen smoke-test outputs
frequently contained complete explanatory sentences. Before S1-B and without
trying alternatives, S1 locks the following stronger answer-vocabulary prompt:

> Answer the question using only the lowercase answer (one word or a short
> phrase). Do not write a sentence or explanation.

The chat template, checkpoint, processor, decoding mode, and 16-token generation
cap remain unchanged. The exact prompt/template hash is stored in every frozen
manifest row. Samples will not be removed based on BF16 correctness.

## Pre-output gate definitions

Because the original C01 protocol defined the semantic baseline/proxy gates but
did not assign numeric operational cutoffs, `paper4/configs/c01_s1.json` freezes
them before S1-A and S1-B output:

- Baseline valid: finite NLL for all samples, empty rate at most 1%, malformed
  rate at most 5%, overall official accuracy at least 10%, and at least three
  adequately supported families with official accuracy at least 5%.
- Proxy has no dynamic range only when all-W4 changes official accuracy by less
  than 0.5 point, absolute mean NLL by less than 0.01, mean JS by less than
  `1e-4`, and flips fewer than 1% of answers.
- Proxy is catastrophic when official accuracy is below 5% or below 20% of the
  BF16 score. Otherwise a measurable, non-catastrophic change passes S1-B.

These operational definitions do not replace or relax the preregistered S1
STRONG PASS, NEGATIVE, or INCONCLUSIVE decision rules.
