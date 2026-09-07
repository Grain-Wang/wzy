# C01-S1 Bounded Sample Expansion Protocol

Status: pre-registered before any new quantized output.

This is the single authorized bounded expansion after the S1 inconclusive result. Its sole rationale is a measurement limitation: most retained image×query-family cells contained only two questions, so leave-one-query-out interaction prediction used one same-cell neighbour and could be dominated by question-specific residual noise.

The expansion tests whether increasing within-cell replication depth makes the image×family interaction stable, positive, and predictive on held-out questions. It is not a power-seeking threshold change, proxy correction, benchmark replication, router experiment, block refinement, or S2.

The original S1 image pool and metadata taxonomy are reused where possible. Every retained image×family cell must contain exactly four questions when feasible; otherwise the largest uniformly feasible depth of at least three is selected before any model output is inspected. Selection is deterministic from official GQA metadata and question identifiers, without correctness or sensitivity filtering.

Model, prompt, W4A16 diagnostic quantizer, 13 coarse groups, metrics, and the original conjunctive STRONG PASS gate remain unchanged. Existing S1 outputs are reused only when all available identity hashes and locked metadata match exactly. New inference is limited to questions absent from the frozen expansion manifest's reusable S1 rows.

Primary comparison: the original two-support estimator (one same-cell neighbour) versus the full expanded-support estimator (all other questions in the cell), with optional three-support analysis. The final decision is PASS only if the original STRONG PASS conjunctive gate is met; otherwise CIQ-PP is FAIL and no further expansion is permitted.
