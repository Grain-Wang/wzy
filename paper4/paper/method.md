# Method

## Status

The method is a candidate design, not a validated contribution.

## Counterfactual Sensitivity

For image \(I\), query \(q\), module group \(m\), and low-bit base profile, measure the output or task loss change after restoring \(m\) to a higher precision. Subtract the image-conditioned mean across queries to estimate the interaction residual rather than conflating image difficulty with query dependence.

## Hardware-Constrained Profile Codebook

Construct at most three or four precision profiles from backend-supported W4A16, W8A16, BF16, and supported KV formats. Optimize task loss and measured A800 cost jointly. Profiles must be module-aligned and prepacked; arbitrary per-token bit mosaics are excluded.

## Interaction Router

Use already-computed or cheaply available pooled visual and query features. Compare image-only, query-only, task-only, concatenated, and low-rank bilinear interaction routers. Train against cost-sensitive oracle-profile regret, not only profile classification accuracy.

## Optional Rescue

If a calibrated signal predicts quantization-induced disagreement, upgrade at the request or first-token boundary. Generic entropy alone is not sufficient evidence and must be compared against random rescue and an always-medium profile.
