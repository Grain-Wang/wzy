# C02 — Quantization-Induced Visual Dependence Shift

Status: protocol frozen; not run. Research Opportunity only.

## 1. Scientific Question / Hypothesis

Does uniform W4A16 change dependence on task-relevant visual evidence beyond an equally sized irrelevant corruption? Define `E_p^rel=NLL(p,I_rel,Q)-NLL(p,I,Q)` and `E_p^irr=NLL(p,I_irr,Q)-NLL(p,I,Q)`. Primary `Delta=(E_W4^rel-E_W4^irr)-(E_BF16^rel-E_BF16^irr)`. Delta < 0 means W4 weakens relevant visual dependence; >0 means amplification.

## 2. Model and Data

Qwen2-VL-2B-Instruct, locked S0 revision, deterministic BF16 versus locked uniform W4A16 (group 128, symmetric RTN, [-7,7], FP32 scales, BF16 activations/KV). Use 60 GQA test-dev images and 240 questions (four/image), selected before outputs across attribute, relation, fine-grained and compositional families. No human labels.

## 3. Automatic Evidence Region Construction

Parse GQA functional programs and scene graphs. Attribute questions use the queried object's box; relation/spatial questions use both endpoints; logical/compositional questions use the union of all object IDs on the minimal answer path. Map IDs to boxes, clip bounds, merge IoU>=0.30. Eligibility requires valid boxes and union area 1–60%.

```text
ids = answer_path_object_ids(parse(semantic_program))
boxes = merge_iou([scene_graph[id].bbox for id in ids], .30)
eligible = ids != empty and .01 <= union_area(boxes)/image_area <= .60
```

## 4. Counterfactual and Control

Expand each evidence box by 10% per side, clip, and fill with the 5-pixel border-ring median; union overlaps before filling. Generate irrelevant masks on a deterministic 16-pixel grid, matching area within 5%, component count and aspect ratios, avoiding all evidence/program boxes. Use the same fill/operator. Discard if no valid candidate in 1,000 positions. Record area, centroid, edge distance, perimeter, image statistics and require area SMD<0.2 between relevant/control.

## 5. Metrics and Statistics

Primary Delta above. Secondary official/normalized score drop, answer flips, gold-position JS and descriptive confidence margin. Fit image-grouped paired effects and Precision×EvidenceType interaction; 2,000 image bootstrap and 1,999 within-image permutations. Covariates: area, box count, dimensions, family, question/answer length, object count, centroid and edge distance.

## 6. Sample Size / Thresholds

Sixty images × four queries gives approximately 0.06 NLL detectable paired interaction at 80% power under cluster SD .20 and ICC .25 (planning assumption). PASS requires Delta≤−0.06 with 95% CI excluding 0, same direction in ≥2 families (≥30 queries/family), relevant-minus-irrelevant score interaction ≥2 points with CI excluding 0, and both end-task and JS support. FAIL requires |Delta|<.02 with CI inside [−.02,.02], score interaction <.5 point, artifact imbalance, or fewer than two replicated families. At most one bounded expansion to 100 images is allowed; no proxy/prompt search or method implementation.

## 7. Feasibility, Outputs, Decision

Before GPU execution, perform CPU-only evidence mapping and report eligible fraction, family coverage, area distributions and area-matched-control success. Hard GPU cap is 3 A800-hours. Freeze JSONL manifest and SHA-256 before inference; save raw paired outputs, processed estimand tables, plots and a decision report. Distinguish evidence-construction failure, generic corruption, absent quantization effect, metric-only effect and single-family effect. C02 requires separate authorization and must not be run during this audit.
