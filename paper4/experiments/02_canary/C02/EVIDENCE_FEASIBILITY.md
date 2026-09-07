# RefCOCOg Offline Evidence Feasibility v2

## Environment and source

Execution used the repository-local Conda environment (`Python 3.12.14`, `pyarrow 25.0.0`) and only `paper4/cache/refcocog/train.parquet`. No network, model, or GPU call occurred. Parquet SHA-256 matches provenance: `6ab9631fb9353e94f440832f272ab15bae5726b62168e930a922aa687ed62b96`.

## BBox schema audit

All 5,000 scanned refs had valid top-level xyxy boxes and raw-annotation xywh consistency: schema consistency 100%, raw consistency 100%. The old 17.26% grounded rate is `INVALIDATED_BY_BBOX_SCHEMA_BUG`.

## Sentence-level scan

5,000 refs yielded 9,606 independent sentence expressions, 3,621 images and 4,068 referred target objects. Mean sentences/ref was 1.9212; median 2. Multiple-expression images: 4,083; multiple-expression target objects: 4,597. Known-object coverage is RefCOCOg-referred objects only, not complete COCO instances.

## Controls and geometry

Exact-shape Stage-A controls succeeded for 7,792/9,606 sentences (81.12%); 1,814 failed deterministic placement. Stage-B was not needed for the gate and produced 0 additional records. Area and aspect mismatches are exactly zero for Stage-A. Center-X SMD 0.071, center-Y SMD 0.461 (moderate concern), center-radius SMD −0.164, edge-distance SMD −0.008; no severe (>0.50) spatial confound. Matching is deterministic and uses fixed geometry cost/hash tie-breaking.

Semantic usable counts: attribute 3,895; spatial 2,156; relational 3,782; recognition 2,832; compositional 2,105.

## Decision

**GO.** All hard feasibility gates pass: schema and grounding ≥95%, controls ≥80%, usable sentences/images exceed thresholds, at least two semantic categories exceed 200, area mismatch is zero, and no severe spatial confound. C02 remains a separate authorization gate; this result authorizes protocol consideration, not GPU execution.
