# C02 Dataset Pivot Feasibility

## Comparison

| Criterion | RefCOCO | RefCOCOg |
|---|---:|---:|
| Grounded bbox availability | annotation-level yes | annotation-level yes |
| Expressions | 142,209 reported | 85,474 reported |
| Images | 19,994 reported | 26,711 reported |
| Multiple queries/image | yes | yes |
| Multiple queries/object | yes | yes |
| Query richness | shorter (3.5-word mean) | richer (8.4-word mean) |
| Attribute / spatial coverage | yes | yes |
| Compositional coverage | limited but present | stronger |
| Irrelevant-control success | unverified | unverified |
| Area-match quality | unverified | unverified |
| Center/edge confound | unverified | unverified |
| C02 suitability | promising, not authorized | promising, likely preferred |

## Decision

Both candidates provide the crucial target-object grounding annotation, unlike the current GQA snapshot. RefCOCOg is the preferred primary candidate because its longer expressions provide richer relational/compositional language. Nevertheless, the required sample-level deterministic irrelevant-window search was not completed in this bounded run; therefore C02 remains unauthorized and the formal gate is **NO-GO pending completion of that CPU audit**. No GPU inference was run.
