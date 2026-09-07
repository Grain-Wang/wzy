# RefCOCOg BBox Schema Audit

The public RefCOCOg row schema was checked against `raw_anns` on a real sample. The mirror stores top-level `bbox` as **[x1,y1,x2,y2]**, while `raw_anns["bbox"]` stores COCO **[x,y,width,height]**. Example observed: top-level `[213.72,456.51,405.73,590.26]`; raw `[213.72,456.51,192.01,133.75]`; differences are `405.73-213.72=192.01` and `590.26-456.51=133.75`.

The parser now has explicit `parse_bbox(..., xyxy)` conversion and rejects invalid ordering/out-of-bounds boxes. A corrected 5,000-row rerun was attempted, but the public datasets-server returned HTTP 429 before completing the scan. Therefore no corrected aggregate rate is asserted in this run.
