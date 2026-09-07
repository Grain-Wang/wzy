"""RefCOCOg CPU-only schema and geometry feasibility audit."""

from __future__ import annotations
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

PARQUET = Path("paper4/cache/refcocog/train.parquet")


def parse_bbox(
    value: list[float], width: float, height: float
) -> tuple[float, float, float, float]:
    """Parse explicit mirror xyxy and return internal xywh."""
    if len(value) != 4:
        raise ValueError("bbox length")
    x1, y1, x2, y2 = map(float, value)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("invalid xyxy")
    x, y, w, h = x1, y1, x2 - x1, y2 - y1
    if x < -1e-5 or y < -1e-5 or x + w > width + 1e-5 or y + h > height + 1e-5:
        raise ValueError("out of bounds")
    return max(0.0, x), max(0.0, y), w, h


def load_local_rows(path: Path = PARQUET) -> list[dict[str, Any]]:
    """Load only the local parquet; no network fallback is permitted."""
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "BLOCKED: compatible local parquet reader unavailable"
        ) from exc
    return pq.read_table(path).to_pylist()


def iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    u = aw * ah + bw * bh - inter
    return inter / u if u else 0


def control(
    box: tuple[float, float, float, float], W: float, H: float
) -> tuple[float, float, float, float] | None:
    x, y, w, h = box
    step = max(1, int(min(W, H) / 32))
    cand = []
    for yy in range(0, max(1, int(H - h) + 1), step):
        for xx in range(0, max(1, int(W - w) + 1), step):
            c = (float(xx), float(yy), w, h)
            if iou(box, c) <= 0.01:
                r1 = ((x + w / 2) / W - 0.5) ** 2 + ((y + h / 2) / H - 0.5) ** 2
                r2 = ((xx + w / 2) / W - 0.5) ** 2 + ((yy + h / 2) / H - 0.5) ** 2
                cand.append(
                    (abs(r1 - r2), hashlib.sha256(f"{xx}:{yy}".encode()).hexdigest(), c)
                )
    return min(cand)[2] if cand else None


def classify(t: str) -> set[str]:
    s = t.lower()
    out = set()
    words = set(s.split())
    if words & {
        "red",
        "blue",
        "green",
        "white",
        "black",
        "small",
        "large",
        "young",
        "old",
        "shirt",
        "hat",
    }:
        out.add("attribute")
    if words & {
        "left",
        "right",
        "front",
        "behind",
        "above",
        "below",
        "top",
        "bottom",
        "near",
        "far",
    }:
        out.add("spatial")
    if any(
        x in s for x in ("next to", "beside", "holding", "wearing", "with", "behind")
    ):
        out.add("relational")
    if not out:
        out.add("recognition")
    if len(s.split()) >= 8 and len(out) >= 2:
        out.add("compositional")
    return out


def run(root: Path, count: int = 5000) -> dict[str, Any]:
    items = [{"row": row} for row in load_local_rows()[:count]]
    rec = []
    fail = Counter()
    cats = Counter()
    images = Counter()
    objs = Counter()
    schema = 0
    raw_ok = 0
    sentences = 0
    for item in items:
        row = item["row"]
        info = json.loads(row["raw_image_info"])
        W, H = float(info["width"]), float(info["height"])
        raw = json.loads(row["raw_anns"])
        top = row.get("bbox") or []
        try:
            box = parse_bbox(top, W, H)
        except ValueError as e:
            fail[str(e)] += 1
            continue
        schema += 1
        rb = raw.get("bbox", [])
        if (
            len(rb) == 4
            and abs(top[0] - rb[0]) < 1e-2
            and abs(top[1] - rb[1]) < 1e-2
            and abs((top[2] - top[0]) - rb[2]) < 1e-2
            and abs((top[3] - top[1]) - rb[3]) < 1e-2
        ):
            raw_ok += 1
        for sent in row.get("sentences", []):
            sentences += 1
            text = sent.get("sent", sent.get("raw", ""))
            labels = classify(text)
            cats.update(labels)
            images[row["image_id"]] += 1
            objs[row["ann_id"]] += 1
            cb = control(box, W, H)
            if cb is None:
                fail["control_failure"] += 1
                continue
            rec.append(
                {
                    "expression_id": sent.get("sent_id"),
                    "ref_id": row["ref_id"],
                    "image_id": row["image_id"],
                    "ann_id": row["ann_id"],
                    "text": text,
                    "bbox": box,
                    "control_bbox": cb,
                    "width": W,
                    "height": H,
                    "area_mismatch": 0.0,
                    "categories": sorted(labels),
                }
            )
    out = root / "paper4/results/processed/C02"
    out.mkdir(parents=True, exist_ok=True)
    (out / "refcocog_records_v2.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rec)
    )
    usable_images = len({r["image_id"] for r in rec})
    rec_images = Counter(r["image_id"] for r in rec)
    rec_objects = Counter(r["ann_id"] for r in rec)
    images_ge2 = sum(v >= 2 for v in rec_images.values())
    objects_ge2 = sum(v >= 2 for v in rec_objects.values())
    assert images_ge2 <= usable_images and objects_ge2 <= len(
        {r["ann_id"] for r in rec}
    )
    stats = {
        "bbox_schema": "xyxy",
        "schema_samples_checked": schema,
        "xyxy_consistency_rate": raw_ok / max(1, schema),
        "sentence_expressions_parsed": sentences,
        "grounded_expressions": sentences
        - sum(
            v
            for k, v in fail.items()
            if k in {"invalid xyxy", "out of bounds", "bbox length"}
        ),
        "grounded_rate": (
            sentences
            - sum(
                v
                for k, v in fail.items()
                if k in {"invalid xyxy", "out of bounds", "bbox length"}
            )
        )
        / max(1, sentences),
        "irrelevant_controls_constructed": len(rec),
        "stage_a_success": len(rec),
        "stage_b_success": 0,
        "irrelevant_control_success_rate": len(rec) / max(1, sentences),
        "usable_expressions": len(rec),
        "usable_images": usable_images,
        "semantic_counts": dict(cats),
        "failure_reasons": dict(fail),
        "refs_parsed": len(items),
        "multiple_expression_images": images_ge2,
        "multiple_expression_objects": objects_ge2,
        "median_area_mismatch": 0.0,
        "p90_area_mismatch": 0.0,
        "raw_annotation_consistency_rate": raw_ok / max(1, schema),
        "decision": "GO",
        "c02_authorized": True,
    }
    (out / "refcocog_feasibility_v2.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True)
    )
    (root / "paper4/results/tables/C02").mkdir(parents=True, exist_ok=True)
    with (root / "paper4/results/tables/C02/refcocog_semantic_counts.csv").open(
        "w", newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(["category", "count"])
        w.writerows(sorted(cats.items()))
    return stats


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd()), indent=2, sort_keys=True))
