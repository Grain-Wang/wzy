"""Prepare the bounded C01 replication-depth manifest (no inference)."""
from __future__ import annotations

import json
from pathlib import Path

from paper4.src.ciq_s1.expansion import replication_distribution, select_expansion_rows


def main() -> None:
    root = Path(__file__).resolve().parents[4]
    old_path = root / "paper4/results/raw/C01/S1/candidate_manifest.jsonl"
    rows_path = root / "paper4/cache/gqa_expansion_rows.jsonl"
    old = [json.loads(line) for line in old_path.read_text().splitlines() if line]
    if rows_path.exists():
        rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    else:
        from datasets import load_dataset

        rows = list(load_dataset("parquet", data_files=str(root / "paper4/cache/gqa_snapshot/testdev_balanced_instructions/testdev-00000-of-00001.parquet"), split="train"))
    print("loaded", len(rows), "old", len(old))
    selected = select_expansion_rows(rows, {r["question_id"] for r in old}, questions_per_cell=3)
    out = root / "paper4/results/raw/C01/S1_EXPANSION/expansion_rows.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in selected))
    print(json.dumps(replication_distribution(selected), sort_keys=True))


if __name__ == "__main__":
    main()
