#!/usr/bin/env python3
"""CLI for explicitly selected C01-S1 stages; never advances automatically."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PAPER_ROOT = REPOSITORY_ROOT / "paper4"
sys.path.insert(0, str(PAPER_ROOT))

from src.ciq_s1.runner import (  # noqa: E402
    prepare_s1,
    run_s1_profiles,
    run_s1_a,
    run_s1_b,
    run_s1_sweep,
)
from src.ciq_s1.analysis import analyze_s1  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse one explicitly requested C01-S1 stage."""
    parser = argparse.ArgumentParser(description="Run one locked C01-S1 stage")
    parser.add_argument(
        "stage",
        choices=("prepare", "baseline", "proxy", "sweep", "profiles", "analyze"),
    )
    parser.add_argument(
        "--config", type=Path, default=PAPER_ROOT / "configs" / "c01_s1.json"
    )
    return parser.parse_args()


def main() -> int:
    """Run only the selected stage and print its gate outcome."""
    args = parse_args()
    runners = {
        "prepare": prepare_s1,
        "baseline": run_s1_a,
        "proxy": run_s1_b,
        "sweep": run_s1_sweep,
        "profiles": run_s1_profiles,
        "analyze": analyze_s1,
    }
    summary = runners[args.stage](args.config.resolve(), REPOSITORY_ROOT)
    print(f"C01-S1 {args.stage}: {summary['outcome']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
