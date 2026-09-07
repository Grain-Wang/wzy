#!/usr/bin/env python3
"""CLI for the explicitly phased C02-S0 integrity smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PAPER_ROOT = REPOSITORY_ROOT / "paper4"
sys.path.insert(0, str(PAPER_ROOT))

from src.causalquant.s0_runner import run_phase  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse the single authorized S0 phase."""
    parser = argparse.ArgumentParser(description="Run one C02-S0 phase")
    parser.add_argument(
        "--config",
        type=Path,
        default=PAPER_ROOT / "configs" / "c02_s0.json",
    )
    parser.add_argument(
        "--phase",
        choices=(
            "freeze-manifest",
            "prepare-counterfactuals",
            "bf16-original",
            "remaining",
        ),
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    """Execute and print one compact JSON phase summary."""
    arguments = parse_args()
    summary = run_phase(arguments.config.resolve(), REPOSITORY_ROOT, arguments.phase)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
