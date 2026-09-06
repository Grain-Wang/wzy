#!/usr/bin/env python3
"""CLI entry point for C01-S0 only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PAPER_ROOT = REPOSITORY_ROOT / "paper4"
sys.path.insert(0, str(PAPER_ROOT))

from src.ciq_s0.runner import run_s0, write_failure  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse the locked S0 command-line arguments."""
    parser = argparse.ArgumentParser(description="Run C01-S0 integrity checks only")
    parser.add_argument(
        "--config",
        type=Path,
        default=PAPER_ROOT / "configs" / "c01_s0.json",
    )
    return parser.parse_args()


def main() -> int:
    """Run S0 and persist a failure artifact when any gate fails."""
    args = parse_args()
    try:
        summary = run_s0(args.config.resolve(), REPOSITORY_ROOT)
    except BaseException as error:
        write_failure(
            repository_root=REPOSITORY_ROOT,
            config_path=args.config.resolve(),
            error=error,
        )
        raise
    print(
        f"C01-S0 {summary['outcome']}; recommendation={summary['recommendation']}; "
        f"gpu_hours={summary['gpu_hours']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
