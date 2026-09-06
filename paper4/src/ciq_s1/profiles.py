"""Exact-byte deterministic profile construction for C01-S1 controls."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .precision import profile_logical_bytes


@dataclass(frozen=True)
class PrecisionProfile:
    """One deterministic all-W4 restoration profile."""

    assignments: dict[str, str]
    logical_bytes: int
    added_bytes: int
    budget_bytes: int
    estimated_gain: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def all_w4_logical_bytes(manifest: Mapping[str, Any]) -> int:
    """Return exact all-W4 logical bytes from the runtime manifest."""
    return sum(int(group["w4_bytes"]) for group in manifest["groups"])


def restoration_budget_bytes(manifest: Mapping[str, Any], fraction: float) -> int:
    """Convert a relative resident-byte budget into an integer byte cap."""
    if fraction <= 0:
        raise ValueError("restoration budget fraction must be positive")
    return int(all_w4_logical_bytes(manifest) * fraction)


def _stable_tie(seed: int, name: str) -> str:
    return hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()


def construct_profile(
    manifest: dict[str, Any],
    group_gains: Mapping[str, float],
    *,
    budget_fraction: float,
    seed: int,
) -> PrecisionProfile:
    """Greedily pack sequential W4→W8→BF16 upgrades under exact byte costs."""
    groups = {str(group["group_id"]): group for group in manifest["groups"]}
    if set(group_gains) != set(groups):
        raise ValueError("group gains must cover exactly the runtime manifest groups")
    budget = restoration_budget_bytes(manifest, budget_fraction)
    remaining = budget
    assignments: dict[str, str] = {}
    estimated_gain = 0.0
    completed: set[tuple[str, str]] = set()
    while True:
        candidates: list[tuple[float, str, str, int, float]] = []
        for group_id, group in groups.items():
            gain = max(0.0, float(group_gains[group_id]))
            current = assignments.get(group_id, "w4")
            if current == "w4":
                cost = int(group["w8_bytes"]) - int(group["w4_bytes"])
                step_gain = 0.5 * gain
                action = "w8"
            elif current == "w8":
                cost = int(group["bf16_bytes"]) - int(group["w8_bytes"])
                step_gain = 0.5 * gain
                action = "bf16"
            else:
                continue
            key = (group_id, action)
            if key in completed or cost > remaining or cost <= 0 or step_gain <= 0:
                continue
            density = step_gain / cost
            candidates.append(
                (
                    density,
                    _stable_tie(seed, f"{group_id}:{action}"),
                    group_id,
                    cost,
                    step_gain,
                )
            )
        if not candidates:
            break
        _, _, group_id, cost, step_gain = max(
            candidates, key=lambda item: (item[0], item[1])
        )
        action = "w8" if assignments.get(group_id, "w4") == "w4" else "bf16"
        assignments[group_id] = action
        completed.add((group_id, action))
        remaining -= cost
        estimated_gain += step_gain

    logical_bytes = profile_logical_bytes(manifest, assignments)
    base = all_w4_logical_bytes(manifest)
    if logical_bytes - base > budget:
        raise AssertionError("constructed profile exceeds exact logical byte budget")
    return PrecisionProfile(
        assignments=assignments,
        logical_bytes=logical_bytes,
        added_bytes=logical_bytes - base,
        budget_bytes=budget,
        estimated_gain=estimated_gain,
    )


def deterministic_random_gains(
    group_ids: Sequence[str], *, seed: int, key: str
) -> dict[str, float]:
    """Produce output-independent random-control priorities reproducibly."""
    values: dict[str, float] = {}
    for group_id in group_ids:
        digest = hashlib.sha256(f"{seed}:{key}:{group_id}".encode()).digest()
        values[group_id] = int.from_bytes(digest[:8], "big") / float(2**64)
    return values


def additive_gain_for_assignments(
    assignments: Mapping[str, str],
    measured_group_gains: Mapping[str, float],
    *,
    w8_gain_fraction: float = 0.5,
) -> float:
    """Estimate realized gain from measured single-group gains, not routing scores."""
    if not 0.0 <= w8_gain_fraction <= 1.0:
        raise ValueError("W8 gain fraction must be between zero and one")
    gain = 0.0
    for group_id, precision in assignments.items():
        measured = max(0.0, float(measured_group_gains[group_id]))
        if precision == "w8":
            gain += w8_gain_fraction * measured
        elif precision == "bf16":
            gain += measured
        else:
            raise ValueError(f"unknown restoration precision: {precision}")
    return gain
