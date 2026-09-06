"""Precision-safe cache keys and equivalence checks for visual features."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class VisionCacheKey:
    """Cache key that makes vision-weight precision part of cache identity."""

    image_sha256: str
    preprocessing_sha256: str
    vision_weights_sha256: str


class VisionFeatureCache:
    """Minimal in-memory cache that refuses cross-precision reuse by construction."""

    def __init__(self) -> None:
        self._entries: dict[VisionCacheKey, torch.Tensor] = {}

    def put(self, key: VisionCacheKey, features: torch.Tensor) -> None:
        """Store a detached clone for an exact key."""
        self._entries[key] = features.detach().clone()

    def get(self, key: VisionCacheKey) -> torch.Tensor | None:
        """Return a detached clone only for an exact key match."""
        value = self._entries.get(key)
        return None if value is None else value.detach().clone()


def image_sha256(path: Path) -> str:
    """Hash exact image-file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combine_tensor_hashes(hashes: dict[str, str]) -> str:
    """Combine sorted tensor hashes into one precision fingerprint."""
    digest = hashlib.sha256()
    for name, value in sorted(hashes.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
