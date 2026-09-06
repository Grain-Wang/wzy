"""Deterministic answer normalization and automatic VQA scoring."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence

_ARTICLES = frozenset({"a", "an", "the"})
_NUMBER_WORDS = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_PUNCTUATION_TABLE = str.maketrans({character: " " for character in string.punctuation})


def normalize_answer(text: str) -> str:
    """Normalize a short VQA answer using fixed punctuation/article/number rules."""
    lowered = text.lower().replace("\n", " ").replace("\t", " ")
    tokens = lowered.translate(_PUNCTUATION_TABLE).split()
    normalized = [
        _NUMBER_WORDS.get(token, token) for token in tokens if token not in _ARTICLES
    ]
    return re.sub(r"\s+", " ", " ".join(normalized)).strip()


def normalized_exact_score(prediction: str, reference: str) -> float:
    """Return deterministic normalized exact-match accuracy for one GQA answer."""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def vqa_soft_score(prediction: str, references: Sequence[str]) -> float:
    """Return the standard VQA soft score min(matches / 3, 1)."""
    if not references:
        raise ValueError("references must not be empty")
    normalized_prediction = normalize_answer(prediction)
    counts = Counter(normalize_answer(reference) for reference in references)
    return min(counts[normalized_prediction] / 3.0, 1.0)
