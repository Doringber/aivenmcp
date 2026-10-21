"""Deterministic stand-ins for the ONNX models, so tests run fast and offline."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Optional

_WORD = re.compile(r"\w+")


class WordCounter:
    """Token counter that treats every whitespace-separated word as one token."""

    def count(self, text: str) -> int:
        return len(text.split())


class HashingEmbedder:
    """Bag-of-words hashed into a small normalized vector.

    Texts sharing words get similar vectors, which is enough to exercise the vector branch
    of the store with real pgvector operators.
    """

    model_name = "test-hashing-embedder"
    dimension = 64

    def __init__(self, fail_on: Optional[str] = None) -> None:
        self._fail_on = fail_on

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if self._fail_on and any(self._fail_on in t for t in texts):
            raise RuntimeError("simulated embedding failure")
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for word in _WORD.findall(text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        if norm == 1.0 and not any(vector):
            vector[0] = 1.0
        return [v / norm for v in vector]
