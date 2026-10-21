"""The inference gate must bound concurrent ONNX runs.

Unbounded concurrent runs each keep their own activation buffers; under 16 parallel
searches the server grew to ~10 GB RSS and stalled a 4-core host.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from aiven_rag_mcp.embeddings import FastEmbedEmbedder, FastEmbedReranker


class _ConcurrencyProbe:
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()

    def work(self) -> None:
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.02)
        with self._lock:
            self.active -= 1


class _FakeModel:
    def __init__(self, probe: _ConcurrencyProbe) -> None:
        self._probe = probe

    def embed(self, texts: Sequence[str], batch_size: int = 1) -> Iterator[np.ndarray]:
        self._probe.work()
        for _ in texts:
            yield np.zeros(1)

    def rerank(self, query: str, documents: Sequence[str], batch_size: int = 1) -> list[float]:
        self._probe.work()
        return [0.0 for _ in documents]


def _embedder(limit: int, probe: _ConcurrencyProbe) -> FastEmbedEmbedder:
    embedder = FastEmbedEmbedder("Xenova/bge-small-en-v1.5", max_concurrency=limit)
    embedder._model = _FakeModel(probe)
    return embedder


def _reranker(limit: int, probe: _ConcurrencyProbe) -> FastEmbedReranker:
    reranker = FastEmbedReranker("fake", max_concurrency=limit)
    reranker._model = _FakeModel(probe)
    return reranker


@pytest.mark.parametrize("limit", [1, 2])
@pytest.mark.parametrize(
    ("factory", "call"),
    [
        (_embedder, lambda m: m.embed_query("q")),
        (_embedder, lambda m: m.embed_documents(["a", "b"])),
        (_reranker, lambda m: m.score("q", ["a", "b"])),
    ],
    ids=["embed_query", "embed_documents", "rerank"],
)
def test_inference_concurrency_is_bounded(limit: int, factory, call) -> None:  # type: ignore[no-untyped-def]
    probe = _ConcurrencyProbe()
    model = factory(limit, probe)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: call(model), range(16)))
    assert probe.peak == limit
