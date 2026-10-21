"""Local embedding models (ONNX via fastembed, no API key, no PyTorch)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

from aiven_rag_mcp.config import Settings
from aiven_rag_mcp.errors import ConfigurationError

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def count_tokens(self, text: str) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class _CustomModel:
    """Models that fastembed does not ship in its registry, registered on first use."""

    hf_repo: str
    dim: int
    pooling: str
    model_file: str


_CUSTOM_MODELS: dict[str, _CustomModel] = {
    # Full-precision ONNX export of BAAI/bge-small-en-v1.5 (CLS pooling, normalized).
    "Xenova/bge-small-en-v1.5": _CustomModel(
        "Xenova/bge-small-en-v1.5", 384, "CLS", "onnx/model.onnx"
    ),
    "Xenova/bge-base-en-v1.5": _CustomModel(
        "Xenova/bge-base-en-v1.5", 768, "CLS", "onnx/model.onnx"
    ),
}
_registered: set[str] = set()
_register_lock = threading.Lock()


def _ensure_registered(model_name: str) -> None:
    spec = _CUSTOM_MODELS.get(model_name)
    if spec is None:
        return
    with _register_lock:
        if model_name in _registered:
            return
        from fastembed import TextEmbedding
        from fastembed.common.model_description import ModelSource, PoolingType

        known = {m["model"] for m in TextEmbedding.list_supported_models()}
        if model_name not in known:
            TextEmbedding.add_custom_model(
                model=model_name,
                pooling=PoolingType[spec.pooling],
                normalization=True,
                sources=ModelSource(hf=spec.hf_repo),
                dim=spec.dim,
                model_file=spec.model_file,
            )
        _registered.add(model_name)


def known_dimension(model_name: str) -> Optional[int]:
    if model_name in _CUSTOM_MODELS:
        return _CUSTOM_MODELS[model_name].dim
    from fastembed import TextEmbedding

    for description in TextEmbedding.list_supported_models():
        if description["model"] == model_name:
            return int(description["dim"])
    return None


class FastEmbedEmbedder:
    """Thread-safe, lazily loaded fastembed model.

    Loading is deferred to the first call so the MCP handshake is not blocked by
    reading ~130 MB of weights from disk.
    """

    def __init__(
        self,
        model_name: str,
        *,
        model_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        query_prefix: str = "",
        batch_size: int = 32,
        threads: Optional[int] = None,
        max_concurrency: int = 1,
    ) -> None:
        dim = known_dimension(model_name)
        if dim is None:
            raise ConfigurationError(
                f"Unknown embedding model '{model_name}'. Use one supported by fastembed "
                f"or one of: {', '.join(sorted(_CUSTOM_MODELS))}."
            )
        self._model_name = model_name
        self._dim = dim
        self._model_path = model_path
        self._cache_dir = cache_dir
        self._query_prefix = query_prefix
        self._batch_size = batch_size
        self._threads = threads
        self._model: Any = None
        self._tokenizer: Any = None
        self._lock = threading.Lock()
        self._gate = threading.BoundedSemaphore(max_concurrency)

    @classmethod
    def from_settings(cls, settings: Settings) -> FastEmbedEmbedder:
        return cls(
            settings.embedding_model,
            model_path=settings.embedding_model_path,
            cache_dir=settings.model_cache_dir,
            query_prefix=settings.query_prefix,
            batch_size=settings.embed_batch_size,
            threads=settings.embed_threads,
            max_concurrency=settings.inference_concurrency,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dim

    def load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                _ensure_registered(self._model_name)
                kwargs: dict[str, Any] = {"threads": self._threads}
                if self._cache_dir is not None:
                    kwargs["cache_dir"] = str(self._cache_dir)
                if self._model_path is not None:
                    kwargs["specific_model_path"] = str(self._model_path)
                logger.info("Loading embedding model %s", self._model_name)
                try:
                    self._model = TextEmbedding(self._model_name, **kwargs)
                except Exception as exc:
                    raise ConfigurationError(
                        f"Could not load embedding model '{self._model_name}': {exc}"
                    ) from exc
        return self._model

    def _counting_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from tokenizers import Tokenizer

            # The model's tokenizer truncates at 512 tokens, which would hide oversized chunks.
            tokenizer = Tokenizer.from_str(self.load().model.tokenizer.to_str())
            tokenizer.no_truncation()
            tokenizer.no_padding()
            self._tokenizer = tokenizer
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        return len(self._counting_tokenizer().encode(text, add_special_tokens=False).ids)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self.load()
        with self._gate:
            vectors = model.embed(list(texts), batch_size=self._batch_size)
            return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        model = self.load()
        with self._gate:
            vector = next(iter(model.embed([self._query_prefix + text])))
        return list(vector.tolist())


class Reranker(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class FastEmbedReranker:
    """Cross-encoder that scores (query, chunk) pairs jointly; lazily loaded and thread-safe."""

    def __init__(
        self,
        model_name: str,
        *,
        model_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        threads: Optional[int] = None,
        max_concurrency: int = 1,
    ) -> None:
        self._model_name = model_name
        self._model_path = model_path
        self._cache_dir = cache_dir
        self._threads = threads
        self._model: Any = None
        self._lock = threading.Lock()
        self._gate = threading.BoundedSemaphore(max_concurrency)

    @classmethod
    def from_settings(cls, settings: Settings) -> Optional[FastEmbedReranker]:
        if not settings.reranker_model:
            return None
        return cls(
            settings.reranker_model,
            model_path=settings.reranker_model_path,
            cache_dir=settings.model_cache_dir,
            threads=settings.embed_threads,
            max_concurrency=settings.inference_concurrency,
        )

    def load(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    kwargs: dict[str, Any] = {"threads": self._threads}
                    if self._cache_dir is not None:
                        kwargs["cache_dir"] = str(self._cache_dir)
                    if self._model_path is not None:
                        kwargs["specific_model_path"] = str(self._model_path)
                    logger.info("Loading reranker %s", self._model_name)
                    try:
                        self._model = TextCrossEncoder(self._model_name, **kwargs)
                    except Exception as exc:
                        raise ConfigurationError(
                            f"Could not load reranker '{self._model_name}': {exc}"
                        ) from exc
        return self._model

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        model = self.load()
        with self._gate:
            return [float(s) for s in model.rerank(query, list(documents), batch_size=32)]


def download_models(settings: Optional[Settings] = None) -> list[str]:
    """Fetch the embedding and reranker weights into the cache (run at image build time)."""
    settings = settings or Settings()
    embedder = FastEmbedEmbedder.from_settings(settings)
    embedder.embed_query("warm up")
    ready = [f"{embedder.model_name} ({embedder.dimension} dims)"]
    reranker = FastEmbedReranker.from_settings(settings)
    if reranker is not None:
        reranker.score("warm up", ["warm up"])
        ready.append(settings.reranker_model or "")
    return ready


class EmbedderTokenCounter:
    """Adapts an ``Embedder`` to the chunker's ``TokenCounter`` protocol."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def count(self, text: str) -> int:
        return self._embedder.count_tokens(text)
