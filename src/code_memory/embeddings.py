"""Embedding generation for code snippets using sentence-transformers."""

import asyncio
import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingStats:
    """Track embedding generation performance statistics."""

    def __init__(self) -> None:
        self.total_embeddings: int = 0
        self.total_time: float = 0.0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.memory_cleanups: int = 0

    def record_embedding(self, count: int, time_taken: float) -> None:
        """Record embedding generation statistics."""
        self.total_embeddings += count
        self.total_time += time_taken

    def record_cache_hit(self) -> None:
        """Record cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record cache miss."""
        self.cache_misses += 1

    def record_memory_cleanup(self) -> None:
        """Record memory cleanup event."""
        self.memory_cleanups += 1

    def get_average_time(self) -> float:
        """Get average time per embedding."""
        if self.total_embeddings == 0:
            return 0.0
        return self.total_time / self.total_embeddings

    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return self.cache_hits / total_requests


class CodeEmbedder:
    """Generate embeddings for code snippets using Jina embeddings model."""

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v2-base-code",
        device: str | None = None,
        max_workers: int = 4,
        cache_size: int = 512,
        memory_cleanup_threshold: int = 1000,
    ) -> None:
        """Initialize the code embedder.

        Args:
            model_name: HuggingFace model name for embeddings
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
            max_workers: Number of threads for async operations
            cache_size: Size of LRU cache for embeddings
            memory_cleanup_threshold: Number of embeddings after which to cleanup
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: SentenceTransformer | None = None
        self.embedding_dim: int = 768  # Default for Jina base model
        self.max_seq_length: int = 8192
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cache_size = cache_size
        self.memory_cleanup_threshold = memory_cleanup_threshold
        self.stats = EmbeddingStats()
        self._embeddings_since_cleanup = 0

        # Create LRU cache for embeddings
        self._cached_encode = lru_cache(maxsize=cache_size)(self._encode_with_cache_key)

        logger.info(
            f"CodeEmbedder initialized with model: {model_name}, device: {self.device}, "
            f"cache_size: {cache_size}"
        )

    async def initialize(self) -> None:
        """Initialize the embedding model asynchronously."""
        if self.model is not None:
            logger.warning("Model already initialized")
            return

        logger.info(f"Loading embedding model: {self.model_name}")

        # Load model in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        self.model = await loop.run_in_executor(self.executor, self._load_model)
        if self.model is None:
            raise RuntimeError("Failed to load model")

        embedding_dim = self.model.get_sentence_embedding_dimension()
        if embedding_dim is None:
            raise RuntimeError("Failed to get embedding dimension")
        self.embedding_dim = embedding_dim
        logger.info(
            f"Model loaded successfully. Embedding dimension: {self.embedding_dim}"
        )

    def _load_model(self) -> SentenceTransformer:
        """Load the sentence transformer model (runs in thread pool)."""
        model = SentenceTransformer(
            self.model_name, device=self.device, trust_remote_code=True
        )

        # Configure model settings
        if hasattr(model, "max_seq_length"):
            model.max_seq_length = self.max_seq_length

        # Enable FP16 for GPU inference
        if self.device == "cuda" and torch.cuda.is_available():
            model.half()

        return model

    def _encode_with_cache_key(
        self, text: str, normalize: bool = True
    ) -> NDArray[np.float32]:
        """Encode single text with caching (cache key function)."""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        self.stats.record_cache_miss()
        start_time = time.time()

        embedding = self.model.encode(
            [text],
            batch_size=1,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        time_taken = time.time() - start_time
        self.stats.record_embedding(1, time_taken)

        result: NDArray[np.float32] = embedding[0].astype(np.float32)
        return result

    def _chunk_texts(self, texts: list[str], chunk_size: int) -> list[list[str]]:
        """Split texts into chunks for memory-efficient processing."""
        chunks = []
        for i in range(0, len(texts), chunk_size):
            chunks.append(texts[i : i + chunk_size])
        return chunks

    def _periodic_memory_cleanup(self) -> None:
        """Perform periodic memory cleanup."""
        if self._embeddings_since_cleanup >= self.memory_cleanup_threshold:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            self.stats.record_memory_cleanup()
            self._embeddings_since_cleanup = 0
            logger.debug("Performed memory cleanup")

    async def encode_async(
        self,
        texts: str | list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        use_cache: bool = True,
        max_chunk_size: int = 1000,
    ) -> NDArray[np.float32]:
        """Generate embeddings for text(s) asynchronously with optimizations.

        Args:
            texts: Single text or list of texts to embed
            batch_size: Batch size for processing
            normalize_embeddings: Whether to normalize embeddings
            use_cache: Whether to use LRU cache for single texts
            max_chunk_size: Maximum chunk size for large batches

        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        if isinstance(texts, str):
            texts = [texts]

        logger.debug(f"Generating embeddings for {len(texts)} texts")

        # For single text with caching enabled
        if len(texts) == 1 and use_cache:
            try:
                # Check cache first
                cached_result = self._cached_encode(texts[0], normalize_embeddings)
                self.stats.record_cache_hit()
                return np.array([cached_result])
            except Exception:
                # Fall through to regular processing
                pass

        # For large batches, use chunking
        if len(texts) > max_chunk_size:
            return await self._encode_chunked(
                texts, batch_size, normalize_embeddings, max_chunk_size
            )

        # Regular batch processing
        start_time = time.time()
        loop = asyncio.get_event_loop()
        if self.model is None:
            raise RuntimeError("Model not initialized")

        def _encode_batch() -> Any:
            if self.model is None:
                raise RuntimeError("Model not initialized")
            return self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        embeddings = await loop.run_in_executor(self.executor, _encode_batch)

        time_taken = time.time() - start_time
        self.stats.record_embedding(len(texts), time_taken)
        self._embeddings_since_cleanup += len(texts)

        # Periodic memory cleanup
        self._periodic_memory_cleanup()

        result: NDArray[np.float32] = embeddings.astype(np.float32)
        return result

    async def _encode_chunked(
        self,
        texts: list[str],
        batch_size: int,
        normalize_embeddings: bool,
        chunk_size: int,
    ) -> NDArray[np.float32]:
        """Encode large text lists in chunks to manage memory."""
        chunks = self._chunk_texts(texts, chunk_size)
        all_embeddings = []

        logger.info(f"Processing {len(texts)} texts in {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i + 1}/{len(chunks)} ({len(chunk)} texts)")

            start_time = time.time()
            loop = asyncio.get_event_loop()
            if self.model is None:
                raise RuntimeError("Model not initialized")

            def _encode_chunk(current_chunk: list[str] = chunk) -> Any:
                if self.model is None:
                    raise RuntimeError("Model not initialized")
                return self.model.encode(
                    current_chunk,
                    batch_size=batch_size,
                    normalize_embeddings=normalize_embeddings,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )

            chunk_embeddings = await loop.run_in_executor(self.executor, _encode_chunk)

            all_embeddings.append(chunk_embeddings)

            time_taken = time.time() - start_time
            self.stats.record_embedding(len(chunk), time_taken)
            self._embeddings_since_cleanup += len(chunk)

            # Memory cleanup after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Combine all embeddings
        combined_embeddings = np.vstack(all_embeddings)

        # Final memory cleanup
        self._periodic_memory_cleanup()

        result: NDArray[np.float32] = combined_embeddings.astype(np.float32)
        return result

    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings."""
        return self.embedding_dim

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for embeddings."""
        return {
            "total_embeddings": self.stats.total_embeddings,
            "total_time": self.stats.total_time,
            "average_time_per_embedding": self.stats.get_average_time(),
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "cache_hit_rate": self.stats.get_cache_hit_rate(),
            "memory_cleanups": self.stats.memory_cleanups,
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cached_encode.cache_clear()
        logger.info(f"Cleared embedding cache (size: {self.cache_size})")

    def get_cache_info(self) -> dict[str, int]:
        """Get cache information."""
        cache_info = self._cached_encode.cache_info()
        return {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "maxsize": cache_info.maxsize or 0,
            "currsize": cache_info.currsize,
        }

    async def cleanup(self) -> None:
        """Clean up resources."""
        # Log final performance statistics
        stats = self.get_performance_stats()
        logger.info(f"CodeEmbedder performance stats: {stats}")

        # Clear cache
        self.clear_cache()

        if self.model is not None:
            # Move model to CPU and clear cache
            if hasattr(self.model, "to"):
                self.model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.executor.shutdown(wait=True)
        logger.info("CodeEmbedder resources cleaned up")
