"""Tests for the embedding generation functionality."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from code_memory.embeddings import CodeEmbedder


class TestCodeEmbedder:
    """Test CodeEmbedder class."""

    def test_init_default_parameters(self) -> None:
        """Test CodeEmbedder initialization with default parameters."""
        embedder = CodeEmbedder()

        assert embedder.model_name == "jinaai/jina-embeddings-v2-base-code"
        assert embedder.device in ["cpu", "cuda"]
        assert embedder.model is None
        assert embedder.embedding_dim == 768
        assert embedder.max_seq_length == 8192
        assert embedder.executor is not None

    def test_init_custom_parameters(self) -> None:
        """Test CodeEmbedder initialization with custom parameters."""
        embedder = CodeEmbedder(model_name="custom-model", device="cpu", max_workers=2)

        assert embedder.model_name == "custom-model"
        assert embedder.device == "cpu"
        assert embedder.executor._max_workers == 2

    @patch("code_memory.embeddings.torch.cuda.is_available")
    def test_device_auto_detection_cuda(self, mock_cuda_available: MagicMock) -> None:
        """Test automatic CUDA device detection."""
        mock_cuda_available.return_value = True
        embedder = CodeEmbedder()
        assert embedder.device == "cuda"

    @patch("code_memory.embeddings.torch.cuda.is_available")
    def test_device_auto_detection_cpu(self, mock_cuda_available: MagicMock) -> None:
        """Test automatic CPU device detection."""
        mock_cuda_available.return_value = False
        embedder = CodeEmbedder()
        assert embedder.device == "cpu"


@pytest.mark.asyncio
class TestCodeEmbedderAsync:
    """Test async methods of CodeEmbedder."""

    @patch("code_memory.embeddings.SentenceTransformer")
    async def test_initialize_success(
        self, mock_sentence_transformer: MagicMock
    ) -> None:
        """Test successful model initialization."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 512
        mock_sentence_transformer.return_value = mock_model

        embedder = CodeEmbedder()
        await embedder.initialize()

        assert embedder.model is mock_model
        assert embedder.embedding_dim == 512
        mock_sentence_transformer.assert_called_once()

    async def test_initialize_already_initialized(self) -> None:
        """Test initialization when model is already loaded."""
        embedder = CodeEmbedder()
        embedder.model = MagicMock()

        # Should not raise exception and should warn
        await embedder.initialize()

    @patch("code_memory.embeddings.SentenceTransformer")
    async def test_encode_async_single_text(
        self, mock_sentence_transformer: MagicMock
    ) -> None:
        """Test encoding a single text string."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_embeddings = np.random.random((1, 768)).astype(np.float32)
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model

        embedder = CodeEmbedder()
        await embedder.initialize()

        result = await embedder.encode_async("test code")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (1, 768)
        mock_model.encode.assert_called_once()

    @patch("code_memory.embeddings.SentenceTransformer")
    async def test_encode_async_multiple_texts(
        self, mock_sentence_transformer: MagicMock
    ) -> None:
        """Test encoding multiple texts."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_embeddings = np.random.random((3, 768)).astype(np.float32)
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model

        embedder = CodeEmbedder()
        await embedder.initialize()

        texts = ["code1", "code2", "code3"]
        result = await embedder.encode_async(texts)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (3, 768)

    async def test_encode_async_not_initialized(self) -> None:
        """Test encoding fails when model not initialized."""
        embedder = CodeEmbedder()

        with pytest.raises(RuntimeError, match="Model not initialized"):
            await embedder.encode_async("test code")

    def test_get_embedding_dim(self) -> None:
        """Test getting embedding dimension."""
        embedder = CodeEmbedder()
        assert embedder.get_embedding_dim() == 768

        embedder.embedding_dim = 512
        assert embedder.get_embedding_dim() == 512

    @patch("code_memory.embeddings.torch.cuda.empty_cache")
    async def test_cleanup(self, mock_empty_cache: MagicMock) -> None:
        """Test resource cleanup."""
        embedder = CodeEmbedder()
        embedder.model = MagicMock()

        await embedder.cleanup()

        # Should shutdown executor
        assert embedder.executor._shutdown
