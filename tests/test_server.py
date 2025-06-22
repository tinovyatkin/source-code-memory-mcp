"""Tests for the FastMCP server implementation."""

from unittest.mock import MagicMock

import pytest

from code_memory.server import AppContext, app_lifespan, create_server


class TestAppContext:
    """Test AppContext dataclass."""

    def test_app_context_initialization(self) -> None:
        """Test AppContext can be initialized with default values."""
        context = AppContext()
        assert context.embedder is None
        assert context.storage is None

    def test_app_context_with_values(self) -> None:
        """Test AppContext can be initialized with custom values."""
        mock_embedder = MagicMock()
        mock_storage = MagicMock()

        context = AppContext(embedder=mock_embedder, storage=mock_storage)
        assert context.embedder is mock_embedder
        assert context.storage is mock_storage


class TestServerCreation:
    """Test server creation and configuration."""

    def test_create_server(self) -> None:
        """Test server creation returns FastMCP instance."""
        server = create_server()
        assert server is not None
        assert hasattr(server, "run")


@pytest.mark.asyncio
class TestAppLifespan:
    """Test application lifecycle management."""

    async def test_app_lifespan_context_manager(self) -> None:
        """Test app_lifespan works as async context manager."""
        async with app_lifespan() as context:
            assert isinstance(context, AppContext)
            # TODO: Add assertions for initialized components once implemented


# TODO: Add tests for tool implementations once they are complete
class TestToolImplementations:
    """Test MCP tool implementations."""

    def test_store_code_snippet_placeholder(self) -> None:
        """Test store_code_snippet returns not implemented status."""
        # TODO: Implement actual test once tool is implemented

    def test_search_code_placeholder(self) -> None:
        """Test search_code returns empty list."""
        # TODO: Implement actual test once tool is implemented

    def test_list_languages_placeholder(self) -> None:
        """Test list_languages returns empty list."""
        # TODO: Implement actual test once tool is implemented

    def test_get_snippet_by_id_placeholder(self) -> None:
        """Test get_snippet_by_id returns None."""
        # TODO: Implement actual test once tool is implemented
