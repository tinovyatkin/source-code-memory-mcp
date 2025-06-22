"""Tests for the vector storage functionality."""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from code_memory.storage import VectorStorage


class TestVectorStorage:
    """Test VectorStorage class."""

    def test_init_default_path(self) -> None:
        """Test VectorStorage initialization with default path."""
        storage = VectorStorage()
        assert storage.db_path == Path(".ai/code_memory.db")
        assert storage.connection is None
        assert storage.embedding_dim is None

    def test_init_custom_path(self) -> None:
        """Test VectorStorage initialization with custom path."""
        custom_path = "/tmp/test.db"
        storage = VectorStorage(custom_path)
        assert storage.db_path == Path(custom_path)

    def test_init_path_object(self) -> None:
        """Test VectorStorage initialization with Path object."""
        path_obj = Path("/tmp/test.db")
        storage = VectorStorage(path_obj)
        assert storage.db_path == path_obj


@pytest.mark.asyncio
class TestVectorStorageAsync:
    """Test async methods of VectorStorage."""

    async def test_initialize_creates_directory(self) -> None:
        """Test that initialize creates parent directory for any path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test nested directory creation
            db_path = Path(temp_dir) / "deep" / "nested" / "path" / "test.db"
            storage = VectorStorage(db_path)

            # Mock sqlite connection to avoid actual DB operations
            with patch("code_memory.storage.sqlite3.connect") as mock_connect:
                mock_conn = MagicMock()
                mock_connect.return_value = mock_conn
                mock_conn.execute.return_value = None

                await storage.initialize(768)

                assert db_path.parent.exists()
                assert storage.embedding_dim == 768

                # Verify the full path exists
                assert (Path(temp_dir) / "deep" / "nested" / "path").exists()

    async def test_initialize_creates_ai_directory_with_readme(self) -> None:
        """Test that initialize creates .ai directory with README."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / ".ai" / "code_memory.db"
            storage = VectorStorage(db_path)

            # Mock sqlite connection to avoid actual DB operations
            with patch("code_memory.storage.sqlite3.connect") as mock_connect:
                mock_conn = MagicMock()
                mock_connect.return_value = mock_conn
                mock_conn.execute.return_value = None

                await storage.initialize(768)

                assert db_path.parent.exists()
                assert (db_path.parent / "README.md").exists()

                # Check README content
                readme_content = (db_path.parent / "README.md").read_text()
                assert "AI Data Directory" in readme_content
                assert "code_memory.db" in readme_content

    @patch("code_memory.storage.sqlite3.connect")
    async def test_initialize_database_setup(self, mock_connect: MagicMock) -> None:
        """Test database schema creation during initialization."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        storage = VectorStorage()
        await storage.initialize(768)

        # Verify database connection was established
        mock_connect.assert_called_once()

        # Verify schema creation calls
        mock_conn.execute.assert_called()
        mock_conn.commit.assert_called()

    @patch("code_memory.storage.sqlite3.connect")
    async def test_initialize_sqlite_vec_extension_error(
        self, mock_connect: MagicMock
    ) -> None:
        """Test handling of sqlite-vec extension loading failure."""
        mock_conn = MagicMock()
        mock_conn.load_extension.side_effect = sqlite3.OperationalError(
            "Extension not found"
        )
        mock_connect.return_value = mock_conn

        storage = VectorStorage()

        with pytest.raises(RuntimeError, match="sqlite-vec extension not available"):
            await storage.initialize(768)

    async def test_add_snippet_not_initialized(self) -> None:
        """Test add_snippet fails when storage not initialized."""
        storage = VectorStorage()
        embedding = np.random.random((768,)).astype(np.float32)

        with pytest.raises(RuntimeError, match="Storage not initialized"):
            await storage.add_snippet("code", "python", embedding)

    async def test_search_similar_not_initialized(self) -> None:
        """Test search_similar fails when storage not initialized."""
        storage = VectorStorage()
        query_embedding = np.random.random((768,)).astype(np.float32)

        with pytest.raises(RuntimeError, match="Storage not initialized"):
            await storage.search_similar(query_embedding)

    async def test_get_snippet_not_initialized(self) -> None:
        """Test get_snippet fails when storage not initialized."""
        storage = VectorStorage()

        with pytest.raises(RuntimeError, match="Storage not initialized"):
            await storage.get_snippet(1)

    async def test_list_languages_not_initialized(self) -> None:
        """Test list_languages fails when storage not initialized."""
        storage = VectorStorage()

        with pytest.raises(RuntimeError, match="Storage not initialized"):
            await storage.list_languages()


class TestVectorStorageSync:
    """Test synchronous helper methods of VectorStorage."""

    def test_insert_snippet(self) -> None:
        """Test _insert_snippet method."""
        # Create in-memory database for testing
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row

        # Create tables manually since we can't load sqlite-vec in tests
        conn.execute("""
            CREATE TABLE code_snippets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                language TEXT NOT NULL,
                description TEXT DEFAULT '',
                tags TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create a mock embedding table since we can't use vec0
        conn.execute("""
            CREATE TABLE code_embeddings (
                snippet_id INTEGER PRIMARY KEY,
                embedding BLOB
            )
        """)

        storage = VectorStorage()
        storage.connection = conn

        embedding = np.random.random((768,)).astype(np.float32)
        snippet_id = storage._insert_snippet(
            "print('hello')", "python", "Test code", "test,example", embedding
        )

        assert isinstance(snippet_id, int)
        assert snippet_id > 0

        # Verify snippet was inserted
        cursor = conn.execute("SELECT * FROM code_snippets WHERE id = ?", (snippet_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row["code"] == "print('hello')"
        assert row["language"] == "python"
        assert row["description"] == "Test code"
        assert row["tags"] == "test,example"

    def test_get_snippet_sync_not_found(self) -> None:
        """Test _get_snippet_sync when snippet doesn't exist."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE code_snippets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                language TEXT NOT NULL,
                description TEXT DEFAULT '',
                tags TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        storage = VectorStorage()
        storage.connection = conn

        result = storage._get_snippet_sync(999)
        assert result is None

    def test_list_languages_sync_empty(self) -> None:
        """Test _list_languages_sync with empty database."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE code_snippets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                language TEXT NOT NULL,
                description TEXT DEFAULT '',
                tags TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        storage = VectorStorage()
        storage.connection = conn

        result = storage._list_languages_sync()
        assert result == []


@pytest.mark.asyncio
class TestVectorStorageCleanup:
    """Test cleanup functionality."""

    async def test_cleanup_with_connection(self) -> None:
        """Test cleanup with active connection."""
        storage = VectorStorage()
        mock_conn = MagicMock()
        storage.connection = mock_conn

        await storage.cleanup()

        mock_conn.close.assert_called_once()
        assert storage.connection is None

    async def test_cleanup_without_connection(self) -> None:
        """Test cleanup without active connection."""
        storage = VectorStorage()

        # Should not raise exception
        await storage.cleanup()
        assert storage.connection is None
