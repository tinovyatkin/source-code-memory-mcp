"""Vector storage implementation using SQLite with sqlite-vec extension."""

import asyncio
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class VectorStorage:
    """SQLite-based vector storage with sqlite-vec extension."""

    def __init__(self, db_path: str | Path = ".ai/code_memory.db") -> None:
        """Initialize vector storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.connection: sqlite3.Connection | None = None
        self.embedding_dim: int | None = None

        logger.info(f"VectorStorage initialized with database: {self.db_path}")

    async def initialize(self, embedding_dim: int) -> None:
        """Initialize database and create tables.

        Args:
            embedding_dim: Dimension of embedding vectors
        """
        self.embedding_dim = embedding_dim

        # Create database directory tree if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured database directory exists: {self.db_path.parent}")

        # Create .ai directory structure documentation if it's the default .ai location
        if (
            self.db_path.parent.name == ".ai"
            and not (self.db_path.parent / "README.md").exists()
        ):
            readme_content = """# AI Data Directory

This directory contains application-specific data for the Code Memory MCP Server.

## Contents

- `code_memory.db` - SQLite database with sqlite-vec extension containing:
  - Code snippets and metadata
  - Vector embeddings for semantic search
  - Access statistics and timestamps

- `cache/` - Application-specific cache files (if needed)

## Notes

- **HuggingFace Models**: The embedding models are automatically cached by HuggingFace Hub in the system cache directory (usually `~/.cache/huggingface/hub/`). We don't duplicate this caching.

- **Database Backup**: The `code_memory.db` file contains your entire code snippet collection. Consider backing it up regularly.

- **Portability**: This directory can be copied between machines to transfer your code snippet database.

## Environment Configuration

You can override the default database location by setting the `DATABASE_PATH` environment variable:

```bash
export DATABASE_PATH="/path/to/your/custom/location/code_memory.db"
```"""
            try:
                with open(
                    self.db_path.parent / "README.md", "w", encoding="utf-8"
                ) as f:
                    f.write(readme_content)
                logger.info(
                    f"Created .ai directory documentation at {self.db_path.parent / 'README.md'}"
                )
            except Exception as e:
                logger.warning(f"Could not create .ai/README.md: {e}")

        # Initialize in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._initialize_db)

        logger.info(f"Database initialized with embedding dimension: {embedding_dim}")

    def _initialize_db(self) -> None:
        """Initialize database schema (runs in thread pool)."""
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row

        # Load sqlite-vec extension
        try:
            self.connection.load_extension("sqlite-vec")
            logger.info("sqlite-vec extension loaded successfully")
        except sqlite3.OperationalError as e:
            logger.error(f"Failed to load sqlite-vec extension: {e}")
            raise RuntimeError("sqlite-vec extension not available") from e

        # Configure SQLite optimizations
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA cache_size=10000")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=MEMORY")

        # Create main snippets table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS code_snippets (
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

        # Create virtual table for embeddings
        self.connection.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS code_embeddings
            USING vec0(
                snippet_id INTEGER PRIMARY KEY,
                embedding FLOAT[{self.embedding_dim}]
            )
        """)

        # Create indices for performance
        self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_language ON code_snippets(language)
        """)
        self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON code_snippets(created_at)
        """)
        self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags ON code_snippets(tags)
        """)

        self.connection.commit()

    async def add_snippet(
        self,
        code: str,
        language: str,
        embedding: NDArray[np.float32],
        description: str = "",
        tags: list[str] | None = None,
    ) -> int:
        """Add a code snippet with its embedding.

        Args:
            code: Source code content
            language: Programming language
            embedding: Code embedding vector
            description: Optional description
            tags: List of tags

        Returns:
            ID of the inserted snippet
        """
        if self.connection is None:
            raise RuntimeError("Storage not initialized")

        tags_str = ",".join(tags or [])

        # Insert in thread pool
        loop = asyncio.get_event_loop()
        snippet_id = await loop.run_in_executor(
            None,
            lambda: self._insert_snippet(
                code, language, description, tags_str, embedding
            ),
        )

        logger.debug(f"Added snippet {snippet_id} for language: {language}")
        return snippet_id

    def _insert_snippet(
        self,
        code: str,
        language: str,
        description: str,
        tags_str: str,
        embedding: NDArray[np.float32],
    ) -> int:
        """Insert snippet and embedding (runs in thread pool)."""
        if self.connection is None:
            raise RuntimeError("Connection not initialized")
        cursor = self.connection.cursor()

        # Insert snippet metadata
        cursor.execute(
            """
            INSERT INTO code_snippets (code, language, description, tags)
            VALUES (?, ?, ?, ?)
        """,
            (code, language, description, tags_str),
        )

        snippet_id = cursor.lastrowid
        if snippet_id is None:
            raise RuntimeError("Failed to get snippet ID after insert")

        # Insert embedding
        cursor.execute(
            """
            INSERT INTO code_embeddings (snippet_id, embedding)
            VALUES (?, ?)
        """,
            (snippet_id, embedding.tobytes()),
        )

        self.connection.commit()
        return snippet_id

    async def search_similar(
        self,
        query_embedding: NDArray[np.float32],
        limit: int = 5,
        threshold: float = 0.7,
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar code snippets.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            language: Optional language filter

        Returns:
            List of similar snippets with metadata
        """
        if self.connection is None:
            raise RuntimeError("Storage not initialized")

        # Search in thread pool
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._search_similar_sync(
                query_embedding, limit, threshold, language
            ),
        )

        logger.debug(f"Found {len(results)} similar snippets")
        return results

    def _search_similar_sync(
        self,
        query_embedding: NDArray[np.float32],
        limit: int,
        threshold: float,
        language: str | None,
    ) -> list[dict[str, Any]]:
        """Synchronous similarity search (runs in thread pool)."""
        # Build query with optional language filter
        base_query = """
            SELECT
                s.id, s.code, s.language, s.description, s.tags,
                s.created_at, s.access_count,
                vec_distance_cosine(e.embedding, ?) as distance
            FROM code_snippets s
            JOIN code_embeddings e ON s.id = e.snippet_id
        """

        params: list[object] = [query_embedding.tobytes()]

        if language:
            base_query += " WHERE s.language = ?"
            params.append(language)

        base_query += """
            ORDER BY distance ASC
            LIMIT ?
        """
        params.append(limit)

        if self.connection is None:
            raise RuntimeError("Connection not initialized")
        cursor = self.connection.cursor()
        cursor.execute(base_query, params)

        results = []
        for row in cursor.fetchall():
            # Convert cosine distance to similarity score
            similarity = 1.0 - row["distance"]

            if similarity >= threshold:
                results.append(
                    {
                        "id": row["id"],
                        "code": row["code"],
                        "language": row["language"],
                        "description": row["description"],
                        "tags": row["tags"].split(",") if row["tags"] else [],
                        "similarity": float(similarity),
                        "created_at": row["created_at"],
                        "access_count": row["access_count"],
                    }
                )

        return results

    async def get_snippet(self, snippet_id: int) -> dict[str, Any] | None:
        """Retrieve a snippet by ID and update access statistics."""
        if self.connection is None:
            raise RuntimeError("Storage not initialized")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._get_snippet_sync(snippet_id)
        )

    def _get_snippet_sync(self, snippet_id: int) -> dict[str, Any] | None:
        """Synchronous snippet retrieval (runs in thread pool)."""
        if self.connection is None:
            raise RuntimeError("Connection not initialized")
        cursor = self.connection.cursor()

        # Get snippet
        cursor.execute(
            """
            SELECT id, code, language, description, tags, created_at, updated_at,
                   access_count, last_accessed
            FROM code_snippets
            WHERE id = ?
        """,
            (snippet_id,),
        )

        row = cursor.fetchone()
        if not row:
            return None

        # Update access statistics
        cursor.execute(
            """
            UPDATE code_snippets
            SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
            WHERE id = ?
        """,
            (snippet_id,),
        )

        self.connection.commit()

        return {
            "id": row["id"],
            "code": row["code"],
            "language": row["language"],
            "description": row["description"],
            "tags": row["tags"].split(",") if row["tags"] else [],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "access_count": row["access_count"] + 1,
            "last_accessed": datetime.now().astimezone().isoformat(),
        }

    async def update_snippet(
        self,
        snippet_id: int,
        code: str | None = None,
        language: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        embedding: NDArray[np.float32] | None = None,
    ) -> bool:
        """Update an existing code snippet.

        Args:
            snippet_id: ID of snippet to update
            code: New code content (optional)
            language: New language (optional)
            description: New description (optional)
            tags: New tags list (optional)
            embedding: New embedding if code changed (optional)

        Returns:
            True if snippet was updated, False if not found
        """
        if self.connection is None:
            raise RuntimeError("Storage not initialized")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._update_snippet_sync(
                snippet_id, code, language, description, tags, embedding
            ),
        )

    def _update_snippet_sync(
        self,
        snippet_id: int,
        code: str | None,
        language: str | None,
        description: str | None,
        tags: list[str] | None,
        embedding: NDArray[np.float32] | None,
    ) -> bool:
        """Synchronous snippet update (runs in thread pool)."""
        if self.connection is None:
            raise RuntimeError("Connection not initialized")

        cursor = self.connection.cursor()

        # Check if snippet exists
        cursor.execute("SELECT id FROM code_snippets WHERE id = ?", (snippet_id,))
        if not cursor.fetchone():
            return False

        # Build update query dynamically
        update_fields = []
        params = []

        if code is not None:
            update_fields.append("code = ?")
            params.append(code)

        if language is not None:
            update_fields.append("language = ?")
            params.append(language)

        if description is not None:
            update_fields.append("description = ?")
            params.append(description)

        if tags is not None:
            update_fields.append("tags = ?")
            params.append(",".join(tags))

        if update_fields:
            update_fields.append("updated_at = CURRENT_TIMESTAMP")
            params.append(str(snippet_id))

            update_query = f"""
                UPDATE code_snippets
                SET {", ".join(update_fields)}
                WHERE id = ?
            """
            cursor.execute(update_query, params)

        # Update embedding if provided
        if embedding is not None:
            cursor.execute(
                """
                UPDATE code_embeddings
                SET embedding = ?
                WHERE snippet_id = ?
            """,
                (embedding.tobytes(), snippet_id),
            )

        self.connection.commit()
        return True

    async def delete_snippet(self, snippet_id: int) -> bool:
        """Delete a code snippet and its embedding.

        Args:
            snippet_id: ID of snippet to delete

        Returns:
            True if snippet was deleted, False if not found
        """
        if self.connection is None:
            raise RuntimeError("Storage not initialized")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._delete_snippet_sync(snippet_id)
        )

    def _delete_snippet_sync(self, snippet_id: int) -> bool:
        """Synchronous snippet deletion (runs in thread pool)."""
        if self.connection is None:
            raise RuntimeError("Connection not initialized")

        cursor = self.connection.cursor()

        # Check if snippet exists
        cursor.execute("SELECT id FROM code_snippets WHERE id = ?", (snippet_id,))
        if not cursor.fetchone():
            return False

        # Delete embedding first (foreign key relationship)
        cursor.execute(
            "DELETE FROM code_embeddings WHERE snippet_id = ?", (snippet_id,)
        )

        # Delete snippet
        cursor.execute("DELETE FROM code_snippets WHERE id = ?", (snippet_id,))

        self.connection.commit()
        return True

    async def search_similar_with_filters(
        self,
        query_embedding: NDArray[np.float32],
        limit: int = 5,
        threshold: float = 0.7,
        language: str | None = None,
        tags: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar code snippets with advanced filters.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            language: Optional language filter
            tags: Optional tags filter (any of these tags)
            date_from: Optional start date filter (ISO format)
            date_to: Optional end date filter (ISO format)

        Returns:
            List of similar snippets with metadata
        """
        if self.connection is None:
            raise RuntimeError("Storage not initialized")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._search_similar_with_filters_sync(
                query_embedding, limit, threshold, language, tags, date_from, date_to
            ),
        )

    def _search_similar_with_filters_sync(
        self,
        query_embedding: NDArray[np.float32],
        limit: int,
        threshold: float,
        language: str | None,
        tags: list[str] | None,
        date_from: str | None,
        date_to: str | None,
    ) -> list[dict[str, Any]]:
        """Synchronous similarity search with filters (runs in thread pool)."""
        base_query = """
            SELECT
                s.id, s.code, s.language, s.description, s.tags,
                s.created_at, s.access_count,
                vec_distance_cosine(e.embedding, ?) as distance
            FROM code_snippets s
            JOIN code_embeddings e ON s.id = e.snippet_id
        """

        params: list[object] = [query_embedding.tobytes()]
        where_conditions = []

        if language:
            where_conditions.append("s.language = ?")
            params.append(language)

        if tags:
            # Match any of the provided tags
            tag_conditions = []
            for tag in tags:
                tag_conditions.append("s.tags LIKE ?")
                params.append(f"%{tag}%")
            if tag_conditions:
                where_conditions.append(f"({' OR '.join(tag_conditions)})")

        if date_from:
            where_conditions.append("s.created_at >= ?")
            params.append(date_from)

        if date_to:
            where_conditions.append("s.created_at <= ?")
            params.append(date_to)

        if where_conditions:
            base_query += " WHERE " + " AND ".join(where_conditions)

        base_query += """
            ORDER BY distance ASC
            LIMIT ?
        """
        params.append(limit)

        if self.connection is None:
            raise RuntimeError("Connection not initialized")
        cursor = self.connection.cursor()
        cursor.execute(base_query, params)

        results = []
        for row in cursor.fetchall():
            similarity = 1.0 - row["distance"]

            if similarity >= threshold:
                results.append(
                    {
                        "id": row["id"],
                        "code": row["code"],
                        "language": row["language"],
                        "description": row["description"],
                        "tags": row["tags"].split(",") if row["tags"] else [],
                        "similarity": float(similarity),
                        "created_at": row["created_at"],
                        "access_count": row["access_count"],
                    }
                )

        return results

    async def list_languages(self) -> list[dict[str, str | int]]:
        """List all languages with snippet counts."""
        if self.connection is None:
            raise RuntimeError("Storage not initialized")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._list_languages_sync)

    def _list_languages_sync(self) -> list[dict[str, str | int]]:
        """Synchronous language listing (runs in thread pool)."""
        if self.connection is None:
            raise RuntimeError("Connection not initialized")
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT language, COUNT(*) as count
            FROM code_snippets
            GROUP BY language
            ORDER BY count DESC, language ASC
        """)

        return [
            {"language": row["language"], "count": row["count"]}
            for row in cursor.fetchall()
        ]

    async def cleanup(self) -> None:
        """Clean up database connection."""
        if self.connection:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.connection.close)
            self.connection = None

        logger.info("VectorStorage cleaned up")
