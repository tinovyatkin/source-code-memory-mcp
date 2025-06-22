"""FastMCP server implementation for Code Memory."""

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastmcp import FastMCP

from .embeddings import CodeEmbedder
from .storage import VectorStorage
from .utils import (
    clean_code_snippet,
    extract_tags_from_description,
    normalize_language,
    validate_code_input,
)

logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    """Application context for managing shared resources."""

    embedder: CodeEmbedder | None = None
    storage: VectorStorage | None = None


# Global application context
app_context = AppContext()


@asynccontextmanager
async def app_lifespan() -> AsyncGenerator[AppContext, None]:
    """Manage application lifecycle and shared resources."""
    logger.info("Starting Code Memory MCP Server")

    try:
        # Initialize embedder
        logger.info("Initializing embedding model...")
        app_context.embedder = CodeEmbedder()
        await app_context.embedder.initialize()

        # Initialize storage
        logger.info("Initializing vector storage...")
        db_path = os.getenv("DATABASE_PATH", ".ai/code_memory.db")
        app_context.storage = VectorStorage(db_path)
        await app_context.storage.initialize(app_context.embedder.get_embedding_dim())

        logger.info("Server initialization complete")
        yield app_context

    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        raise
    finally:
        logger.info("Shutting down Code Memory MCP Server")
        # Cleanup resources
        if app_context.embedder:
            await app_context.embedder.cleanup()
        if app_context.storage:
            await app_context.storage.cleanup()


# Initialize FastMCP server
mcp: FastMCP = FastMCP("Code Memory Server")


# Tool implementations will be added here
@mcp.tool()
async def store_code_snippet(
    code: str, language: str, description: str = "", tags: list[str] | None = None
) -> dict[str, object]:
    """Store a code snippet with semantic embedding.

    Args:
        code: The source code to store
        language: Programming language (e.g., python, javascript, etc.)
        description: Optional description of the code snippet
        tags: Optional list of tags for categorization

    Returns:
        Dictionary with status and snippet ID or error details
    """
    try:
        # Validate input
        errors = validate_code_input(code, language, description)
        if errors:
            return {"status": "error", "errors": errors}

        # Check if components are initialized
        if not app_context.embedder or not app_context.storage:
            return {"status": "error", "message": "Server components not initialized"}

        # Normalize and clean inputs
        normalized_language = normalize_language(language)
        cleaned_code = clean_code_snippet(code)

        # Extract tags from description if not provided
        if tags is None:
            tags = extract_tags_from_description(description)

        # Generate embedding for the code
        embedding = await app_context.embedder.encode_async(cleaned_code)

        # Store in vector database
        snippet_id = await app_context.storage.add_snippet(
            code=cleaned_code,
            language=normalized_language,
            embedding=embedding[0],  # Single embedding from batch
            description=description,
            tags=tags,
        )

        logger.info(
            f"Stored code snippet {snippet_id} for language: {normalized_language}"
        )

        return {
            "status": "success",
            "snippet_id": snippet_id,
            "language": normalized_language,
            "tags": tags,
            "message": f"Code snippet stored successfully with ID {snippet_id}",
        }

    except Exception as e:
        logger.error(f"Error storing code snippet: {e}")
        return {"status": "error", "message": f"Failed to store code snippet: {e!s}"}


@mcp.tool()
async def search_code(
    query: str, language: str | None = None, limit: int = 5, threshold: float = 0.7
) -> dict[str, object]:
    """Search for similar code snippets using semantic search.

    Args:
        query: Natural language query or code snippet to search for
        language: Optional programming language filter (e.g., python, javascript)
        limit: Maximum number of results to return (1-50)
        threshold: Minimum similarity score (0.0-1.0)

    Returns:
        Dictionary with search results and metadata
    """
    try:
        # Validate parameters
        if not query.strip():
            return {"status": "error", "message": "Query cannot be empty"}

        if limit < 1 or limit > 50:
            return {"status": "error", "message": "Limit must be between 1 and 50"}

        if threshold < 0.0 or threshold > 1.0:
            return {
                "status": "error",
                "message": "Threshold must be between 0.0 and 1.0",
            }

        # Check if components are initialized
        if not app_context.embedder or not app_context.storage:
            return {"status": "error", "message": "Server components not initialized"}

        # Normalize language if provided
        normalized_language = normalize_language(language) if language else None

        # Generate embedding for the query
        query_embedding = await app_context.embedder.encode_async(query.strip())

        # Search for similar snippets
        results = await app_context.storage.search_similar(
            query_embedding=query_embedding[0],  # Single embedding from batch
            limit=limit,
            threshold=threshold,
            language=normalized_language,
        )

        logger.info(f"Search for '{query[:50]}...' returned {len(results)} results")

        return {
            "status": "success",
            "query": query,
            "language_filter": normalized_language,
            "results": results,
            "count": len(results),
            "threshold": threshold,
            "message": f"Found {len(results)} matching code snippets",
        }

    except Exception as e:
        logger.error(f"Error searching code snippets: {e}")
        return {"status": "error", "message": f"Failed to search code snippets: {e!s}"}


@mcp.tool()
async def list_languages() -> dict[str, object]:
    """List all programming languages with stored snippets.

    Returns:
        Dictionary with list of languages and their snippet counts
    """
    try:
        # Check if storage is initialized
        if not app_context.storage:
            return {"status": "error", "message": "Storage not initialized"}

        # Get languages from storage
        languages = await app_context.storage.list_languages()

        logger.info(f"Retrieved {len(languages)} languages from storage")

        return {
            "status": "success",
            "languages": languages,
            "total_languages": len(languages),
            "message": f"Found {len(languages)} programming languages with stored snippets",
        }

    except Exception as e:
        logger.error(f"Error listing languages: {e}")
        return {"status": "error", "message": f"Failed to list languages: {e!s}"}


@mcp.tool()
async def get_snippet_by_id(snippet_id: int) -> dict[str, object]:
    """Retrieve a specific code snippet by ID.

    Args:
        snippet_id: The ID of the code snippet to retrieve

    Returns:
        Dictionary with snippet details or error message
    """
    try:
        # Validate snippet ID
        if snippet_id <= 0:
            return {
                "status": "error",
                "message": "Snippet ID must be a positive integer",
            }

        # Check if storage is initialized
        if not app_context.storage:
            return {"status": "error", "message": "Storage not initialized"}

        # Retrieve snippet from storage
        snippet = await app_context.storage.get_snippet(snippet_id)

        if snippet is None:
            return {
                "status": "error",
                "message": f"No code snippet found with ID {snippet_id}",
            }

        logger.info(f"Retrieved snippet {snippet_id}")

        return {
            "status": "success",
            "snippet": snippet,
            "message": f"Successfully retrieved snippet {snippet_id}",
        }

    except Exception as e:
        logger.error(f"Error retrieving snippet {snippet_id}: {e}")
        return {"status": "error", "message": f"Failed to retrieve snippet: {e!s}"}


def create_server() -> FastMCP:
    """Create and return the FastMCP server instance."""
    return mcp


async def main() -> None:
    """Main entry point for running the server."""
    try:
        await mcp.run()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
