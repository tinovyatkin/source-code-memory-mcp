"""MCP server implementation for Code Memory using FastMCP framework."""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import List, Dict

from mcp.server.fastmcp import FastMCP, Context

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

    embedder: CodeEmbedder
    storage: VectorStorage


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with proper resource cleanup"""
    logger.info("Initializing code memory server...")
    
    # Initialize embedding model
    embedder = CodeEmbedder(model_name="jinaai/jina-embeddings-v2-base-code")
    await embedder.initialize()
    
    # Initialize vector storage
    db_path = os.getenv("DATABASE_PATH", ".ai/code_memory.db")
    storage = VectorStorage(db_path)
    await storage.initialize(embedder.get_embedding_dim())
    
    try:
        yield AppContext(embedder=embedder, storage=storage)
    finally:
        logger.info("Shutting down code memory server...")
        await storage.cleanup()
        await embedder.cleanup()


# Create server with lifecycle management
mcp = FastMCP("Code Memory Server", lifespan=app_lifespan)


@mcp.tool()
async def store_code_snippet(
    code: str, 
    language: str, 
    description: str,
    ctx: Context,
    tags: List[str] = None
) -> str:
    """Store a code snippet with semantic embedding"""
    try:
        # Validate input
        if not code.strip():
            raise ValueError("Code cannot be empty")
        
        ctx.info(f"Storing {language} code snippet")
        
        # Normalize and clean inputs
        normalized_language = normalize_language(language)
        cleaned_code = clean_code_snippet(code)
        
        # Extract tags from description if not provided
        if tags is None:
            tags = extract_tags_from_description(description)
        
        # Generate embedding
        embedding = await ctx.app.embedder.encode_async([cleaned_code])
        
        # Store in database
        snippet_id = await ctx.app.storage.add_snippet(
            code=cleaned_code,
            language=normalized_language,
            description=description,
            embedding=embedding[0],
            tags=tags or []
        )
        
        ctx.info(f"Successfully stored snippet with ID: {snippet_id}")
        return f"Code snippet stored successfully with ID: {snippet_id}"
        
    except ValueError as e:
        ctx.error(f"Validation error: {str(e)}")
        return f"Error: {str(e)}"
    except Exception as e:
        ctx.error(f"Unexpected error: {str(e)}")
        logger.error("Failed to store code snippet", exc_info=True)
        return "An unexpected error occurred"


@mcp.tool()
async def search_code(
    query: str,
    ctx: Context,
    language: str = None,
    limit: int = 5
) -> List[Dict]:
    """Search for similar code snippets using semantic search"""
    try:
        ctx.info(f"Searching for code similar to: {query[:50]}...")
        
        # Normalize language if provided
        normalized_language = normalize_language(language) if language else None
        
        # Generate query embedding
        query_embedding = await ctx.app.embedder.encode_async([query])
        
        # Search in storage
        results = await ctx.app.storage.search_similar(
            query_embedding=query_embedding[0],
            language=normalized_language,
            limit=limit
        )
        
        ctx.info(f"Found {len(results)} matching snippets")
        return results
        
    except Exception as e:
        ctx.error(f"Search failed: {str(e)}")
        logger.error("Failed to search code", exc_info=True)
        return []


@mcp.tool()
async def list_languages(ctx: Context) -> List[Dict]:
    """List all programming languages with stored snippets"""
    try:
        ctx.info("Retrieving language statistics")
        
        # Get languages from storage
        languages = await ctx.app.storage.list_languages()
        
        ctx.info(f"Found {len(languages)} languages with stored snippets")
        return languages
        
    except Exception as e:
        ctx.error(f"Failed to list languages: {str(e)}")
        logger.error("Failed to list languages", exc_info=True)
        return []


@mcp.tool()
async def get_snippet_by_id(snippet_id: int, ctx: Context) -> Dict:
    """Retrieve a specific code snippet by ID"""
    try:
        # Validate snippet ID
        if snippet_id <= 0:
            raise ValueError("Snippet ID must be a positive integer")
            
        ctx.info(f"Retrieving snippet with ID: {snippet_id}")
        
        # Retrieve snippet from storage
        snippet = await ctx.app.storage.get_snippet(snippet_id)
        
        if snippet is None:
            ctx.error(f"No snippet found with ID: {snippet_id}")
            return {"error": f"No code snippet found with ID {snippet_id}"}
        
        ctx.info(f"Successfully retrieved snippet {snippet_id}")
        return snippet
        
    except ValueError as e:
        ctx.error(f"Validation error: {str(e)}")
        return {"error": str(e)}
    except Exception as e:
        ctx.error(f"Failed to retrieve snippet: {str(e)}")
        logger.error(f"Failed to retrieve snippet {snippet_id}", exc_info=True)
        return {"error": "An unexpected error occurred"}


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
