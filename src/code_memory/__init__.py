"""Code Memory MCP Server

A Model Context Protocol server for storing and retrieving code snippets using semantic search.
"""

__version__ = "0.1.0"
__author__ = "Code Memory Team"
__description__ = "MCP server for semantic code snippet storage and retrieval"

from .server import create_server

__all__ = ["create_server"]
