# AI Data Directory

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
```