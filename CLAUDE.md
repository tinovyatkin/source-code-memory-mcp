# Development Guidelines for Code Memory MCP Server

This document contains specific instructions and best practices for developing the Code Memory MCP Server. Follow these guidelines to ensure consistent, high-quality code.

## Core Development Tools

### 1. Code Quality - Use Ruff
- **Always use `ruff` for both linting and formatting**
- Run before every commit:
  ```bash
  ruff check . --fix  # Linting with auto-fix
  ruff format .       # Code formatting
  ```
- Configure in `pyproject.toml`:
  ```toml
  [tool.ruff]
  line-length = 88
  target-version = "py311"
  
  [tool.ruff.lint]
  select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "DTZ", "T10", "ICN", "PIE", "PYI", "PT", "RET", "SIM", "TCH", "INT", "ERA", "PD", "PGH", "PL", "NPY", "RUF"]
  ignore = ["E501"]  # Line too long - handled by formatter
  
  [tool.ruff.format]
  quote-style = "double"
  indent-style = "space"
  ```

### 2. Dependency Management - Use UV
- **Always use `uv` for dependency management**
- Common commands:
  ```bash
  uv pip install -e .           # Install project in editable mode
  uv pip install -r requirements.txt  # Install from requirements
  uv pip compile pyproject.toml -o requirements.txt  # Generate requirements
  uv pip sync requirements.txt  # Sync exact versions
  ```
- Add new dependencies to `pyproject.toml`, not directly via pip
- Use `uv` for virtual environment management:
  ```bash
  uv venv .venv
  source .venv/bin/activate  # or .venv\Scripts\activate on Windows
  ```

## Type Safety Requirements

### 3. Static Typing - No Any Types
- **Every function, method, class, and parameter MUST have explicit type annotations**
- **Never use `Any` type** - always be specific
- Examples of proper typing:

```python
from typing import Dict, List, Optional, Union, Tuple, Literal, TypeVar, Protocol
from collections.abc import Sequence, Mapping, Callable, Awaitable
import numpy as np
from numpy.typing import NDArray

# Function parameters and returns
async def store_code_snippet(
    code: str,
    language: str,
    description: str,
    tags: List[str],
    metadata: Optional[Dict[str, Union[str, int, float]]] = None
) -> Tuple[int, str]:
    ...

# Class attributes
class CodeEmbedder:
    model_name: str
    device: Literal["cpu", "cuda"]
    model: Optional[SentenceTransformer]
    embedding_dim: int
    
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-code") -> None:
        ...

# Complex types
EmbeddingVector = NDArray[np.float32]
SnippetID = int
SearchResult = Dict[Literal["id", "code", "language", "similarity"], Union[int, str, float]]

# Generic types
T = TypeVar("T")
def batch_process(items: Sequence[T], batch_size: int) -> List[List[T]]:
    ...

# Protocol for type checking
class Embeddable(Protocol):
    def to_text(self) -> str: ...
    def get_language(self) -> str: ...
```

### 4. Type Checking
- Run mypy before commits:
  ```bash
  mypy src/ --strict --no-implicit-optional
  ```
- Configure in `pyproject.toml`:
  ```toml
  [tool.mypy]
  python_version = "3.11"
  strict = true
  no_implicit_optional = true
  warn_return_any = true
  warn_unused_ignores = true
  disallow_untyped_defs = true
  ```

## Best Practices Research

### 5. Stay Current with GitHub
- **When implementing new features or unsure about best practices:**
  1. Search GitHub for recent implementations using the MCP GitHub tool
  2. Look for repositories updated within the last 3-6 months
  3. Check official MCP repositories and examples
  4. Review community implementations with high star counts

- **Search queries to use:**
  ```
  # For MCP server implementations
  mcp server python created:>2024-01-01 language:python

  # For FastMCP usage patterns
  fastmcp tool implementation updated:>2024-06-01

  # For sqlite-vec best practices
  sqlite-vec vector search python stars:>10

  # For embedding model usage
  jina embeddings code sentence-transformers updated:>2024-01-01
  ```

### 6. Technology Currency
- **Remember: AI training data may be outdated**
- Always verify:
  - Current version numbers of dependencies
  - Latest API changes in MCP SDK
  - Recent updates to sqlite-vec
  - Current best practices for async Python
  - Latest HuggingFace model recommendations

## Development Workflow

### 7. Before Each Implementation Session
1. Check for updates:
   ```bash
   uv pip list --outdated
   ```
2. Search GitHub for recent changes to similar projects
3. Review MCP documentation for updates
4. Run linting and type checking on existing code

### 8. Code Organization Principles
- One class per file when the class is substantial
- Group related utilities in single modules
- Keep tool implementations in the server module
- Separate concerns: storage, embeddings, server logic

### 9. Error Handling Standards
```python
from typing import Optional, Union
from dataclasses import dataclass

@dataclass
class CodeMemoryError(Exception):
    """Base exception with structured error info"""
    message: str
    error_code: str
    details: Optional[Dict[str, str]] = None

# Specific typed exceptions
class EmbeddingError(CodeMemoryError):
    """Raised when embedding generation fails"""
    pass

class StorageError(CodeMemoryError):
    """Raised when storage operations fail"""
    pass

# Usage with proper types
def handle_error(error: CodeMemoryError) -> Dict[str, Union[str, int]]:
    return {
        "error": error.message,
        "code": error.error_code,
        "details": error.details or {}
    }
```

### 10. Async Best Practices
```python
from typing import List, AsyncIterator
from collections.abc import AsyncContextManager

# Proper async context manager typing
class DatabaseConnection(AsyncContextManager["DatabaseConnection"]):
    async def __aenter__(self) -> "DatabaseConnection":
        await self.connect()
        return self
    
    async def __aexit__(self, *args: object) -> None:
        await self.disconnect()

# Async generator with types
async def batch_embeddings(
    texts: List[str], 
    batch_size: int
) -> AsyncIterator[NDArray[np.float32]]:
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        yield await generate_embeddings(batch)
```

## Testing Standards

### 11. Test Requirements
- Every public function must have tests
- Use pytest with async support
- Type check test files too
- Minimum 80% code coverage

### 12. Test Structure
```python
import pytest
from typing import List, Dict
from pytest import FixtureRequest

@pytest.fixture
async def embedder() -> CodeEmbedder:
    """Fixture with proper typing"""
    embedder = CodeEmbedder()
    await embedder.initialize()
    return embedder

async def test_embedding_generation(
    embedder: CodeEmbedder,
    sample_code: str
) -> None:
    """Test with explicit types"""
    result: NDArray[np.float32] = await embedder.encode_async([sample_code])
    assert result.shape == (1, 768)
    assert result.dtype == np.float32
```

## Documentation Standards

### 13. Docstring Requirements
```python
def search_similar_code(
    query: str,
    language: Optional[str] = None,
    limit: int = 5,
    threshold: float = 0.7
) -> List[SearchResult]:
    """Search for similar code snippets using semantic search.
    
    Args:
        query: Natural language query or code snippet to search for
        language: Optional programming language filter (e.g., "python", "javascript")
        limit: Maximum number of results to return
        threshold: Minimum similarity score (0.0 to 1.0)
    
    Returns:
        List of search results, each containing id, code, language, and similarity score
        
    Raises:
        EmbeddingError: If query embedding generation fails
        StorageError: If database search fails
        ValueError: If limit < 1 or threshold not in [0, 1]
    """
```

## Quick Reference Commands

```bash
# Before starting work
uv pip sync requirements.txt
ruff check . --fix
mypy src/ --strict

# During development
ruff format src/code_memory/new_module.py
mypy src/code_memory/new_module.py

# Before committing
ruff check . --fix
ruff format .
mypy src/ --strict
pytest tests/ -v

# Search for best practices (using MCP GitHub tool)
# Example: Search for recent FastMCP implementations
```

## Remember
1. **No `Any` types** - be explicit
2. **Use ruff** for all formatting and linting
3. **Use uv** for all dependency management  
4. **Search GitHub** when unsure about current best practices
5. **Type everything** - parameters, returns, variables, class attributes
6. **Test everything** - with proper type annotations
7. **Document clearly** - with type information in docstrings

This is a living document. Update it as you discover new patterns or best practices through GitHub research.