"""Utility functions for the Code Memory MCP Server."""

import hashlib
import re


def normalize_language(language: str) -> str:
    """Normalize programming language name.

    Args:
        language: Raw language string

    Returns:
        Normalized language name
    """
    # Convert to lowercase and remove common variations
    lang = language.lower().strip()

    # Map common aliases to standard names
    language_map = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "cpp": "c++",
        "cxx": "c++",
        "c#": "csharp",
        "cs": "csharp",
        "golang": "go",
        "yml": "yaml",
        "sh": "shell",
        "bash": "shell",
        "zsh": "shell",
        "ps1": "powershell",
        "rb": "ruby",
        "php": "php",
        "java": "java",
        "kt": "kotlin",
        "rs": "rust",
        "swift": "swift",
        "r": "r",
        "scala": "scala",
        "clj": "clojure",
        "hs": "haskell",
        "ml": "ocaml",
        "fs": "fsharp",
        "vb": "vbnet",
        "pas": "pascal",
        "pl": "perl",
        "lua": "lua",
        "dart": "dart",
        "elm": "elm",
        "ex": "elixir",
        "erl": "erlang",
    }

    return language_map.get(lang, lang)


def extract_code_metadata(code: str, language: str) -> dict[str, list[str]]:
    """Extract metadata from code content.

    Args:
        code: Source code content
        language: Programming language

    Returns:
        Dictionary with extracted metadata (functions, classes, imports, etc.)
    """
    metadata: dict[str, list[str]] = {
        "functions": [],
        "classes": [],
        "imports": [],
        "variables": [],
        "comments": [],
    }

    # Language-specific patterns
    patterns = {
        "python": {
            "functions": r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
            "classes": r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[\(:]",
            "imports": r"(?:from\s+[\w.]+\s+)?import\s+([\w., ]+)",
            "variables": r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=",
            "comments": r"#\s*(.*?)$",
        },
        "javascript": {
            "functions": r"function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(|([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>)",
            "classes": r"class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)",
            "imports": r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]|require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
            "variables": r"(?:let|const|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)",
            "comments": r"//\s*(.*?)$|/\*\s*(.*?)\s*\*/",
        },
        "java": {
            "functions": r"(?:public|private|protected)?\s*(?:static)?\s*\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
            "classes": r"(?:public|private)?\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            "imports": r"import\s+([\w.]+)",
            "variables": r"(?:public|private|protected)?\s*(?:static)?\s*\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[=;]",
            "comments": r"//\s*(.*?)$|/\*\s*(.*?)\s*\*/",
        },
    }

    lang_patterns = patterns.get(
        normalize_language(language), patterns.get("python", {})
    )

    for category, pattern in lang_patterns.items():
        matches = re.findall(pattern, code, re.MULTILINE | re.IGNORECASE)
        if matches:
            # Flatten tuples from regex groups and filter empty strings
            flattened = []
            for match in matches:
                if isinstance(match, tuple):
                    flattened.extend([m.strip() for m in match if m.strip()])
                else:
                    flattened.append(match.strip())
            metadata[category] = list(set(flattened))  # Remove duplicates

    return metadata


def generate_code_hash(code: str) -> str:
    """Generate a hash for code content to detect duplicates.

    Args:
        code: Source code content

    Returns:
        SHA-256 hash of normalized code
    """
    # Normalize code by removing extra whitespace and comments
    normalized = re.sub(r"\s+", " ", code.strip())
    normalized = re.sub(
        r"#.*$", "", normalized, flags=re.MULTILINE
    )  # Remove Python comments
    normalized = re.sub(
        r"//.*$", "", normalized, flags=re.MULTILINE
    )  # Remove JS/Java comments
    normalized = re.sub(
        r"/\*.*?\*/", "", normalized, flags=re.DOTALL
    )  # Remove block comments

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def clean_code_snippet(code: str) -> str:
    """Clean and format code snippet for storage.

    Args:
        code: Raw code content

    Returns:
        Cleaned code content
    """
    # Remove excessive whitespace
    lines = code.split("\n")
    cleaned_lines = []

    for line in lines:
        # Remove trailing whitespace but preserve leading indentation
        cleaned_line = line.rstrip()
        cleaned_lines.append(cleaned_line)

    # Remove empty lines at start and end
    while cleaned_lines and not cleaned_lines[0].strip():
        cleaned_lines.pop(0)
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    return "\n".join(cleaned_lines)


def extract_tags_from_description(description: str) -> list[str]:
    """Extract potential tags from description text.

    Args:
        description: Description text

    Returns:
        List of extracted tags
    """
    # Extract hashtags
    hashtags = re.findall(r"#(\w+)", description.lower())

    # Extract common programming terms
    tech_terms = {
        "api",
        "rest",
        "graphql",
        "database",
        "sql",
        "nosql",
        "authentication",
        "auth",
        "jwt",
        "oauth",
        "security",
        "async",
        "sync",
        "promise",
        "callback",
        "event",
        "algorithm",
        "sort",
        "search",
        "optimization",
        "test",
        "testing",
        "unit",
        "integration",
        "frontend",
        "backend",
        "fullstack",
        "web",
        "mobile",
        "framework",
        "library",
        "package",
        "dependency",
        "docker",
        "kubernetes",
        "deployment",
        "ci",
        "cd",
        "performance",
        "scalability",
        "monitoring",
        "logging",
    }

    words = re.findall(r"\b\w+\b", description.lower())
    extracted_terms = [word for word in words if word in tech_terms]

    # Combine and deduplicate
    all_tags = list(set(hashtags + extracted_terms))
    return sorted(all_tags)


def validate_code_input(
    code: str, language: str, description: str = ""
) -> dict[str, str]:
    """Validate code input parameters.

    Args:
        code: Source code content
        language: Programming language
        description: Optional description

    Returns:
        Dictionary with validation errors (empty if valid)
    """
    errors = {}

    if not code or not code.strip():
        errors["code"] = "Code content cannot be empty"
    elif len(code) > 100000:  # 100KB limit
        errors["code"] = "Code content too large (max 100KB)"

    if not language or not language.strip():
        errors["language"] = "Language cannot be empty"
    elif len(language) > 50:
        errors["language"] = "Language name too long (max 50 characters)"

    if len(description) > 1000:  # 1KB limit for description
        errors["description"] = "Description too long (max 1000 characters)"

    return errors


def format_search_results(results: list[dict[str, object]]) -> str:
    """Format search results for display.

    Args:
        results: List of search result dictionaries

    Returns:
        Formatted string representation
    """
    if not results:
        return "No matching code snippets found."

    formatted = f"Found {len(results)} matching code snippets:\n\n"

    for i, result in enumerate(results, 1):
        similarity = result.get("similarity", 0)
        similarity_pct = (
            float(similarity) * 100 if isinstance(similarity, int | float) else 0
        )
        code = result.get("code", "")
        code_str = str(code) if code else ""
        code_preview = code_str[:100] + "..." if len(code_str) > 100 else code_str

        formatted += f"{i}. [{result['language']}] Similarity: {similarity_pct:.1f}%\n"
        if result.get("description"):
            formatted += f"   Description: {result['description']}\n"
        tags = result.get("tags")
        if tags and isinstance(tags, list | tuple):
            tag_strs = [str(tag) for tag in tags]
            formatted += f"   Tags: {', '.join(tag_strs)}\n"
        formatted += f"   Code: {code_preview}\n"
        formatted += f"   ID: {result['id']}\n\n"

    return formatted
