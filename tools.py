"""
Filesystem and utility helpers for the Mini Coding Agent.
"""
import os
from pathlib import Path
from typing import Optional

from config import IGNORE_DIRS, SUPPORTED_EXTENSIONS, MAX_FILE_SIZE


def should_ignore_path(path: Path) -> bool:
    """Check if a path should be ignored based on directory rules."""
    parts = path.parts
    return any(ignore_dir in parts for ignore_dir in IGNORE_DIRS)


def is_supported_file(path: Path) -> bool:
    """Check if file extension is supported."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def is_file_too_large(path: Path) -> bool:
    """Check if file exceeds max size limit."""
    try:
        return path.stat().st_size > MAX_FILE_SIZE
    except OSError:
        return True


def read_file_content(path: Path) -> Optional[str]:
    """
    Safely read file content with encoding fallback.
    Returns None if file cannot be read.
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
        except (IOError, OSError) as e:
            print(f"Error reading {path}: {e}")
            return None
    
    print(f"Could not decode {path} with any encoding")
    return None


def get_relative_path(file_path: Path, base_path: Path) -> str:
    """Get relative path from base directory."""
    try:
        return str(file_path.relative_to(base_path))
    except ValueError:
        return str(file_path)


def format_file_for_context(file_path: str, content: str) -> str:
    """Format a file's content for LLM context."""
    return f"### File: {file_path}\n```\n{content}\n```\n"


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation (4 chars ≈ 1 token).
    Used for context window management.
    """
    return len(text) // 4


def truncate_content(content: str, max_chars: int = 10000) -> str:
    """Truncate content to max characters with indicator."""
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "\n... [truncated]"
