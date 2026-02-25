"""
Repository ingestion module - scans and chunks codebase.

STEP 1 & 2: Repo scanning + intelligent chunking.
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP,
    TREESITTER_LANGUAGES, TREESITTER_TARGET_NODES, TREESITTER_CLASS_NODES,
)
from tools import (
    should_ignore_path,
    is_supported_file,
    is_file_too_large,
    read_file_content,
    get_relative_path,
)


@dataclass
class CodeChunk:
    """Represents a chunk of code for embedding."""
    file_path: str      # Relative path to file
    content: str        # The actual code content
    start_line: int     # Starting line number
    end_line: int       # Ending line number
    chunk_type: str     # "function", "class", or "block"
    
    def __repr__(self):
        return f"CodeChunk({self.file_path}:{self.start_line}-{self.end_line}, type={self.chunk_type})"


def load_files(repo_path: str) -> Generator[tuple[Path, str, str], None, None]:
    """
    Walk repository and yield (file_path, relative_path, content) for supported files.
    Skips ignored directories and unsupported file types.
    """
    repo = Path(repo_path).resolve()
    
    if not repo.exists():
        raise ValueError(f"Repository path does not exist: {repo_path}")
    
    for root, dirs, filenames in os.walk(repo):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in {"node_modules", ".git", "venv", ".venv", "__pycache__", ".pytest_cache", "dist", "build", ".next", "target", ".idea", ".vscode"}]
        
        for name in filenames:
            file_path = Path(root) / name
            
            # Skip unsupported or too large files
            if not is_supported_file(file_path):
                continue
            if is_file_too_large(file_path):
                print(f"Skipping large file: {file_path}")
                continue
            
            content = read_file_content(file_path)
            if content:
                relative_path = get_relative_path(file_path, repo)
                yield file_path, relative_path, content


def extract_python_chunks(content: str, file_path: str) -> list[CodeChunk]:
    """
    Extract functions and classes from Python code.
    Falls back to block chunking if no structures found.
    """
    chunks = []
    lines = content.split("\n")
    
    # Patterns for Python
    func_pattern = re.compile(r"^(\s*)(async\s+)?def\s+\w+")
    class_pattern = re.compile(r"^(\s*)class\s+\w+")
    
    current_chunk = []
    current_start = 1
    current_type = "block"
    base_indent = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for function or class definition
        func_match = func_pattern.match(line)
        class_match = class_pattern.match(line)
        
        if func_match or class_match:
            # Save previous chunk if exists
            if current_chunk:
                chunks.append(CodeChunk(
                    file_path=file_path,
                    content="\n".join(current_chunk),
                    start_line=current_start,
                    end_line=i,
                    chunk_type=current_type
                ))
            
            # Start new chunk
            current_start = i + 1
            current_type = "function" if func_match else "class"
            base_indent = len(func_match.group(1) if func_match else class_match.group(1))
            current_chunk = [line]
            i += 1
            
            # Collect the entire function/class body
            while i < len(lines):
                next_line = lines[i]
                # Check if we're still in the function/class
                if next_line.strip() == "":
                    current_chunk.append(next_line)
                    i += 1
                    continue
                
                current_indent = len(next_line) - len(next_line.lstrip())
                if next_line.strip() and current_indent <= base_indent:
                    # Hit a line at same or lower indentation
                    if func_pattern.match(next_line) or class_pattern.match(next_line):
                        break
                    elif current_indent < base_indent:
                        break
                
                current_chunk.append(next_line)
                i += 1
            
            # Don't increment i here, let main loop check for new pattern
            continue
        
        current_chunk.append(line)
        i += 1
    
    # Save remaining chunk
    if current_chunk:
        chunks.append(CodeChunk(
            file_path=file_path,
            content="\n".join(current_chunk),
            start_line=current_start,
            end_line=len(lines),
            chunk_type=current_type
        ))
    
    return chunks


def chunk_by_size(content: str, file_path: str) -> list[CodeChunk]:
    """
    Fallback: chunk by fixed size with overlap.
    Used for languages without specific parsers.
    """
    chunks = []
    lines = content.split("\n")
    
    current_chunk = []
    current_chars = 0
    current_start = 1
    
    for i, line in enumerate(lines, 1):
        current_chunk.append(line)
        current_chars += len(line) + 1  # +1 for newline
        
        if current_chars >= CHUNK_SIZE:
            chunks.append(CodeChunk(
                file_path=file_path,
                content="\n".join(current_chunk),
                start_line=current_start,
                end_line=i,
                chunk_type="block"
            ))
            
            # Overlap: keep last few lines
            overlap_lines = []
            overlap_chars = 0
            for ln in reversed(current_chunk):
                if overlap_chars + len(ln) > CHUNK_OVERLAP:
                    break
                overlap_lines.insert(0, ln)
                overlap_chars += len(ln) + 1
            
            current_chunk = overlap_lines
            current_chars = overlap_chars
            current_start = i - len(overlap_lines) + 1
    
    # Don't forget remaining content
    if current_chunk:
        chunks.append(CodeChunk(
            file_path=file_path,
            content="\n".join(current_chunk),
            start_line=current_start,
            end_line=len(lines),
            chunk_type="block"
        ))
    
    return chunks


# ---------------------------------------------------------------------------
# Tree-sitter chunking (language-aware AST parsing)
# ---------------------------------------------------------------------------

# Cache for loaded tree-sitter parsers
_ts_parsers = {}
_ts_available = None  # None = not checked yet


def _is_treesitter_available() -> bool:
    """Check if tree-sitter is installed."""
    global _ts_available
    if _ts_available is None:
        try:
            import tree_sitter
            _ts_available = True
        except ImportError:
            _ts_available = False
    return _ts_available


def _get_treesitter_parser(lang_name: str):
    """
    Get or create a tree-sitter parser for the given language.
    Returns (parser, language) or raises ImportError.
    """
    if lang_name in _ts_parsers:
        return _ts_parsers[lang_name]

    import importlib
    from tree_sitter import Language, Parser

    # Import the language grammar (e.g. tree_sitter_python)
    module_name = f"tree_sitter_{lang_name}"
    try:
        lang_module = importlib.import_module(module_name)
    except ImportError:
        raise ImportError(
            f"Tree-sitter grammar not installed for {lang_name}. "
            f"Install with: pip install tree-sitter-{lang_name}"
        )

    language = Language(lang_module.language())
    parser = Parser(language)

    _ts_parsers[lang_name] = (parser, language)
    return parser, language


def _extract_class_methods(
    node, content: str, file_path: str, target_types: list
) -> list[CodeChunk]:
    """
    Split a large class node into its individual methods/functions.
    Also includes the class header (name, decorators, docstring) as a chunk.
    """
    chunks = []
    body = None

    # Find the class body node
    for child in node.children:
        if child.type in ("block", "class_body", "declaration_list"):
            body = child
            break

    if not body:
        # No body found — return the whole class as one chunk
        return []

    # Extract class header (everything before the body)
    header_content = content[node.start_byte:body.start_byte].rstrip()
    if header_content.strip():
        chunks.append(CodeChunk(
            file_path=file_path,
            content=header_content,
            start_line=node.start_point[0] + 1,
            end_line=body.start_point[0] + 1,
            chunk_type="class_header",
        ))

    # Extract each method/function in the body
    for child in body.children:
        if child.type in target_types or child.type in (
            "function_definition", "method_definition", "function_declaration",
            "method_declaration", "decorated_definition", "function_item",
        ):
            child_content = content[child.start_byte:child.end_byte]
            chunks.append(CodeChunk(
                file_path=file_path,
                content=child_content,
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                chunk_type=child.type,
            ))

    return chunks


def chunk_with_treesitter(content: str, file_path: str) -> list[CodeChunk] | None:
    """
    Parse code with tree-sitter and extract meaningful chunks.

    Returns:
        List of CodeChunk objects, or None if tree-sitter can't handle this file.
    """
    if not _is_treesitter_available():
        return None

    ext = Path(file_path).suffix.lower()
    lang_name = TREESITTER_LANGUAGES.get(ext)
    if not lang_name:
        return None

    try:
        parser, language = _get_treesitter_parser(lang_name)
    except ImportError:
        return None  # Grammar not installed for this language

    # Parse the source code
    tree = parser.parse(bytes(content, "utf-8"))
    root = tree.root_node

    target_types = TREESITTER_TARGET_NODES.get(lang_name, [])
    if not target_types:
        return None

    chunks = []
    block_lines = []  # Accumulate non-target lines (imports, constants, etc.)
    block_start = 1

    for node in root.children:
        if node.type in target_types:
            # Save any accumulated block lines first
            if block_lines:
                block_content = "\n".join(block_lines).strip()
                if block_content:
                    chunks.append(CodeChunk(
                        file_path=file_path,
                        content=block_content,
                        start_line=block_start,
                        end_line=node.start_point[0],
                        chunk_type="block",
                    ))
                block_lines = []

            # Extract this function/class as a chunk
            node_content = content[node.start_byte:node.end_byte]

            # If it's a large class, split into methods
            if (
                node.type in TREESITTER_CLASS_NODES
                and len(node_content) > CHUNK_SIZE * 2
            ):
                sub_chunks = _extract_class_methods(
                    node, content, file_path, target_types
                )
                if sub_chunks:
                    chunks.extend(sub_chunks)
                    block_start = node.end_point[0] + 2
                    continue

            chunks.append(CodeChunk(
                file_path=file_path,
                content=node_content,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                chunk_type=node.type,
            ))
            block_start = node.end_point[0] + 2
        else:
            # Non-target node — accumulate as block
            node_text = content[node.start_byte:node.end_byte]
            if node_text.strip():
                block_lines.append(node_text)
                if not block_lines[:-1]:  # First block line
                    block_start = node.start_point[0] + 1

    # Save remaining block lines
    if block_lines:
        block_content = "\n".join(block_lines).strip()
        if block_content:
            chunks.append(CodeChunk(
                file_path=file_path,
                content=block_content,
                start_line=block_start,
                end_line=len(content.split("\n")),
                chunk_type="block",
            ))

    # If tree-sitter produced nothing useful, signal fallback
    if not chunks:
        return None

    # Split any remaining oversized chunks by size
    final_chunks = []
    for chunk in chunks:
        if len(chunk.content) > CHUNK_SIZE * 3:
            final_chunks.extend(chunk_by_size(chunk.content, file_path))
        else:
            final_chunks.append(chunk)

    return final_chunks


def chunk_code(content: str, file_path: str) -> list[CodeChunk]:
    """
    Intelligently chunk code based on file type.
    Tries tree-sitter first, falls back to regex/size-based chunking.
    """
    # Try tree-sitter first (best quality, multi-language)
    ts_chunks = chunk_with_treesitter(content, file_path)
    if ts_chunks:
        return ts_chunks

    # Fallback: language-specific regex or size-based chunking
    ext = Path(file_path).suffix.lower()

    if ext == ".py":
        chunks = extract_python_chunks(content, file_path)
        if len(chunks) == 1 and len(chunks[0].content) > CHUNK_SIZE * 2:
            return chunk_by_size(content, file_path)
        return chunks
    else:
        return chunk_by_size(content, file_path)


def ingest_repository(repo_path: str) -> list[CodeChunk]:
    """
    Main ingestion function: scan repo and return all chunks.
    
    Args:
        repo_path: Path to the repository root
        
    Returns:
        List of CodeChunk objects ready for embedding
    """
    all_chunks = []
    file_count = 0
    
    print(f"📂 Ingesting repository: {repo_path}")
    
    for file_path, relative_path, content in load_files(repo_path):
        chunks = chunk_code(content, relative_path)
        all_chunks.extend(chunks)
        file_count += 1
        
        if file_count % 10 == 0:
            print(f"  Processed {file_count} files...")
    
    print(f"✅ Ingested {file_count} files into {len(all_chunks)} chunks")
    return all_chunks


if __name__ == "__main__":
    import sys
    
    # Test ingestion on current directory or specified path
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    chunks = ingest_repository(path)
    
    print("\nSample chunks:")
    for chunk in chunks[:5]:
        print(f"\n{chunk}")
        print(f"Content preview: {chunk.content[:200]}...")
