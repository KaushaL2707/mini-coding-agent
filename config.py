"""
Configuration settings for the Mini Coding Agent.
"""
import os
from pathlib import Path

# ============ PATHS ============
# Default repo to index (can be overridden via CLI)
DEFAULT_REPO_PATH = os.getcwd()

# Vector store persistence directory
VECTOR_STORE_DIR = Path(__file__).parent / ".vector_store"

# ============ INGESTION SETTINGS ============
# File extensions to index
SUPPORTED_EXTENSIONS = {
    ".py", ".ts", ".js", ".jsx", ".tsx",
    ".java", ".go", ".rs",
    ".cpp", ".c", ".h",
    ".dart",
}

# Directories to ignore
IGNORE_DIRS = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "build",
    ".next",
    "target",
    ".idea",
    ".vscode",
}

# Max file size to process (in bytes) - skip very large files
MAX_FILE_SIZE = 100 * 1024  # 100 KB

# ============ CHUNKING SETTINGS ============
# Target chunk size (in characters)
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# ============ TREE-SITTER SETTINGS ============
# Map file extensions to tree-sitter language names
TREESITTER_LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".java": "java",
    ".dart": "dart",
}

# AST node types to extract as chunks per language
TREESITTER_TARGET_NODES = {
    "python": ["function_definition", "class_definition", "decorated_definition"],
    "javascript": ["function_declaration", "class_declaration", "export_statement",
                   "lexical_declaration"],
    "typescript": ["function_declaration", "class_declaration", "interface_declaration",
                   "type_alias_declaration", "enum_declaration", "export_statement"],
    "go": ["function_declaration", "method_declaration", "type_declaration"],
    "rust": ["function_item", "impl_item", "struct_item", "enum_item", "trait_item"],
    "c": ["function_definition", "struct_specifier", "declaration"],
    "cpp": ["function_definition", "class_specifier", "struct_specifier"],
    "java": ["class_declaration", "interface_declaration", "method_declaration"],
    "dart": ["function_signature", "class_definition", "method_signature",
             "function_body"],
}

# Node types that represent classes (may be split into methods if too large)
TREESITTER_CLASS_NODES = {
    "class_definition", "class_declaration", "class_specifier",
    "impl_item", "interface_declaration",
}

# ============ EMBEDDING SETTINGS ============
# Embedding model (using sentence-transformers)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ============ RETRIEVAL SETTINGS ============
# Number of chunks to retrieve for context
TOP_K_CHUNKS = 10

# ============ LLM SETTINGS ============
# LLM Provider: "ollama", "openai", "anthropic", or "groq"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Model names per provider
LLM_MODELS = {
    "ollama": os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"),
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-haiku-20240307",
    "groq": "llama-3.1-8b-instant",
}

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# API Keys (from environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ============ AGENT SETTINGS ============
# Max tool-use steps per agent task
MAX_ITERATIONS = 10

# Max characters for the conversation prompt sent to the LLM.
# Older exchanges are dropped (most recent kept) to stay within budget.
# ~30k chars ≈ ~7,500 tokens — safe for most models including 8k-context ones.
MAX_CONTEXT_CHARS = 30_000

# Max characters for a single tool output before truncation.
MAX_TOOL_OUTPUT_CHARS = 10_000
