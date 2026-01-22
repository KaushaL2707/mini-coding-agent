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
SUPPORTED_EXTENSIONS = {".py", ".ts", ".js", ".jsx", ".tsx", ".java", ".go", ".rs", ".cpp", ".c", ".h"}

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

# ============ EMBEDDING SETTINGS ============
# Embedding model (using sentence-transformers)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ============ RETRIEVAL SETTINGS ============
# Number of chunks to retrieve for context
TOP_K_CHUNKS = 10

# ============ LLM SETTINGS ============
# LLM Provider: "openai", "anthropic", or "groq"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# Model names per provider
LLM_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-haiku-20240307",
    "groq": "llama-3.1-8b-instant",
}

# API Keys (from environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ============ AGENT SETTINGS ============
# Max iterations for agent loop (future use)
MAX_ITERATIONS = 5
