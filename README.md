# 🤖 Mini Coding Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A **local, repo-aware coding agent** that indexes your codebase, performs semantic search, and uses LLMs to answer questions and suggest fixes.

> 🎓 **Built for learning** — understand how real coding agents (like Cursor, Copilot, Cody) work under the hood!

![Demo](https://via.placeholder.com/800x400?text=Mini+Coding+Agent+Demo)

---

## ✨ Features

- 📂 **Smart Indexing** — Scans repositories, filters irrelevant files
- 🧠 **Intelligent Chunking** — Splits code by functions/classes (not arbitrary lines)
- 🔍 **Semantic Search** — Find code by meaning, not just keywords
- 🤖 **Multi-LLM Support** — OpenAI, Anthropic Claude, Groq
- 💾 **Persistent Index** — Re-use embeddings across sessions
- 🖥️ **CLI & Interactive Mode** — Single queries or REPL

---

## 🏗️ How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                         USER PROMPT                          │
│              "Fix the memory leak in audio.py"              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      EMBED PROMPT                            │
│         Convert question to 384-dim vector                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    VECTOR SEARCH (FAISS)                     │
│         Find top-K most similar code chunks                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      BUILD CONTEXT                           │
│         Combine relevant code with user prompt               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        LLM ANALYSIS                          │
│         GPT-4 / Claude / Llama analyzes and responds         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     SUGGESTED FIX / DIFF                     │
│         Actionable code changes in diff format               │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
mini_coding_agent/
├── agent.py         # 🎯 Main orchestrator + CLI
├── ingest.py        # 📂 Repo scanning + code chunking
├── embed.py         # 🧮 Embeddings + FAISS vector store
├── retrieve.py      # 🔍 Semantic search
├── llm.py           # 🤖 LLM wrapper (OpenAI/Anthropic/Groq)
├── tools.py         # 🔧 Filesystem utilities
├── config.py        # ⚙️ Configuration settings
└── requirements.txt # 📦 Dependencies
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/mini-coding-agent.git
cd mini-coding-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set API Key

```bash
# OpenAI (default)
export OPENAI_API_KEY="sk-..."

# OR Anthropic
export ANTHROPIC_API_KEY="..."
export LLM_PROVIDER="anthropic"

# OR Groq (free tier available!)
export GROQ_API_KEY="..."
export LLM_PROVIDER="groq"
```

### 3. Run!

```bash
# Single query
python agent.py --repo /path/to/project --prompt "explain the main function"

# Interactive mode
python agent.py --repo /path/to/project -i
```

---

## 📖 Usage Examples

### Index and Query a Repository

```bash
python agent.py --repo ./my_project --prompt "find potential security issues"
```

### Interactive REPL

```bash
$ python agent.py --repo ./my_project -i

🤖 Mini Coding Agent - Interactive Mode
============================================================
Commands:
  /index   - Re-index the repository
  /quit    - Exit
  /help    - Show this help
============================================================

🔹 Your query: how does authentication work?
📚 Retrieving relevant code...
📄 Found 8 relevant chunks:
   • auth/jwt.py:45-89 (score: 0.72)
   • middleware/auth.py:12-56 (score: 0.68)
   ...

🤖 Analyzing with LLM...

💡 Response:
The authentication system uses JWT tokens...
```

### Force Re-indexing

```bash
python agent.py --repo . --reindex --prompt "what changed in the API?"
```

### Use Different LLM Providers

```bash
# Fast inference with Groq
python agent.py -r . --provider groq -p "explain this function"

# Claude for complex analysis
python agent.py -r . --provider anthropic -p "find all bugs"
```

---

## ⚙️ Configuration

Edit `config.py` to customize behavior:

| Setting | Description | Default |
|---------|-------------|---------|
| `SUPPORTED_EXTENSIONS` | File types to index | `.py`, `.ts`, `.js`, `.go`, etc. |
| `IGNORE_DIRS` | Directories to skip | `node_modules`, `.git`, `venv` |
| `CHUNK_SIZE` | Target chunk size (chars) | `1500` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `TOP_K_CHUNKS` | Chunks to retrieve per query | `10` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |

---

## 🧠 Technical Deep Dive

### Chunking Strategy

**Python files**: Parsed by function/class boundaries
```python
# This becomes ONE chunk:
def authenticate_user(token: str) -> User:
    """Validate JWT and return user."""
    payload = decode_jwt(token)
    return User.from_payload(payload)
```

**Other languages**: Fixed-size chunking with overlap to preserve context

### Vector Search

- **Model**: `all-MiniLM-L6-v2` (384 dimensions, fast)
- **Index**: FAISS `IndexFlatIP` (inner product for cosine similarity)
- **Storage**: Persisted to `.vector_store/` directory

### LLM Prompt Engineering

```python
SYSTEM_PROMPT = """You are a senior software engineer.
Analyze code carefully and provide precise suggestions.
When suggesting fixes:
1. Explain the issue clearly
2. Show exact code changes in diff format
3. Consider edge cases and side effects
"""
```

---

## 🛣️ Roadmap

- [x] Repository ingestion & chunking
- [x] Vector embeddings with FAISS
- [x] Semantic code search
- [x] Multi-provider LLM support
- [x] CLI with interactive mode
- [ ] Auto-apply patches
- [ ] Run tests and iterate on failures
- [ ] JavaScript/TypeScript AST parsing
- [ ] Conversation memory
- [ ] VS Code extension

---

## 🤝 Contributing

Contributions are welcome! Some ideas:

- **Better chunking** for more languages (AST-based)
- **Caching** for LLM responses
- **Streaming** output for long responses
- **Web UI** with FastAPI/Streamlit

---

## 📄 License

MIT License - feel free to use this for learning and building!

---

## 🙏 Acknowledgments

Built to understand how modern coding agents work. Inspired by:
- [Cursor](https://cursor.sh/)
- [GitHub Copilot](https://github.com/features/copilot)
- [Sourcegraph Cody](https://sourcegraph.com/cody)

---

<p align="center">
  <b>⭐ Star this repo if you learned something!</b>
</p>
