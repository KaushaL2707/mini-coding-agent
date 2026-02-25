# 🤖 Mini Coding Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A **local, agentic coding assistant** that indexes your codebase, performs semantic search, and uses LLMs with tools to explore, analyze, and modify code — all running on your machine with no API keys required.

> 🎓 **Built for learning** — understand how real coding agents (like Cursor, Copilot, Cody) work under the hood!

---

## ✨ Features

- 🧠 **Agentic Tool Use** — ReAct-style loop: thinks step-by-step, reads/writes files, runs commands
- 🏠 **100% Local** — Runs with Ollama + open-source LLMs, no internet needed
- 💸 **No API Keys** — Free to use, no cloud costs (cloud providers also supported)
- 🌊 **Streaming Output** — See LLM responses token-by-token in real-time
- 💬 **Conversation Memory** — Follow-up questions work across tasks
- 🌳 **Tree-sitter Parsing** — Language-aware code chunking for 10+ languages
- 🔍 **Semantic Search** — Find code by meaning, not just keywords (FAISS + embeddings)
- 🔐 **Safe by Default** — Confirmation prompt before any file write or command execution
- 📦 **Per-Project Index** — Each project gets its own persistent vector store

---

## 🏗️ How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                         USER TASK                           │
│              "Fix the bug in auth.py"                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               SEMANTIC SEARCH (FAISS)                       │
│        Embed prompt → find top-K code chunks                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────────┐
              │    🗨️ THINK (LLM reasons)       │
              │    🔧 ACT   (call a tool)       │
         ┌───▶│    👁️ SEE   (observe result)    │
         │    │    🔁 REPEAT until done        │
         │    └─────────────────┬─────────────────┘
         │                  │
         │    Tools:        │
         │    • read_file    │
         │    • write_file   │
         └─── • run_command  │
              • search_code  │
              • list_dir     │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      FINAL ANSWER                           │
│        Explanation + code changes applied                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
mini-coding-agent/
├── agent.py         # 🎯 Agentic loop (ReAct) + CLI
├── agent_tools.py   # 🔧 Tool definitions (read, write, run, list)
├── ingest.py        # 📂 Repo scanning + tree-sitter / regex chunking
├── embed.py         # 🧮 Embeddings + FAISS vector store
├── retrieve.py      # 🔍 Semantic search + context formatting
├── llm.py           # 🤖 LLM providers (Ollama/OpenAI/Anthropic/Groq)
├── tools.py         # 🛠️ Filesystem utilities
├── config.py        # ⚙️ Configuration settings
└── requirements.txt # 📦 Dependencies
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/KaushaL2707/mini-coding-agent.git
cd mini-coding-agent

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up Ollama (Local LLM — Free & Offline)

```bash
# Install Ollama: https://ollama.com/download
# Then pull a coding model:
ollama pull qwen2.5-coder:7b

# That's it! No API keys needed.
# Ollama runs automatically in the background.
```

<details>
<summary>💡 Other model options</summary>

| Model                   | RAM Needed | Best For                 |
| ----------------------- | ---------- | ------------------------ |
| `qwen2.5-coder:3b`      | ~4 GB      | Low-spec machines        |
| `qwen2.5-coder:7b`      | ~8 GB      | **Recommended default**  |
| `deepseek-coder-v2:16b` | ~16 GB     | Best quality (needs GPU) |
| `codellama:7b`          | ~8 GB      | Alternative              |

Switch models with: `set OLLAMA_MODEL=deepseek-coder-v2:16b`

</details>

<details>
<summary>☁️ Cloud providers (optional)</summary>

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
export LLM_PROVIDER="openai"

# Anthropic
export ANTHROPIC_API_KEY="..."
export LLM_PROVIDER="anthropic"

# Groq (free tier available)
export GROQ_API_KEY="..."
export LLM_PROVIDER="groq"
```

</details>

### 3. Run!

```bash
# Interactive mode (recommended)
python agent.py --repo /path/to/project -i

# Single task
python agent.py --repo /path/to/project -p "find bugs in the auth module"
```

---

## 📖 Usage

### Interactive Mode

```bash
$ python agent.py --repo ./my_project -i

🤖 Mini Coding Agent — Interactive Mode
============================================================
   Provider : qwen2.5-coder:7b (local)
   Repo     : /path/to/my_project
   Max steps: 10 per task
   Memory   : enabled (use /clear to reset)
────────────────────────────────────────────────────────────
Commands:
  /index  — Re-index the repository
  /tools  — List available tools
  /clear  — Clear conversation memory
  /quit   — Exit
============================================================

🔹 Your task: find and fix the division by zero bug

📚 Retrieving initial context from codebase...
   Found 10 relevant chunks:
   • utils/math.py:12-45 (score: 0.82)
   • tests/test_math.py:8-22 (score: 0.71)

────────────────────────────────────────────────────────────
🔄 Step 1/10
   THOUGHT: I need to read the math utility file to find the bug.
   ACTION: read_file
   ACTION_INPUT: {"path": "utils/math.py"}
📋 Result: File: utils/math.py (45 lines)...

────────────────────────────────────────────────────────────
🔄 Step 2/10
   THOUGHT: I see the issue on line 23 - no zero check before division.
🔧 Tool: write_file({"path": "utils/math.py", "content": "..."})
⚠️  About to write to: utils/math.py
   Approve? [y/n]: y
📋 Result: Updated utils/math.py (890 chars, 47 lines)

✅ Agent completed in 3 step(s)

============================================================
💡 Answer: (memory: 1 task(s))
============================================================

I fixed the division by zero bug in utils/math.py by adding...
```

### Available Tools

| Tool             | Description                           | Confirmation Required |
| ---------------- | ------------------------------------- | --------------------- |
| `read_file`      | Read file contents with line numbers  | No                    |
| `write_file`     | Create or overwrite a file            | **Yes** ✋            |
| `run_command`    | Execute a shell command (30s timeout) | **Yes** ✋            |
| `list_directory` | List files and subdirectories         | No                    |
| `search_code`    | Semantic search through indexed code  | No                    |

### Multi-Project Support

Each project gets its own persistent index — no overwriting:

```bash
python agent.py --repo C:\projects\flask-app -i     # → index: "flask-app"
python agent.py --repo C:\projects\react-site -i    # → index: "react-site"
python agent.py --repo . -i                          # → index: "mini-coding-agent"
```

Re-indexing only when you want:

```bash
python agent.py --repo . --reindex -i   # Force re-index from CLI
# or type /index inside interactive mode
```

### Conversation Memory

Follow-up questions work within a session:

```
🔹 Your task: read config.py and explain the settings
💡 Answer: (memory: 1 task(s))
   CHUNK_SIZE is 1500 chars, CHUNK_OVERLAP is 200...

🔹 Your task: change the chunk size to 2000
💡 Answer: (memory: 2 task(s))
   ✅ Updated CHUNK_SIZE to 2000 in config.py    ← remembers the context!

🔹 Your task: /clear
🧹 Conversation history cleared.
```

---

## ⚙️ Configuration

Edit `config.py` to customize:

| Setting                | Description                  | Default                                          |
| ---------------------- | ---------------------------- | ------------------------------------------------ |
| `SUPPORTED_EXTENSIONS` | File types to index          | `.py`, `.ts`, `.js`, `.go`, `.rs`, `.dart`, etc. |
| `IGNORE_DIRS`          | Directories to skip          | `node_modules`, `.git`, `venv`, etc.             |
| `CHUNK_SIZE`           | Target chunk size (chars)    | `1500`                                           |
| `TOP_K_CHUNKS`         | Chunks to retrieve per query | `10`                                             |
| `EMBEDDING_MODEL`      | Sentence transformer model   | `all-MiniLM-L6-v2`                               |
| `LLM_PROVIDER`         | Default LLM provider         | `ollama`                                         |
| `MAX_ITERATIONS`       | Max tool-use steps per task  | `10`                                             |

---

## 🧠 Technical Deep Dive

### Chunking: Tree-sitter vs Regex

The agent uses **tree-sitter** (a real parser) when available, falling back to regex for Python and size-based splitting for other languages.

```
File comes in → check extension → try tree-sitter
                                    ├── ✅ grammar installed → AST-based chunks
                                    └── ❌ not installed → regex/size fallback
```

**Tree-sitter** parses code into an Abstract Syntax Tree and extracts meaningful nodes (functions, classes, methods). This works across **10+ languages**:

```bash
# Install grammars for the languages you use:
pip install tree-sitter-python        # .py
pip install tree-sitter-javascript    # .js, .jsx
pip install tree-sitter-typescript    # .ts, .tsx
pip install tree-sitter-go            # .go
pip install tree-sitter-rust          # .rs
pip install tree-sitter-java          # .java
pip install tree-sitter-c             # .c, .h
pip install tree-sitter-cpp           # .cpp
```

Large classes are automatically split into individual methods for better search granularity.

### Vector Search

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions, fast)
- **Index**: FAISS `IndexFlatIP` (cosine similarity), with numpy fallback
- **Storage**: Persisted to `.vector_store/<project-name>/`

### Agent Loop (ReAct)

```
User prompt → Semantic search for context
            → System prompt with tools + context
            → LLM thinks (streamed to terminal)
            → LLM calls a tool OR gives final answer
            → Tool result fed back to LLM
            → Repeat until FINAL_ANSWER (max 10 steps)
```

### Streaming

Ollama responses are streamed token-by-token. The agent reads Ollama's newline-delimited JSON stream and prints each token as it arrives — no waiting for the full response.

### Safety

- `write_file` and `run_command` require explicit user approval (`y/n` prompt)
- `read_file`, `list_directory`, and `search_code` run without confirmation (read-only)
- Commands have a 30-second timeout
- File reads limited to 100KB

---

## 🗂️ CLI Reference

```
python agent.py [OPTIONS]

Options:
  --repo, -r PATH       Repository to analyze (default: current dir)
  --prompt, -p TEXT      Single task to run
  --interactive, -i      Interactive mode (recommended)
  --provider NAME        LLM provider: ollama, openai, anthropic, groq
  --reindex              Force re-index the repository
  --top-k, -k N          Number of chunks to retrieve (default: 10)
  --index-name NAME      Custom index name (default: auto from folder name)
```

---

<p align="center">
  <b>⭐ Star this repo if you learned something!</b>
</p>
