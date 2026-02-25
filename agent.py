"""
Main Agent module - the agentic reasoning loop.

This brings together all components with a ReAct-style tool-use loop:
  User prompt → Retrieve context → Think → Act (tool) → Observe → Repeat → Answer

Usage:
    python agent.py --repo /path/to/repo --prompt "fix the authentication bug"
    python agent.py --repo /path/to/repo -i   # interactive mode
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

from config import TOP_K_CHUNKS, MAX_ITERATIONS
from embed import VectorStore, index_repository
from retrieve import CodeRetriever
from llm import LLM
from agent_tools import create_default_tools, Tool


# ---------------------------------------------------------------------------
# System prompt template for the agentic loop
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """You are an expert coding agent. You solve coding tasks step by step using the tools available to you.

## Available Tools

{tool_descriptions}

## How to respond

At each step, think about what you need to do, then either use a tool or give your final answer.

**To use a tool**, respond in EXACTLY this format:

THOUGHT: <your reasoning about what to do next>
ACTION: <tool_name>
ACTION_INPUT: <json arguments, e.g. {{"path": "src/main.py"}}>

**When you have enough information to answer**, respond in EXACTLY this format:

THOUGHT: <your reasoning>
FINAL_ANSWER: <your complete, detailed answer to the user>

## Rules
- ALWAYS start with THOUGHT.
- Use tools to gather real information — never guess file contents or command output.
- When modifying code, ALWAYS read the file first to see its current state.
- You can only use ONE tool per step. Wait for the result before deciding the next step.
- Be thorough: explore the code before making changes.
- When writing code changes, include the COMPLETE file content in write_file, not just the diff.

## Repository Info
Working directory: {repo_path}

{initial_context}"""


class CodingAgent:
    """
    The main coding agent with a ReAct-style tool-use loop.

    On each user task, the agent:
    1. Retrieves relevant code via semantic search (initial context)
    2. Enters a Think → Act → Observe loop
    3. Uses tools (read_file, write_file, run_command, etc.) as needed
    4. Produces a final answer when done
    """

    def __init__(
        self,
        repo_path: Optional[str] = None,
        index_name: str = "default",
        llm_provider: Optional[str] = None,
    ):
        self.repo_path = repo_path or "."
        self.index_name = index_name
        self.retriever = CodeRetriever()
        self.llm = LLM(provider=llm_provider)
        self.indexed = False

        # Conversation memory (persists across tasks in a session)
        self.session_history = []  # [{"task": str, "answer": str, "tools_used": [str]}]
        self.max_history_chars = 6000  # Keep history within model context limits

        # Set up tools
        self.tools = create_default_tools()

        # Register search_code tool (it needs the retriever)
        self.tools.register(Tool(
            name="search_code",
            description="Semantic search through the indexed codebase. Use natural language queries to find relevant code.",
            parameters={"query": "string - natural language description of what you're looking for"},
            function=self._search_code,
        ))

    def _search_code(self, query: str) -> str:
        """Search tool backed by the FAISS vector store."""
        if not self.indexed:
            return "Error: Repository not indexed yet. Run /index first."
        return self.retriever.retrieve_as_context(query)

    def _build_history_context(self) -> str:
        """Build a summary of previous tasks for the LLM's context."""
        if not self.session_history:
            return ""

        parts = ["## Conversation History (previous tasks in this session)\n"]
        total_chars = 0

        # Include recent history, trimming from the oldest if too long
        for entry in reversed(self.session_history):
            summary = f"**User task:** {entry['task']}\n"
            if entry.get('tools_used'):
                summary += f"Tools used: {', '.join(entry['tools_used'])}\n"
            # Truncate long answers in history
            answer = entry['answer']
            if len(answer) > 500:
                answer = answer[:500] + "... [truncated]"
            summary += f"**Agent answer:** {answer}\n---\n"

            if total_chars + len(summary) > self.max_history_chars:
                break
            parts.insert(1, summary)  # Insert after header, keeping chronological order
            total_chars += len(summary)

        if len(parts) <= 1:
            return ""  # Nothing fit

        return "\n".join(parts)

    def clear_history(self):
        """Clear the conversation history."""
        self.session_history.clear()
        print("🧹 Conversation history cleared.")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, force: bool = False) -> bool:
        """
        Index the repository. Skips if already indexed (unless force=True).

        Args:
            force: Force re-indexing even if an index exists on disk

        Returns:
            True if indexing was successful
        """
        if not self.repo_path:
            print("❌ No repository path specified")
            return False

        if not force and self.retriever.load_index(self.index_name):
            print(f"📂 Loaded existing index: {self.index_name}")
            self.indexed = True
            return True

        print(f"\n{'='*60}")
        print(f"🔄 Indexing repository: {self.repo_path}")
        print(f"{'='*60}\n")

        try:
            store = index_repository(self.repo_path, self.index_name)
            self.retriever.store = store
            self.indexed = True
            return True
        except Exception as e:
            print(f"❌ Indexing failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response: str):
        """
        Parse the LLM's response to extract a tool call or final answer.

        Returns:
            ("action", tool_name, args_dict)
            ("answer", answer_text, None)
            ("error", error_message, None)
        """
        # Check for FINAL_ANSWER
        if "FINAL_ANSWER:" in response:
            answer = response.split("FINAL_ANSWER:", 1)[1].strip()
            return ("answer", answer, None)

        # Check for ACTION + ACTION_INPUT
        if "ACTION:" in response:
            try:
                after_action = response.split("ACTION:", 1)[1]

                # Split tool name from args
                if "ACTION_INPUT:" in after_action:
                    tool_name = after_action.split("ACTION_INPUT:", 1)[0].strip()
                    args_raw = after_action.split("ACTION_INPUT:", 1)[1].strip()
                else:
                    # No ACTION_INPUT — tool with no arguments
                    tool_name = after_action.strip().split("\n")[0].strip()
                    args_raw = "{}"

                # Clean tool name (take first line only)
                tool_name = tool_name.split("\n")[0].strip()

                # Clean args — strip markdown code fences if present
                args_raw = re.sub(r"^```(?:json)?\s*", "", args_raw, flags=re.MULTILINE)
                args_raw = re.sub(r"\s*```\s*$", "", args_raw, flags=re.MULTILINE)

                # Find the JSON object (handles extra text after the JSON)
                json_match = re.search(r"\{.*?\}", args_raw, re.DOTALL)
                if json_match:
                    args = json.loads(json_match.group())
                else:
                    args = {}

                return ("action", tool_name, args)

            except (json.JSONDecodeError, IndexError, ValueError) as e:
                return ("error", f"Could not parse your tool call. Error: {e}", None)

        # No clear structure — treat the whole response as the answer
        return ("answer", response.strip(), None)

    # ------------------------------------------------------------------
    # Main agent loop
    # ------------------------------------------------------------------

    def run(self, prompt: str, top_k: int = TOP_K_CHUNKS) -> str:
        """
        Run the agentic loop for a user task.

        Args:
            prompt: User's question or coding task
            top_k: Number of chunks to retrieve for initial context

        Returns:
            The agent's final answer
        """
        if not self.indexed:
            if not self.index():
                return "Error: Repository not indexed."

        # ── Step 1: Retrieve initial context via semantic search ──
        print("\n📚 Retrieving initial context from codebase...")
        initial_context = self.retriever.retrieve_as_context(prompt, top_k)

        # Show retrieved chunks summary
        results = self.retriever.retrieve(prompt, top_k)
        if results:
            print(f"   Found {len(results)} relevant chunks:")
            for chunk, score in results[:3]:
                print(f"   • {chunk.file_path}:{chunk.start_line}-{chunk.end_line} (score: {score:.2f})")
            if len(results) > 3:
                print(f"   ... and {len(results) - 3} more")

        # ── Step 2: Build system prompt with tools + context ──
        context_section = ""
        if initial_context and initial_context != "No relevant code found in the repository.":
            context_section = f"## Initial Code Context (from semantic search)\n\n{initial_context}"

        system_prompt = AGENT_SYSTEM_PROMPT.format(
            tool_descriptions=self.tools.get_tool_descriptions(),
            repo_path=self.repo_path,
            initial_context=context_section,
        )

        # ── Step 3: Agent loop ──
        # Include conversation history for follow-up awareness
        history_context = self._build_history_context()
        if history_context:
            conversation = f"{history_context}\n\nCurrent task: {prompt}"
        else:
            conversation = f"User task: {prompt}"

        tools_used = []  # Track tools used in this task

        for step in range(1, MAX_ITERATIONS + 1):
            print(f"\n{'─'*60}")
            print(f"🔄 Step {step}/{MAX_ITERATIONS}")
            print(f"{'─'*60}")

            # Stream LLM response (tokens appear in real-time)
            response = ""
            sys.stdout.write("   ")
            sys.stdout.flush()
            for token in self.llm.provider.generate_stream(conversation, system_prompt):
                sys.stdout.write(token)
                sys.stdout.flush()
                response += token
            print()  # newline after streaming

            # Parse the response
            result_type, value, args = self._parse_response(response)

            if result_type == "answer":
                print(f"\n✅ Agent completed in {step} step(s)")
                # Save to session history
                self.session_history.append({
                    "task": prompt,
                    "answer": value,
                    "tools_used": tools_used,
                })
                return value

            elif result_type == "action":
                # Show tool call
                args_display = json.dumps(args, indent=2) if args else "{}"
                print(f"🔧 Tool: {value}({args_display})")
                tools_used.append(value)

                # Check if tool requires user confirmation
                tool = self.tools.get(value)
                if tool and tool.requires_confirmation:
                    # Show what's about to happen
                    if value == "write_file":
                        print(f"⚠️  About to write to: {args.get('path', '?')}")
                    elif value == "run_command":
                        print(f"⚠️  About to run: {args.get('command', '?')}")

                    try:
                        confirm = input("   Approve? [y/n]: ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        confirm = "n"

                    if confirm not in ("y", "yes"):
                        tool_result = f"User declined this action."
                        print("   ❌ Declined")

                        conversation += (
                            f"\n\nAssistant:\n{response}"
                            f"\n\nObservation: User declined the {value} action. "
                            f"Find an alternative approach or provide your FINAL_ANSWER with the suggested changes instead."
                        )
                        continue

                # Execute the tool
                tool_result = self.tools.execute(value, args)

                # Show truncated result
                preview = tool_result[:300]
                if len(tool_result) > 300:
                    preview += f"... [{len(tool_result)} chars total]"
                print(f"📋 Result:\n{preview}")

                # Append the full exchange to conversation
                conversation += (
                    f"\n\nAssistant:\n{response}"
                    f"\n\nObservation (result of {value}):\n{tool_result}"
                    f"\n\nContinue. Decide your next step: use another tool or give your FINAL_ANSWER."
                )

            elif result_type == "error":
                print(f"⚠️ {value}")
                conversation += (
                    f"\n\nAssistant:\n{response}"
                    f"\n\nSystem: {value}. Please use the correct format:\n"
                    f"THOUGHT: ...\nACTION: tool_name\nACTION_INPUT: {{...}}\n"
                    f"or\nTHOUGHT: ...\nFINAL_ANSWER: ..."
                )

        # Exhausted iterations
        print(f"\n⚠️ Reached max steps ({MAX_ITERATIONS})")
        partial = "I reached the maximum number of steps. Here is what I have so far:\n\n" + response
        # Still save to history so the agent remembers what was attempted
        self.session_history.append({
            "task": prompt,
            "answer": partial[:500],
            "tools_used": tools_used,
        })
        return partial

    # ------------------------------------------------------------------
    # Interactive REPL
    # ------------------------------------------------------------------

    def interactive(self):
        """Run an interactive REPL with the agentic loop."""
        print(f"\n{'='*60}")
        print("🤖 Mini Coding Agent — Interactive Mode")
        print(f"{'='*60}")
        print(f"   Provider : {self.llm.provider.model_name}")
        print(f"   Repo     : {self.repo_path}")
        print(f"   Max steps: {MAX_ITERATIONS} per task")
        print(f"   Memory   : enabled (use /clear to reset)")
        print(f"{'─'*60}")
        print("Commands:")
        print("  /index  — Re-index the repository")
        print("  /tools  — List available tools")
        print("  /clear  — Clear conversation memory")
        print("  /quit   — Exit")
        print(f"{'='*60}\n")

        while True:
            try:
                prompt = input("🔹 Your task: ").strip()

                if not prompt:
                    continue

                if prompt.lower() in ["/quit", "/exit", "/q"]:
                    print("👋 Goodbye!")
                    break

                if prompt.lower() == "/index":
                    self.index(force=True)
                    continue

                if prompt.lower() == "/tools":
                    print("\n🔧 Available tools:\n")
                    print(self.tools.get_tool_descriptions())
                    continue

                if prompt.lower() == "/clear":
                    self.clear_history()
                    continue

                if prompt.lower() == "/help":
                    print("Commands: /index, /tools, /clear, /quit")
                    print("Or type any coding task and the agent will work on it.")
                    continue

                # Run the agentic loop
                answer = self.run(prompt)

                print(f"\n{'='*60}")
                print(f"💡 Answer: (memory: {len(self.session_history)} task(s))")
                print(f"{'='*60}\n")
                print(answer)
                print()

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Mini Coding Agent — Agentic AI assistant with tool use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ask the agent to explore and fix code (it will use tools automatically)
  python agent.py --repo ./my_project --prompt "find and fix the bug in auth.py"

  # Interactive mode (recommended)
  python agent.py --repo ./my_project -i

  # Use a specific LLM provider
  python agent.py --repo . --provider ollama -p "refactor the main function"

  # Force re-indexing
  python agent.py --repo . --reindex -p "what changed recently?"
""",
    )

    parser.add_argument(
        "--repo", "-r",
        type=str,
        default=".",
        help="Path to the repository to analyze (default: current directory)",
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Coding task or question to process",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode (recommended)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["ollama", "openai", "anthropic", "groq"],
        help="LLM provider to use (default: ollama)",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing of the repository",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=TOP_K_CHUNKS,
        help=f"Number of code chunks to retrieve for context (default: {TOP_K_CHUNKS})",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default=None,
        help="Name for the vector store index (default: auto from repo folder name)",
    )

    args = parser.parse_args()

    # Resolve repo path
    repo_path = str(Path(args.repo).resolve())

    # Auto-derive index name from the repo folder name if not provided
    index_name = args.index_name or Path(repo_path).name or "default"
    print(f"📦 Index: {index_name}")

    # Create agent
    agent = CodingAgent(
        repo_path=repo_path,
        index_name=index_name,
        llm_provider=args.provider,
    )

    # Index the repo
    if args.reindex or not agent.retriever.load_index(index_name):
        if not agent.index(force=args.reindex):
            sys.exit(1)
    else:
        agent.indexed = True

    # Run
    if args.interactive:
        agent.interactive()
    elif args.prompt:
        answer = agent.run(args.prompt, args.top_k)
        print(f"\n{'='*60}")
        print("💡 Answer:")
        print(f"{'='*60}\n")
        print(answer)
    else:
        agent.interactive()


if __name__ == "__main__":
    main()
