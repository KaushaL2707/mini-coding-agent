"""
Agent tools - actions the coding agent can perform.

Each tool is a callable with a name, description, and parameter schema.
The agent decides which tools to call based on the user's task.
"""
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Tool:
    """A tool the agent can use."""
    name: str
    description: str
    parameters: dict          # {param_name: "type - description"}
    function: Callable


class ToolRegistry:
    """Manages and executes agent tools."""

    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def get(self, name: str):
        return self.tools.get(name)

    def execute(self, name: str, args: dict) -> str:
        """Execute a tool by name with the given arguments."""
        tool = self.tools.get(name)
        if not tool:
            available = ", ".join(self.tools.keys())
            return f"Error: Unknown tool '{name}'. Available tools: {available}"
        try:
            return tool.function(**args)
        except TypeError as e:
            expected = ", ".join(tool.parameters.keys())
            return f"Error: Wrong arguments for {name}. Expected: {expected}. Got: {list(args.keys())}. ({e})"
        except Exception as e:
            return f"Error executing {name}: {e}"

    def get_tool_descriptions(self) -> str:
        """Format all tool descriptions for the LLM system prompt."""
        lines = []
        for tool in self.tools.values():
            params = ", ".join(f"{k}: {v}" for k, v in tool.parameters.items())
            lines.append(f"- **{tool.name}**({params})\n  {tool.description}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def tool_read_file(path: str) -> str:
    """Read the contents of a file."""
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: File '{path}' does not exist."
        if not p.is_file():
            return f"Error: '{path}' is not a file."
        if p.stat().st_size > 100 * 1024:
            return f"Error: File too large ({p.stat().st_size:,} bytes). Max 100 KB."

        for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
            try:
                content = p.read_text(encoding=encoding)
                lines = content.split("\n")
                # Add line numbers for easy reference
                numbered = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
                return f"File: {path} ({len(lines)} lines)\n{'─'*60}\n{numbered}"
            except UnicodeDecodeError:
                continue
        return f"Error: Could not decode file '{path}'."
    except Exception as e:
        return f"Error reading file: {e}"


def tool_write_file(path: str, content: str) -> str:
    """Write content to a file (creates parent directories if needed)."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        existed = p.exists()
        old_size = p.stat().st_size if existed else 0
        p.write_text(content, encoding="utf-8")

        action = "Updated" if existed else "Created"
        return f"{action} {path} ({len(content)} chars, {len(content.splitlines())} lines)"
    except Exception as e:
        return f"Error writing file: {e}"


def tool_run_command(command: str) -> str:
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.getcwd(),
        )
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"STDERR:\n{result.stderr}")
        if result.returncode != 0:
            output_parts.append(f"(exit code: {result.returncode})")

        output = "\n".join(output_parts).strip()

        # Truncate very long command output
        if len(output) > 5000:
            output = output[:5000] + "\n... [output truncated]"

        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error running command: {e}"


def tool_list_directory(path: str = ".") -> str:
    """List files and directories in a path."""
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: Directory '{path}' does not exist."
        if not p.is_dir():
            return f"Error: '{path}' is not a directory."

        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))

        dirs = []
        files = []
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                # Count children
                try:
                    count = sum(1 for _ in entry.iterdir())
                except PermissionError:
                    count = "?"
                dirs.append(f"  📁 {entry.name}/ ({count} items)")
            else:
                size = entry.stat().st_size
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size / (1024*1024):.1f} MB"
                files.append(f"  📄 {entry.name} ({size_str})")

        result = f"Directory: {p.resolve()}\n{'─'*60}\n"
        if dirs:
            result += "Directories:\n" + "\n".join(dirs[:50]) + "\n"
        if files:
            result += "Files:\n" + "\n".join(files[:50]) + "\n"
        if not dirs and not files:
            result += "(empty or only hidden files)\n"

        total = len(dirs) + len(files)
        if total > 50:
            result += f"\n... and {total - 50} more entries"

        return result.strip()
    except Exception as e:
        return f"Error listing directory: {e}"


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------

def create_default_tools() -> ToolRegistry:
    """Create the default set of agent tools."""
    registry = ToolRegistry()

    registry.register(Tool(
        name="read_file",
        description="Read the full contents of a file (with line numbers).",
        parameters={"path": "string - file path to read"},
        function=tool_read_file,
    ))

    registry.register(Tool(
        name="write_file",
        description="Create or overwrite a file with the given content.",
        parameters={
            "path": "string - file path to write",
            "content": "string - complete file content to write",
        },
        function=tool_write_file,
    ))

    registry.register(Tool(
        name="run_command",
        description="Run a shell command and return stdout/stderr. Timeout: 30s.",
        parameters={"command": "string - shell command to execute"},
        function=tool_run_command,
    ))

    registry.register(Tool(
        name="list_directory",
        description="List files and subdirectories in a directory.",
        parameters={"path": "string - directory path (default: current directory)"},
        function=tool_list_directory,
    ))

    return registry
