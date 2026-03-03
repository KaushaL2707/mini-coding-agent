"""
Agent tools - actions the coding agent can perform.

Each tool is a callable with a name, description, and parameter schema.
The agent decides which tools to call based on the user's task.
"""
import os
import subprocess
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

from config import MAX_FILE_SIZE


@dataclass
class Tool:
    """A tool the agent can use."""
    name: str
    description: str
    parameters: dict          # {param_name: "type - description"}
    function: Callable
    requires_confirmation: bool = False  # If True, ask user before executing


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
        if p.stat().st_size > MAX_FILE_SIZE:
            return f"Error: File too large ({p.stat().st_size:,} bytes). Max {MAX_FILE_SIZE // 1024} KB."

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


def tool_edit_file(path: str, old_text: str, new_text: str) -> str:
    """Find and replace a specific text block in a file."""
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: File '{path}' does not exist."
        if not p.is_file():
            return f"Error: '{path}' is not a file."

        # Read current content
        content = None
        for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
            try:
                content = p.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        if content is None:
            return f"Error: Could not decode file '{path}'."

        # Count occurrences
        count = content.count(old_text)
        if count == 0:
            # Show a helpful snippet of the file so the LLM can retry
            preview = content[:500]
            return (
                f"Error: old_text not found in '{path}'. "
                f"Make sure it matches the file EXACTLY (whitespace, indentation, etc.).\n"
                f"File preview:\n{preview}"
            )
        if count > 1:
            return (
                f"Error: old_text appears {count} times in '{path}'. "
                f"Include more surrounding context to make the match unique."
            )

        # Apply the edit
        new_content = content.replace(old_text, new_text, 1)
        p.write_text(new_content, encoding="utf-8")

        # Build a simple diff summary
        old_lines = old_text.strip().splitlines()
        new_lines = new_text.strip().splitlines()
        return (
            f"Edited {path}: replaced {len(old_lines)} line(s) with {len(new_lines)} line(s).\n"
            f"  - Removed: {old_lines[0][:80]}{'...' if len(old_lines[0]) > 80 else ''}\n"
            f"  + Added:   {new_lines[0][:80]}{'...' if len(new_lines[0]) > 80 else ''}"
        )
    except Exception as e:
        return f"Error editing file: {e}"


def tool_run_command(command: str, repo_path: str = ".") -> str:
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=repo_path,
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

def create_default_tools(repo_path: str = ".") -> ToolRegistry:
    """Create the default set of agent tools.

    Args:
        repo_path: Repository root path, used as cwd for run_command.
    """
    registry = ToolRegistry()

    registry.register(Tool(
        name="read_file",
        description="Read the full contents of a file (with line numbers).",
        parameters={"path": "string - file path to read"},
        function=tool_read_file,
    ))

    registry.register(Tool(
        name="write_file",
        description="Create a NEW file or fully overwrite an existing file. For modifying specific parts of a file, use edit_file instead.",
        parameters={
            "path": "string - file path to write",
            "content": "string - complete file content to write",
        },
        function=tool_write_file,
        requires_confirmation=True,
    ))

    registry.register(Tool(
        name="edit_file",
        description="Make a surgical edit to a file by replacing a specific text block. PREFERRED over write_file for modifying existing files — only changes what you specify, leaving the rest untouched. The old_text must match EXACTLY (whitespace included) and appear only once in the file.",
        parameters={
            "path": "string - file path to edit",
            "old_text": "string - exact text to find (must be unique in the file)",
            "new_text": "string - replacement text",
        },
        function=tool_edit_file,
        requires_confirmation=True,
    ))

    registry.register(Tool(
        name="run_command",
        description="Run a shell command and return stdout/stderr. Timeout: 30s.",
        parameters={"command": "string - shell command to execute"},
        function=partial(tool_run_command, repo_path=repo_path),
        requires_confirmation=True,
    ))

    registry.register(Tool(
        name="list_directory",
        description="List files and subdirectories in a directory.",
        parameters={"path": "string - directory path (default: current directory)"},
        function=tool_list_directory,
    ))

    return registry
