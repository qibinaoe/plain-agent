import os
import subprocess
from pathlib import Path
from anthropic.types import ToolParam
from dotenv import load_dotenv
from anthropic import Anthropic
load_dotenv(override=True)

try:
    import readline
    # #143 UTF-8 backspace fix for macOS libedit
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    pass

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# init llm
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
WORKDIR = Path.cwd()

# Prompt
ENVIRONMENT_PROMPT = (
    "Windows: the bash tool runs through cmd.exe; use cmd.exe syntax, not Unix "
    "Bash or PowerShell syntax"
    if os.name == "nt"
    else "Unix-like: the bash tool runs the system shell"
)

SYSTEM = (
    f"You are a coding agent at {WORKDIR}. Environment: {ENVIRONMENT_PROMPT}. "
    "Use tools to solve tasks. Act, don't explain."
)



# Tool Execution
def run_bash(command: str) -> str:
    dangerous_command = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous_command):
        return "Error. Dangerous command blocked"
    try:
        r = subprocess.run(args=command, shell=True, check=False, cwd=os.getcwd(),
                           capture_output=True, text=True, errors="replace", timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error. Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit and len(lines) > limit:
            lines = lines[:limit] + [f"...({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        filepath = safe_path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        filepath = safe_path(path)
        text = filepath.read_text(encoding="utf-8")
        if old_text not in text:
            return f"Error: {old_text} not in {path}"
        filepath.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

def run_glob(pattern: str) -> str:
    import glob as g
    try:
        mathches = sorted({
            match for match in g.glob(
                pattern, root_dir=WORKDIR, recursive=True)
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR)
        })
        shown = mathches[:200]
        if len(mathches) > 200:
            shown.append("... (more matches omitted; narrow the pattern)")
        return "\n".join(shown) if shown else "(no matches)"
    except Exception as e:
        return f"Error: {e}"

TOOLS: list[ToolParam] = [
    ToolParam(
        name="bash",
        description="Run a shell command",
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string"
                }
            },
            "required": ["command"]
        }
    ),
    ToolParam(
        name="read_file",
        description="Read file contents",
        input_schema= {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string"
                },
                "limit": {
                    "type": "integer"
                }
            },
            "required": ["path"]
        }
    ),
    ToolParam(
        name="write_file",
        description="Write content to a file.",
        input_schema= {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string"
                },
                "content": {
                    "type": "string"
                }
            },
            "required": ["path", "content"]
        }
    ),
    ToolParam(
        name="edit_file",
        description="Replace exact text in a file once.",
        input_schema= {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string"
                },
                "old_text": {
                    "type": "string"
                },
                "new_text": {
                    "type": "string"
                }
            },
            "required": ["path", "old_text", "new_text"]
        }
    ),
    ToolParam(
        name="glob",
        description="Find files matching a glob pattern; ** matches recursively.",
        input_schema= {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string"
                }
            },
            "required": ["pattern"]
        }
    )

]

TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}

# agent loop
def agent_loop(messages: list):
    while True:
        response = client.messages.create(messages=messages, model=MODEL,
                               system=SYSTEM, tools=TOOLS, max_tokens=8000)
        messages.append({"role": "assistant", "content": response.content})
        tool_calls = [block for block in response.content if block.type == "tool_use"]
        if not tool_calls:
            return
        tool_results = []
        for block in tool_calls:
            print(block.name)
            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"
            print(str(output)[:200])
            print(output[:200])
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output
            })
        messages.append({"role": "user", "content": tool_results})

if __name__ == "__main__":
    history = []
    while True:
        query = input("Enter command:")
        if not query.strip():
            continue
        if query.strip().lower() in ("q", "exit"):
            break

        history.append({
            "role": "user",
            "content": query
        })

        agent_loop(history)
        last_message = history[-1]["content"]
        if isinstance(last_message, list):
            for block in last_message:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()




