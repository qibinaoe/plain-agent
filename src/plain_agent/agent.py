import os
import subprocess

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

# Prompt
ENVIRONMENT_PROMPT = (
    "Windows: the bash tool runs through cmd.exe; use cmd.exe syntax, not Unix "
    "Bash or PowerShell syntax"
    if os.name == "nt"
    else "Unix-like: the bash tool runs the system shell"
)

SYSTEM = (
    f"You are a coding agent at {os.getcwd()}. Environment: {ENVIRONMENT_PROMPT}. "
    "Use bash to solve tasks. Act, don't explain."
)

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
    )

]

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
            print(block.input["command"])
            output = run_bash(block.input["command"])
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




