"""Stage 2: Tool registry with @tool decorator + practical tools."""
import inspect
import json
import os
import subprocess
from openai import OpenAI

# ---------- tool registry ----------
TOOLS = []
FUNCS = {}

def tool(func):
    """Register a function as an agent tool. Builds JSON schema from type hints."""
    FUNCS[func.__name__] = func
    sig = inspect.signature(func)
    type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
    properties = {}
    required = []
    for name, param in sig.parameters.items():
        properties[name] = {"type": type_map.get(param.annotation, "string")}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    TOOLS.append({
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    })
    return func

# ---------- tools ----------
@tool
def calc(expr: str) -> str:
    """Evaluate an arithmetic expression."""
    return str(eval(expr, {"__builtins__": {}}, {}))

@tool
def read_file(path: str) -> str:
    """Read and return the content of a text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file, overwriting if it exists."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} bytes to {path}"

@tool
def list_dir(path: str) -> str:
    """List files and directories in a directory."""
    entries = sorted(os.listdir(path))
    return "\n".join(entries) if entries else "(empty)"

@tool
def run_shell(cmd: str) -> str:
    """Run a shell command and return its output. Times out after 30s."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    output = result.stdout + result.stderr
    return output.strip() or f"(exit code {result.returncode}, no output)"

# ---------- agent ----------
class Agent:
    def __init__(self, system_prompt, model=None):
        self.client = OpenAI()
        self.model = model or os.getenv("MODEL", "gpt-4o-mini")
        self.messages = [{"role": "system", "content": system_prompt}]

    def chat(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        while True:
            resp = self.client.chat.completions.create(
                model=self.model, messages=self.messages, tools=TOOLS
            )
            msg = resp.choices[0].message
            self.messages.append(msg.model_dump(exclude_none=True))
            if not msg.tool_calls:
                return msg.content
            for tc in msg.tool_calls:
                result = FUNCS[tc.function.name](**json.loads(tc.function.arguments))
                self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

# ---------- REPL ----------
def main():
    agent = Agent("You are a helpful assistant. Use tools when needed.")
    print("Agent ready. Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
        print("Agent:", end=" ", flush=True)
        print(agent.chat(user_input), "\n")

if __name__ == "__main__":
    main()
