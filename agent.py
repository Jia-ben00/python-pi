"""Stage 1: Agent class with conversation history + REPL."""
import json
import os
from openai import OpenAI

# ---------- tools (keep calc as example) ----------
TOOLS = [{
    "type": "function",
    "function": {
        "name": "calc",
        "description": "Evaluate an arithmetic expression",
        "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]},
    },
}]

def calc(expr):
    return str(eval(expr, {"__builtins__": {}}, {}))

FUNCS = {"calc": calc}

# ---------- agent ----------
class Agent:
    def __init__(self, system_prompt, model=None):
        self.client = OpenAI()  # OPENAI_API_KEY + optional OPENAI_BASE_URL
        self.model = model or os.getenv("MODEL", "gpt-4o-mini")
        self.messages = [{"role": "system", "content": system_prompt}]

    def chat(self, user_input):
        """Send one user message, run the tool loop, return final text."""
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
    agent = Agent("You are a helpful assistant. Use calc for any math.")
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
