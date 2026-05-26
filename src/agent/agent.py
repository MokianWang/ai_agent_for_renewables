"""Core ReAct agent loop: Thought → Action → Observation → repeat (streaming)."""

import sys

from .llm import get_client, chat_stream, parse_action
from .prompts import SYSTEM_PROMPT
from .tools import TOOLS

MAX_ITERATIONS = 10


class ReActAgent:
    """ReAct-pattern agent for power system operation and dispatch."""

    def __init__(self):
        self.client = get_client()
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.history = []

    def run(self, user_query: str) -> str:
        """Run the ReAct loop with streaming output."""
        self.messages.append({"role": "user", "content": user_query})

        for i in range(MAX_ITERATIONS):
            # Stream LLM response token-by-token
            response = ""
            for token in chat_stream(self.client, self.messages):
                sys.stdout.write(token)
                sys.stdout.flush()
                response += token
            print()  # newline after stream

            thought, action, args, final_answer = parse_action(response)

            if final_answer:
                self.messages.append({"role": "assistant", "content": response})
                self.history.append((thought, None, final_answer))
                return final_answer

            if action and action in TOOLS:
                try:
                    observation = TOOLS[action](**(args or {}))
                except Exception as e:
                    observation = f"Error executing {action}: {e}"

                # Print observation visibly
                obs_preview = observation[:500] + ("..." if len(observation) > 500 else "")
                print(f"\n  [Observation]: {obs_preview}\n")

                self.history.append((thought, f"{action}({args})", observation))
                self.messages.append({"role": "assistant", "content": response})
                self.messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}\n\nContinue with your next Thought and Action, or provide your Final Answer."
                })
            else:
                self.messages.append({"role": "assistant", "content": response})
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"Invalid action. Available tools: {', '.join(TOOLS.keys())}.\n"
                        "Use format: Action: tool_name(key=value, ...)"
                    )
                })

        return "Agent reached maximum iterations without a final answer."

    def chat_loop(self):
        """Interactive chat loop for the terminal."""
        print("\n" + "=" * 60)
        print("  Power System AI Operator Assistant")
        print("  Type 'quit' to exit, 'status' for system overview")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye.")
                break
            if user_input.lower() == "status":
                user_input = "Show me the system status and available models"

            print()
            response = self.run(user_input)
            print(f"\n{response}\n")

            self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
