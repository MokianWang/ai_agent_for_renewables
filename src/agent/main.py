"""CLI entry point for the ReAct agent."""

import argparse
import os
import sys

# Force UTF-8 encoding for Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def main():
    parser = argparse.ArgumentParser(
        description="Power System AI Operator Assistant (ReAct Agent)"
    )
    parser.add_argument(
        "--query", "-q", type=str,
        help="Run a single query and exit",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", default=True,
        help="Start interactive chat mode (default)",
    )
    args = parser.parse_args()

    from src.agent.agent import ReActAgent

    agent = ReActAgent()

    if args.query:
        print(f"Query: {args.query}\n")
        result = agent.run(args.query)
        print(f"Agent: {result}")
    else:
        agent.chat_loop()


if __name__ == "__main__":
    main()
