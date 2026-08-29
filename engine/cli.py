import argparse
import sys
import time
from pathlib import Path

from engine.runtime import Runtime
from engine.agent import Agent



def print_help_commands() -> None:
    print(
        """
        Available commands within the chat:
        /help — show this help
        /clear — clear history and KV-cache
        /system <text> — set a new system prompt (resets the dialog)
        /temp <float> — change the temperature (e.g., /temp 0.3)
        /history — show the current message history
        /exit, /quit — exit
        """
    )

def chat_loop(runtime: Runtime, temperature: float = 0.7) -> None:
    print("The chat is running. Enter /help to get a list of commands. \n")
    while True:
        try: user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt): print("\nExit."); break
        if not user_input: continue
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("/exit", "/quit"): print("Bye."); break
            elif cmd == "/help": print_help_commands()
            elif cmd == "/clear": runtime.clear(); print("History and cache cleared")
            elif cmd == "/system":
                if not arg:
                    print("Using: /system <new system prompt>")
                    continue
                runtime.start_conversation(system_prompt=arg)
                print("[system prompt was installed, diaglog is reset]")
            elif cmd == "/temp":
                try:
                    temperature = float(arg)
                    print(f"[temperature = {temperature}]")
                except ValueError:
                    print(f"Need float like: /temp 0.5")
            elif cmd == "/history":
                if not runtime.messages:
                    print("(history is clear)")
                else:
                    for i, msg in enumerate(runtime.messages, 1):
                        role = msg["role"]
                        content = msg["content"]
                        preview = content if len(content) < 120 else content[:117] + "..."
                        print(f"{i}. [{role}] {preview}")
            else:
                print(f"Unknow command: {cmd}. Enter /help")
            continue
        t0 = time.time()
        try:
            response = runtime.generate(
                user_text=user_input,
                max_new_tokens=150,
                temperature=temperature
            )
        except Exception as e:
            print(f"\n[Error generation] {e}")
            continue
        dt = time.time() - t0
        print(f"\nAssistant: {response}")
        print(f"({dt:.1f}s, start_pos={runtime.start_pos})\n")


def agent_loop(agent: Agent, temperature: float = 0.3) -> None:
    print("Agent model. Every prompt is separate task")
    print("Commands: /exit, /quit\n")
    while True:
        try: user_input = input("Task: ").strip()
        except (EOFError, KeyboardInterrupt): print("\nExit"); break
        if not user_input: continue
        if user_input in ("/exit", "/quit"): print("Bye."); break
        t0 = time.time()
        try: answer = agent.run(user_input, temperature=temperature)
        except Exception as e: print(f"\n [Error agent] {e}"); import traceback; traceback.print_exc(); continue
        dt = time.time() - t0
        print(f"\n === Final Answer ===\n {answer} \n ({dt:.1f}s)\n")


def main(): # TODO make choose model via list of models
    parser = argparse.ArgumentParser(
        prog="engine",
        description="Local inference-engine on tinygrad + simple agent"
    )
    subparser = parser.add_subparsers(dest="command", required=True)
    
    # --- chat ---
    chat_parser = subparser.add_parser("chat", help="Simple chat")
    chat_parser.add_argument("model", type=str, help="Path to folder with model")
    chat_parser.add_argument("--system", type=str, default="You are a helpful assistant.", help="System prompt")
    chat_parser.add_argument("--temp", type=float, default=0.7, help="Temperature")
    
    # --- agent ---
    agent_parser = subparser.add_parser("agent", help="ReAct-agent with tools (calculator, time, weather)")
    agent_parser.add_argument("model", type=str, help="Path to folder with model")
    agent_parser.add_argument("--temp", type=float, default=0.3, help="Temperature (for agent lower is better)")
    agent_parser.add_argument("--max-steps", type=int, default=6, help="ReAct-cicle max steps")
    agent_parser.add_argument("--project", type=str, default=".", help="Project root for fs/git tools")
    agent_parser.add_argument("--yes", action="store_true", help="Auto-confirm CONFIRM-risk tool")
    
    

    args = parser.parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"Folder with model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {model_path} ...")
    runtime = Runtime(model_path)
    print("Model ready for user.\n")
    if args.command == "chat":
        runtime.start_conversation(system_prompt=args.system)
        chat_loop(runtime, temperature=args.temp)
    elif args.command == "agent":
        agent = Agent(runtime, max_steps=args.max_steps, project_root=args.project, auto_confirm=args.yes)
        agent_loop(agent, temperature=args.temp)


if __name__ == "__main__":
    main()
