from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from engine.runtime import Runtime
from engine.tools import call_tool, get_tools_prompt, build_tools, parse_action_input, build_preview
from engine.permission import PermissionManager
from engine.security import SandBox


REACT_SYSTEM = """You are a tool-using assistant. Answer ONLY in one of two formats.

Format 1 (need tool):
Thought: brief
Action: tool_name
Action Input: argument

Format 2 (ready to answer):
Thought: brief
Final Answer: answer

NEVER write Python code, class definitions, imports, or markdown code fences.
NEVER repeat this tool list back. Only use ONE tool per step.

Tools:
{tools}

Examples:

User: What is 12 * 8?
Thought: need calculate
Action: calculator
Action Input: 12 * 8

User: Observation: 96
Thought: done
Final Answer: 12 * 8 = 96

User: show me file engine/tools.py
Thought: need to read the file
Action: read_file
Action Input: engine/tools.py

User: Hello
Thought: greeting
Final Answer: Hello! How can I help?
"""

@dataclass
class AgentStep:
    thought: str | None = None
    action: str | None = None
    action_input: str | None = None
    final_answer: str | None = None
    raw: str = ""

def looks_like_math(text: str) -> bool:
    return bool(re.search(r"\d+\s*[\+\-\*/×x]\s*\d+", text))

def looks_like_time(text: str) -> bool:
    t = text.lower()
    keys = ["time", "what time", "clock"]
    return any(k in t for k in keys)

def looks_like_weather(text: str) -> bool:
    t = text.lower()
    keys = ["weather"]
    return any(k in t for k in keys)

def parse_react_output(text: str) -> AgentStep:
    step = AgentStep(raw=text.strip())

    m = re.search(r"Final Answer:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        answer = m.group(1).strip()
        if answer and "SCOPING" not in answer.upper() and answer.count("\n") < 8:
            step.final_answer = answer
            t = re.search(r"Thought:\s*(.+?)(?=Final Answer:)", text, re.IGNORECASE | re.DOTALL)
            if t:
                step.thought = t.group(1).strip()
        return step

    m_action = re.search(r"Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)", text, re.IGNORECASE)
    m_input = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    m_thought = re.search(r"Thought:\s*(.+?)(?=Action:|$)", text, re.IGNORECASE | re.DOTALL)
    
    if m_action:
        step.action = m_action.group(1).strip()
        if m_thought:
            step.thought = m_thought.group(1).strip()
        if m_input:
            step.action_input = m_input.group(1).strip()
        return step
    cleaned = text.strip()
    bad_markers = ("SCOPING", "```", "Write a", "Sure, here", "multiplication table")
    if cleaned and len(cleaned) < 120 and not any(b.lower() in cleaned.lower() for b in bad_markers) and "Action:" not in cleaned:
        step.final_answer = cleaned
        return step
    return step


class Agent:
    def __init__(self, runtime: Runtime, project_root: str = ".", max_steps: int = 5, auto_confirm: bool = False) -> None:
        self.runtime = runtime
        self.max_steps = max_steps
        self.sandbox = SandBox(project_root)
        self.tools = build_tools(self.sandbox)
        self.permissions = PermissionManager(auto_confirm=auto_confirm)
 
    def run(self, user_query: str, temperature: float = 0.2) -> str:
        if looks_like_math(user_query):
            expr = re.search(r"(\d+\s*[\+\-\*/×x]\s*\d+(?:\s*[\+\-\*/×x]\s*\d+)*)", user_query)
            if expr:
                expression = expr.group(1).replace("×", "*").replace("×", "*")
                result = call_tool("calculator", self.tools, expression=expression)
                return f"{expression} = {result}"
        system = REACT_SYSTEM.format(tools=get_tools_prompt(self.tools))
        self.runtime.start_conversation(system_prompt=system)
        prompt = user_query
        failed = 0
        for step_num in range(1, self.max_steps + 1):
            print(f"\n --- Agent step {step_num} ---")
            raw = self.runtime.generate(user_text=prompt, max_new_tokens=80, temperature=temperature)
            print(f"Model: \n{raw}\n")
            step = parse_react_output(raw)
            if step.final_answer is not None: return step.final_answer
            if step.action:
                failed = 0
                tool_name = step.action.lower()
                raw_input = (step.action_input or "").strip()
                tool = self.tools.get(tool_name)
                if tool is None:
                    observation = f"Error: unknown tool '{tool_name}'"
                else:
                    kwargs = parse_action_input(tool, raw_input)
                    preview = build_preview(tool, kwargs, self.sandbox)
                    decision = self.permissions.check(tool_name, tool.risk, preview)
                    observation = call_tool(tool_name, self.tools, **kwargs) if decision.allowed else f"Blocked: {decision.reason}"
                
                print(f"Observation: {observation}")
                prompt = f"Observation: {observation}"
                continue
            failed += 1
            if failed >= 2: return "Не удалось выполнить запрос: модель не смогла сформировать корректное действие."
            print("[Agent] failed to parse Action or Final Answer")
            prompt = (
                "Answer ONLY in this format:\n"
                "Thought: greeting\n"
                "Final Answer: Hello!"
            )
        return "The agent exceeded the maximum number of steps and was unable to provide a response."
# where tinygrad save a model from example?