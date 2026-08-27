from __future__ import annotations

import re
from dataclasses import dataclass

from engine.runtime import Runtime
from engine.tools import call_tool, get_tools_prompt


REACT_SYSTEM = """You are a tool-using assistant. Answer ONLY in one of two formats.
Format 1 (need tool):
Thought: brief
Action: tool_name
Action Input: argument

Format 2 (ready to answer):
Thought: brief
Final Answer: answer

Tools:
{tools}

Examples:

User: What is 12 * 8?
Thought: need to calculate
Action: calculator
Action Input: 12 * 8

User: Observation: 96
Thought: got the result
Final Answer: 12 * 8 = 96

User: What time is it?
Thought: need time
Action: current_time
Action Input:

User: Observation: 2026-08-26 13:05:00
Thought: have time
Final Answer: Now 2026-08-26 13:05:00

User: Hello
Thought: Simple question
Final Answer: Hello! How can I help?

Never write code, tables, or long explanations. Only the above format.
"""

@dataclass
class AgentStep:
    thought: str | None = None
    action: str | None = None
    action_input: str | None = None
    final_answer: str | None = None
    raw: str = ""

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
    if cleaned and len(cleaned) < 120 and not any(b.lower())
        step.final_answer = cleaned
        return step
    return step


class Agent:
    def __init__(self, runtime: Runtime, max_steps: int = 5) -> None:
        self.runtime = runtime
        self.max_steps = max_steps
    
    def run(self, user_query: str, temperature: float = 0.3) -> str:
        system = REACT_SYSTEM.format(tools=get_tools_prompt())
        self.runtime.start_conversation(system_prompt=system)

        prompt = user_query

        for step_num in range(1, self.max_steps + 1):
            print(f"\n --- Agent step {step_num} ---")

            raw = self.runtime.generate(
                user_text=prompt,
                max_new_tokens=80,
                temperature=temperature
            )
            print(f"Model: \n{raw}\n")
            step = parse_react_output(raw)
            if step.final_answer is not None:
                return step.final_answer
            
            if step.action:
                tool_name = step.action.lower()
                raw_input = (step.action_input or "").strip()

                if tool_name == "current_time":
                    observation = call_tool("current_time")
                elif tool_name == "calculator":
                    observation = call_tool("calculator", expression=raw_input)
                elif tool_name == "get_weather":
                    observation = call_tool("get_weather", city=raw_input)
                else: 
                    observation = call_tool(tool_name)
                
                print(f"Observation: {observation}")
                prompt = f"Observation: {observation}"
                continue
            print("[Agent] failed to parse Action or Final Answer")
            prompt = (
                "You've violated the format. Please answer strictly in this format:"
                "Thought: ...\nAction: ...\nAction Input: ...\n"
                "or\nThought: ...\nFinal Answer: ..."
            )
        return "The agent exceeded the maximum number of steps and was unable to provide a response."
# where tinygrad save a model from example?