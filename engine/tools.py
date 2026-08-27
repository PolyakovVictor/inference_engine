# tools for agents

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., str]
    args_schema: str


def _calculator(expression: str) -> str:
    allowed_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        code = compile(expression, "<calc>", "eval")
        for name in code.co_names:
            if name not in allowed_names: return f"Error: unknow name '{name}'"
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error while calculation: {str(e)}"


def _current_time() -> str: return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _get_weather(city: str) -> str:
    fake = {
        "london": "London +12 C, light rain",
        "New York": "New York +21 C, cloudy",
    }
    key = city.strip().lower()
    return fake.get(key, f"No data about weather in this city '{city}'")


TOOLS: dict[str, Tool] = {
    "calculator": Tool(
        name="calculator",
        description="Calc math expressions",
        func=lambda expression: _calculator(expression),
        args_schema="expression: str # example '2 + 2 * 3' or sqrt(16)"
    ),
    "current_time": Tool(
        name="current_time",
        description="Return current date and time",
        func=lambda: _current_time(),
        args_schema="no args"
    ),
    "get_weather": Tool(
        name="get_weather",
        description="Return current weather in the city",
        func=lambda city: _get_weather(city),
        args_schema="city: str # name of the city"
    )
}

def get_tools_prompt() -> str:
    lines = ["Allowed tools:"]
    for tool in TOOLS.values():
        lines.append(f"- {tool.name}: {tool.description}")
        lines.append(f"   Args: {tool.args_schema}")
    return "\n".join(lines)


def call_tool(name: str, **kwargs: Any) -> str:
    tool = TOOLS.get(name)
    if tool is None: return f"Error: unknown tool '{name}', Allowed: {list(TOOLS.keys())}"
    try: return tool.func(**kwargs)
    except TypeError as e: return f"Error args for {name}: {e}"
    except Exception as e: return f"Error execution {name}: {e}"