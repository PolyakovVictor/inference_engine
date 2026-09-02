# tools for agents

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from engine.permission import RiskLevel
from engine.tools_fs import FsTools
from engine.security import SandBox
import engine.tools_shell as shell
import engine.tools_cv as cv


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., str]
    args_schema: str
    risk: RiskLevel = RiskLevel.SAFE
    arg_names: list[str] = field(default_factory=list)


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


def build_tools(sandbox: SandBox) -> dict[str, Tool]:
    fs = FsTools(sandbox)
    root = sandbox.root
    return {
            "calculator": Tool("calculator", "Calc math expressions", lambda expression: _calculator(expression),
                                "expression: str", RiskLevel.SAFE, ["expression"]),
            "current_time": Tool("current_time", "Return current date and time", lambda: _current_time(),
                                "no args", RiskLevel.SAFE, []),
            "get_weather": Tool("get_weather", "Return current weather in the city", lambda city: _get_weather(city),
                                "city: str", RiskLevel.SAFE, ["city"]),

            "read_file": Tool("read_file", "Read a text file from the project", fs.read_file,
                            "path: str", RiskLevel.SAFE, ["path"]),
            "list_dir": Tool("list_dir", "List a directory", fs.list_dir,
                            "path: str (default '.')", RiskLevel.SAFE, ["path"]),
            "search_code": Tool("search_code", "Search a substring across project files", fs.search_code,
                                "query: str", RiskLevel.SAFE, ["query"]),
            "write_file": Tool("write_file", "Overwrite/create a file (needs JSON args)", fs.write_file,
                                'JSON: {"path": str, "content": str}', RiskLevel.CONFIRM, ["path", "content"]),

            "git_status": Tool("git_status", "git status", lambda: shell.git(["status"], root),
                                "no args", RiskLevel.SAFE, []),
            "git_diff": Tool("git_diff", "git diff", lambda: shell.git(["diff"], root),
                            "no args", RiskLevel.SAFE, []),
            "git_log": Tool("git_log", "git log (last 10)", lambda: shell.git(["log", "--oneline", "-10"], root),
                            "no args", RiskLevel.SAFE, []),
            "git_add": Tool("git_add", "git add <path>", lambda path: shell.git(["add", path], root),
                            "path: str", RiskLevel.CONFIRM, ["path"]),
            "git_commit": Tool("git_commit", "git commit -m <message>", lambda message: shell.git(["commit", "-m", message], root),
                                "message: str", RiskLevel.CONFIRM, ["message"]),
            "capture_screen": Tool("capture_screen", "Make a screenshot of the screen or region and describe it", lambda region="full": cv.capture_screen(region),
                                    "region: str (full or x,y,w,h)", RiskLevel.SAFE, ["region"]),
            "mouse_click": Tool("mouse_click", "Click mouse at coordinates", lambda x, y, button="left": cv.mouse_click(x, y, button),
                                "x: int, y: int, button: str", RiskLevel.CONFIRM, ["x", "y", "button"]),
        }

def get_tools_prompt(tools: dict[str, Tool]) -> str:
    lines = ["Tools:"]
    for tool in tools.values():
        lines.append(f"{tool.name}({tool.args_schema})")
    return "\n".join(lines)

def call_tool(name: str, tools: dict[str, Tool], **kwargs: Any) -> str:
    tool = tools.get(name)
    if tool is None: return f"Error: unknown tool '{name}', Allowed: {list(tools.keys())}"
    try: return tool.func(**kwargs)
    except TypeError as e: return f"Error args for {name}: {e}"
    except Exception as e: return f"Error execution {name}: {e}"

def parse_action_input(tool: Tool, raw_input: str) -> dict[str, Any]:
    import json
    raw_input = raw_input.strip()
    if not tool.arg_names: return {}
    if len(tool.arg_names) == 1: return {tool.arg_names[0]: raw_input}
    try:
        data = json.loads(raw_input)
        if not isinstance(data, dict): raise ValueError
        return {k: data[k] for k in tool.arg_names if k in data}
    except (json.JSONDecodeError, ValueError): raise ValueError(f"Awaiting JSON with fields {tool.arg_names}, received: {raw_input}")

def build_preview(tool: Tool, kwargs: dict[str, Any], sandbox: SandBox) -> str:
    if tool.name == "write_file":
        fs = FsTools(sandbox=sandbox)
        return fs.diff_preview(kwargs.get("path", ""), kwargs.get("content", ""))
    return f"{tool.name}({kwargs})"
