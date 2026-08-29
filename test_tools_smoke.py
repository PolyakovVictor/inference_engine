import tempfile
from pathlib import Path

from engine.security import SandBox
from engine.tools import build_tools, call_tool, parse_action_input, build_preview, get_tools_prompt
from engine.permission import PermissionManager, RiskLevel

def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "hello.py").write_text("Print('hi')\n")
        sandbox = SandBox(root)
        tools = build_tools(sandbox=sandbox)
        perms = PermissionManager(auto_confirm=True)
        print("=== prompt ===")
        print(get_tools_prompt(tools))
        print("\n=== read_file ===")
        print(call_tool("read_file", tools, path="hello.py"))
        print("\n=== list_dir ===")
        print(call_tool("list_dir", tools, path="."))
        print("\n=== search_code ===")
        print(call_tool("search_code", tools, query="print"))
        print("\n=== write_file (throw permission pipeline) ===")
        tool = tools['write_file']
        kwargs = parse_action_input(tool, '{"path": "hello.py", "content": "print(123)\\n"}')
        preview = build_preview(tool, kwargs, sandbox)
        print("Preview (diff):\n", preview)
        decision = perms.check("write_file", tool.risk, preview)
        assert decision.allowed
        print(call_tool("write_file", tools, **kwargs))
        print("After write:", call_tool("read_file", tools, path="hello.py"))
        print("\n=== sandbox escape attempt ===")
        print(call_tool("read_file", tools, path="../../etc/passwd"))
        print("\n=== git tools present ===")
        print(sorted(tools.keys()))
        assert "git_push" not in tools
        assert "git_reset" not in tools
        print("\nOk: all basic tests passed")

if __name__ == "__main__": main()