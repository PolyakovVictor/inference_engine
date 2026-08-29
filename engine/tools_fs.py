from __future__ import annotations
import difflib
from engine.security import SandBox, SecurityError

MAX_READ_BYTES = 200_000
MAX_MATCHES = 200

class FsTools:
    def __init__(self, sandbox: SandBox) -> None:
        self.sandbox = sandbox
 
    def read_file(self, path: str) -> str:
        try: p = self.sandbox.resolve(path)
        except SecurityError as e: return f"Error: {e}"
        if not p.is_file(): return f"Error: not a file: {path}"
        data = p.read_bytes()
        if len(data) > MAX_READ_BYTES: return f"Error: file too large ({len(data)}B)"
        try: return data.decode("utf-8") # TODO to utf-8
        except UnicodeDecodeError: return "Error: binary file"
 
    def list_dir(self, path: str = ".") -> str:
        try: p = self.sandbox.resolve(path)
        except SecurityError as e: return f"Error: {e}"
        if not p.is_dir(): return f"Error: not a directory: {path}"
        skip = {".git", "__pycache__", "node_modules", ".venv"}
        entries = [e for e in sorted(p.iterdir()) if e.name not in skip][:500]
        return "\n".join(f"{'d' if e.is_dir() else 'f'} {e.relative_to(self.sandbox.root)}" for e in entries) or "(empty)"

    def search_code(self, query: str, path: str = ".") -> str:
        try: p = self.sandbox.resolve(path)
        except SecurityError as e: return f"Error: {e}"
        skip = {".git", "__pycache__", "node_modules", ".venv"}
        out = []
        for f in p.rglob("*"):
            if not f.is_file() or f.stat().st_size > MAX_READ_BYTES or any(s in f.parts for s in skip): continue
            try: text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError): continue
            for i, line in enumerate(text.splitlines(), 1):
                if query in line:
                    out.append(f"{f.relative_to(self.sandbox.root)}:{i}: {line.strip()[:200]}")
                    if len(out) >= MAX_MATCHES: return "\n".join(out) + "\n...(truncated)"
        return "\n".join(out) or "No matches"

    def diff_preview(self, path: str, new_content: str) -> str:
        try: p = self.sandbox.resolve(path)
        except SecurityError as e: return f"Error: {e}"
        old = p.read_text(encoding="utf-8") if p.is_file() else ""
        diff = difflib.unified_diff(old.splitlines(keepends=True), new_content.splitlines(keepends=True), fromfile=f"a/{path}", tofile=f"b/{path}")
        return "".join(diff) or "(no changes)"

    def write_file(self, path: str, content: str) -> str:
        try: p = self.sandbox.resolve(path)
        except SecurityError as e: return f"Error: {e}"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} chars -> {path}"
