from __future__ import annotations

from pathlib import Path

class SecurityError(Exception): pass

class SandBox:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.exists(): raise SecurityError(f"Project root does not exist: {self.root}")
    
    def resolve(self, rel_path: str) -> Path:
        candidate = (self.root / rel_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise SecurityError(f"Path goes beyond the sandbox: {rel_path}")
        return candidate
