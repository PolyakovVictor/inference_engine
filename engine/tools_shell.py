from __future__ import annotations
import shlex, subprocess
from pathlib import Path


TIMEOUT = 15
MAX_OUT = 8000
READONLY_BINARIES = {"ls", "cat", "grep", "find", "pytest"}


def run_readonly(command: str, cwd: Path) -> str:
    if any(c in command for c in ";&|`$<>\n"): return "Error: banned shell-symbols"
    try: argv = shlex.split(command)
    except ValueError as e: return f"Error: {e}"
    if not argv or argv[0] not in READONLY_BINARIES: return f"Error: command '{argv[0] if argv else ''} is banned"
    try: r = subprocess.run(argv, cwd=str(cwd), capture_output=True, text=True, timeout=TIMEOUT, shell=False)
    except subprocess.TimeoutExpired: return "Error: timeout"
    out = (r.stdout + r.stderr)[:MAX_OUT]
    return out.strip() or f"(exit {r.returncode})"

def git(args: list[str], cwd: Path) -> str:
    try: r = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=TIMEOUT, shell=False)
    except subprocess.TimeoutExpired: return "Error: timeout"
    out = (r.stdout + r.stderr)[:MAX_OUT]
    return out.strip() or f"(exit {r.returncode})"
