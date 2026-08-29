from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    SAFE = "safe"
    CONFIRM = "confirm"
    DENIED = "denied"

@dataclass
class Decision:
    allowed: bool
    reason: str = ''

class PermissionManager:
    def __init__(self, auto_confirm: bool = False) -> None:
        self.auto_confirm = auto_confirm
        self.always_allow: set[str] = set()
    
    def check(self, tool_name: str, risk: RiskLevel, preview: str) -> Decision:
        if risk == RiskLevel.DENIED: return Decision(False, "deniend by policy")
        if risk == RiskLevel.SAFE: return Decision(True)
        if self.auto_confirm or tool_name in self.always_allow: return Decision(True)
        print("\n--- Needed confirmation ---")
        print(f"Tool: {tool_name}\n{preview}")
        ans = input("Execute? [y/N/a=always in current session]").strip().lower()
        if ans == "a": self.always_allow.add(tool_name); return Decision(True)
        return Decision(ans=="y", "user decliend" if ans != "y" else "")
