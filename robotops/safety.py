"""The safety layer: every tool call passes through a policy that does not trust the model.

Tiers
-----
* ``read``     – observe only; allowed.
* ``write``    – creates records (work orders); allowed, rate-limited, audited.
* ``actuate``  – changes how a robot moves; needs a human approval token, EXCEPT
                 actions that only make the robot safer (a protective stop), which are
                 always allowed because delaying them is the risk.
* anything not registered is denied (default deny).

Tool outputs that carry free text (operator notes, logs) are scanned for
instruction-like content and wrapped so the model sees them as quoted data.
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

SAFE_DIRECTION_ACTIONS = {("set_robot_mode", "protective_stop")}
ALLOWED_MODES = {"auto", "manual_only", "protective_stop", "reduced_speed"}
INJECTION = re.compile(r"(ignore (all )?(previous|prior) instructions|you are now|system prompt|"
                       r"call [a-z_]+\(|set_robot_mode|override|disable (the )?safety|bypass)", re.I)


@dataclass
class Decision:
    allowed: bool
    reason: str
    needs_approval: bool = False


@dataclass
class AuditEntry:
    t: float
    tool: str
    args: dict
    decision: str
    reason: str


@dataclass
class SafetyPolicy:
    tiers: dict[str, str]
    max_writes_per_min: int = 30  # stops runaway loops; far above a normal triage run
    approvals: set[str] = field(default_factory=set)  # tokens like "set_robot_mode:amr-03:manual_only"
    audit: list[AuditEntry] = field(default_factory=list)
    _writes: deque = field(default_factory=deque)
    stats: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @staticmethod
    def approval_token(tool: str, args: dict) -> str:
        return f"{tool}:{args.get('robot', '')}:{args.get('mode', '')}"

    def check(self, tool: str, args: dict) -> Decision:
        tier = self.tiers.get(tool)
        if tier is None:
            return self._log(tool, args, Decision(False, "unknown tool (default deny)"))
        if tool == "set_robot_mode" and args.get("mode") not in ALLOWED_MODES:
            return self._log(tool, args, Decision(False, f"mode {args.get('mode')!r} is not permitted for software control"))
        if tier == "read":
            return self._log(tool, args, Decision(True, "read"))
        if tier == "write":
            now = time.monotonic()
            while self._writes and now - self._writes[0] > 60:
                self._writes.popleft()
            if len(self._writes) >= self.max_writes_per_min:
                return self._log(tool, args, Decision(False, "write rate limit exceeded"))
            self._writes.append(now)
            return self._log(tool, args, Decision(True, "write (audited)"))
        if tier == "actuate":
            if (tool, args.get("mode")) in SAFE_DIRECTION_ACTIONS:
                return self._log(tool, args, Decision(True, "safety-increasing action: always allowed"))
            if self.approval_token(tool, args) in self.approvals:
                return self._log(tool, args, Decision(True, "approved by a human"))
            return self._log(tool, args, Decision(False, "requires human approval", needs_approval=True))
        return self._log(tool, args, Decision(False, f"unknown tier {tier!r}"))

    def _log(self, tool: str, args: dict, d: Decision) -> Decision:
        verdict = "allow" if d.allowed else ("pending_approval" if d.needs_approval else "deny")
        self.stats[verdict] += 1
        self.audit.append(AuditEntry(time.time(), tool, dict(args), verdict, d.reason))
        return d

    def audit_jsonl(self) -> str:
        return "\n".join(json.dumps(e.__dict__) for e in self.audit)


def quarantine_text(text: str) -> tuple[str, bool]:
    """Wrap untrusted free text; flag it if it looks like an instruction to the agent."""
    flagged = bool(INJECTION.search(text))
    wrapped = f"<untrusted_data flagged={'true' if flagged else 'false'}>{text}</untrusted_data>"
    return wrapped, flagged
