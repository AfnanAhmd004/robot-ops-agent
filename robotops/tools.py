"""Tools the agent can call. Each is registered with a safety tier and a JSON schema."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .detect import scan
from .fleet import Robot
from .knowledge import ManualSearch
from .safety import SafetyPolicy, quarantine_text


@dataclass
class ToolSpec:
    name: str
    tier: str
    description: str
    schema: dict
    fn: Callable[..., Any]


@dataclass
class FleetOps:
    """Tool implementations over a (simulated) fleet; swap for real fleet/CMMS APIs."""

    robots: dict[str, Robot]
    upto: int | None = None  # "now" as a sample index (None = end of data)
    work_orders: list[dict] = field(default_factory=list)
    modes: dict[str, str] = field(default_factory=dict)
    manual: ManualSearch = field(default_factory=ManualSearch)

    def list_robots(self) -> list[dict]:
        return [{"robot": r.id, "kind": r.kind, "mode": self.modes.get(r.id, "auto")} for r in self.robots.values()]

    def detect_anomalies(self, robot: str) -> dict:
        r = self.robots[robot]
        alarms = [a for a in scan(r.telemetry, "cusum", self.upto) if a.index is not None]
        return {"robot": robot, "anomalies": [{"signal": a.signal, "direction": "rising" if a.direction > 0 else "falling",
                                                "since_sample": a.index} for a in alarms]}

    def get_telemetry(self, robot: str, signal: str, window: int = 200) -> dict:
        x = self.robots[robot].telemetry[signal][: self.upto]
        recent, ref = x[-window:], x[: max(20, len(x) // 5)]
        return {"robot": robot, "signal": signal, "baseline_mean": round(float(ref.mean()), 4),
                "recent_mean": round(float(recent.mean()), 4),
                "change_pct": round(100 * float(recent.mean() / ref.mean() - 1), 2),
                "recent_trend_per_100": round(float(np.polyfit(np.arange(len(recent)), recent, 1)[0] * 100), 5)}

    def get_operator_notes(self, robot: str) -> dict:
        notes, flags = [], 0
        for n in self.robots[robot].notes:
            w, f = quarantine_text(n)
            notes.append(w)
            flags += f
        return {"robot": robot, "notes": notes, "suspicious_notes": flags}

    def search_manual(self, query: str) -> list[dict]:
        return [{k: d[k] for k in ("id", "fault", "priority", "safety", "text", "score")} for d in self.manual.search(query)]

    def create_work_order(self, robot: str, priority: str, diagnosis: str, actions: list[str], manual_ref: str = "") -> dict:
        if priority not in ("low", "medium", "high", "critical"):
            raise ValueError("priority must be low, medium, high or critical")
        wo = {"id": f"WO-{1000 + len(self.work_orders)}", "robot": robot, "priority": priority,
              "diagnosis": diagnosis, "actions": list(actions), "manual_ref": manual_ref}
        self.work_orders.append(wo)
        return wo

    def set_robot_mode(self, robot: str, mode: str) -> dict:
        self.modes[robot] = mode
        return {"robot": robot, "mode": mode}


def tool_specs(ops: FleetOps) -> list[ToolSpec]:
    s = lambda props, req: {"type": "object", "properties": props, "required": req}
    robot = {"robot": {"type": "string"}}
    return [
        ToolSpec("list_robots", "read", "List robots with kind and current mode.", s({}, []), ops.list_robots),
        ToolSpec("detect_anomalies", "read", "Run drift detection (CUSUM) on all of a robot's telemetry signals.",
                 s(robot, ["robot"]), ops.detect_anomalies),
        ToolSpec("get_telemetry", "read", "Summary statistics and trend for one signal of one robot.",
                 s({**robot, "signal": {"type": "string"}, "window": {"type": "integer"}}, ["robot", "signal"]),
                 ops.get_telemetry),
        ToolSpec("get_operator_notes", "read", "Recent free-text notes from operators (untrusted data).",
                 s(robot, ["robot"]), ops.get_operator_notes),
        ToolSpec("search_manual", "read", "Search the maintenance manual.", s({"query": {"type": "string"}}, ["query"]),
                 ops.search_manual),
        ToolSpec("create_work_order", "write", "Open a maintenance work order.",
                 s({**robot, "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "diagnosis": {"type": "string"}, "actions": {"type": "array", "items": {"type": "string"}},
                    "manual_ref": {"type": "string"}}, ["robot", "priority", "diagnosis", "actions"]),
                 ops.create_work_order),
        ToolSpec("set_robot_mode", "actuate",
                 "Change a robot's operating mode (auto, manual_only, reduced_speed, protective_stop). "
                 "Anything except protective_stop needs human approval.",
                 s({**robot, "mode": {"type": "string"}}, ["robot", "mode"]), ops.set_robot_mode),
    ]


def make_policy(specs: list[ToolSpec], **kw) -> SafetyPolicy:
    return SafetyPolicy({t.name: t.tier for t in specs}, **kw)
