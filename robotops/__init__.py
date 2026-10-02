"""robot-ops-agent: an operations agent for robot fleets with a model-independent safety layer."""
from .agent import SYSTEM_PROMPT, Agent, AgentRun, AnthropicModel, HeuristicOpsModel, ScriptedModel
from .detect import cusum_detector, evaluate_detectors, scan, threshold_detector
from .evals import score
from .fleet import FAULTS, INJECTION_NOTE, Robot, make_fleet, simulate_robot
from .knowledge import MANUAL, ManualSearch
from .safety import SafetyPolicy, quarantine_text
from .tools import FleetOps, make_policy, tool_specs

__all__ = [
    "Agent", "AgentRun", "HeuristicOpsModel", "ScriptedModel", "AnthropicModel", "SYSTEM_PROMPT",
    "cusum_detector", "threshold_detector", "scan", "evaluate_detectors", "score",
    "FAULTS", "INJECTION_NOTE", "Robot", "make_fleet", "simulate_robot", "MANUAL", "ManualSearch",
    "SafetyPolicy", "quarantine_text", "FleetOps", "tool_specs", "make_policy",
]
