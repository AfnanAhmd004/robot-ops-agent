"""Run the same triage with Claude as the model (needs `pip install anthropic` and ANTHROPIC_API_KEY).

    python examples/run_with_claude.py --model <model-id> [--heatwave]
"""
from __future__ import annotations

import argparse

from robotops import Agent, AnthropicModel, FleetOps, make_fleet, make_policy, score, tool_specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--heatwave", action="store_true")
    args = ap.parse_args()
    fleet = make_fleet(16, 0.5, seed=7, heatwave_at=0.55 if args.heatwave else None)
    ops = FleetOps({r.id: r for r in fleet})
    specs = tool_specs(ops)
    run = Agent(AnthropicModel(args.model), specs, make_policy(specs), max_turns=25).run(
        "Run the daily fleet health check and open the work orders that are needed.")
    print(run.final, "\n")
    print(score(run, fleet, ops.work_orders, ops.modes))


if __name__ == "__main__":
    main()
