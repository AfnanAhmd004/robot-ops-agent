"""Detection benchmark plus agent triage on two scenarios (fully offline).

    python examples/fleet_triage.py        # ~1 min, writes docs/detection.png
"""
from __future__ import annotations

import os

import numpy as np

from robotops import (Agent, FleetOps, HeuristicOpsModel, cusum_detector, evaluate_detectors, make_fleet, make_policy,
                      score, simulate_robot, threshold_detector, tool_specs)


def detection_benchmark() -> None:
    # Detector thresholds were chosen on fleet seed 1; this is a held-out fleet.
    fleet = make_fleet(300, 0.5, seed=11, injected_robots=0)
    res = evaluate_detectors(fleet)
    healthy = sum(r.fault is None for r in fleet)
    print(f"=== drift detection on {len(fleet)} held-out robots ({healthy} healthy) ===")
    print(f"{'method':<12}{'detected':>10}{'false alarms':>14}{'median delay':>14}{'degradation ahead':>19}")
    for m, r in res.items():
        print(f"{m:<12}{r['detection_rate']:>10.0%}{r['false_alarm_rate']:>14.0%}{r['median_delay']:>11.0f} min"
              f"{r['median_degradation_ahead']:>19.0%}")


def triage(title: str, fleet, fleet_aware: bool) -> None:
    ops = FleetOps({r.id: r for r in fleet})
    specs = tool_specs(ops)
    policy = make_policy(specs)
    run = Agent(HeuristicOpsModel(fleet_aware=fleet_aware), specs, policy).run("Daily fleet health check")
    s = score(run, fleet, ops.work_orders, ops.modes)
    site = sum(w["robot"] == "site" for w in ops.work_orders)
    print(f"{title:<34}{s['detected']:>3}/{s['faulty']:<3}{s['false_positives']:>6}{site:>6}{len(ops.work_orders):>6}"
          f"{s['root_cause_accuracy']:>8.0%}{s['safety_escalated']:>5}/{s['safety_faults']:<3}{s['denied_calls']:>7}"
          f"{s['unsafe_actions_executed']:>8}")


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = simulate_robot("arm-demo", "arm", "gearbox_wear", seed=5, onset_frac=0.45)
    x = r.telemetry["joint_current"]
    i_thr, _ = threshold_detector(x)
    i_cus, _ = cusum_detector(x)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(x, lw=0.6, color="#555", label="joint current (A)")
    ax.axvline(r.onset, color="k", ls=":", label="fault onset")
    ax.axvline(i_cus, color="#2a7", lw=2, label=f"CUSUM alarm (+{i_cus - r.onset} min)")
    ax.axvline(i_thr, color="#c33", lw=2, label=f"threshold alarm (+{i_thr - r.onset} min)")
    ax.set_xlabel("minutes")
    ax.set_title("Gearbox wear: CUSUM catches the drift far earlier at the same false-alarm rate")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    os.makedirs("docs", exist_ok=True)
    fig.savefig("docs/detection.png", dpi=130)


def main() -> None:
    detection_benchmark()
    print("\n=== agent triage, 40 robots ===")
    print(f"{'scenario':<34}{'found':>7}{'FP':>6}{'site':>6}{'WOs':>6}{'cause':>8}{'safety':>9}{'denied':>7}{'unsafe':>8}")
    normal = make_fleet(40, 0.5, seed=7)
    heat = make_fleet(40, 0.5, seed=7, heatwave_at=0.55)
    triage("normal week, per-robot triage", normal, fleet_aware=False)
    triage("heatwave, per-robot triage", heat, fleet_aware=False)
    triage("heatwave, fleet-aware triage", heat, fleet_aware=True)
    plot()
    print("\nsaved docs/detection.png")


if __name__ == "__main__":
    main()
