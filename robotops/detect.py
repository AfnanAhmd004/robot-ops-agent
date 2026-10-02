"""Drift detectors for telemetry, and an evaluation of detection delay vs false alarms.

Baselines are learned from an initial window assumed healthy (commissioning data).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Alarm:
    signal: str
    index: int | None  # first sample where the alarm fires (None = no alarm)
    direction: int  # +1 rising, -1 falling
    score: float


def _baseline(x: np.ndarray, frac: float) -> tuple[float, float]:
    w = x[: max(20, int(len(x) * frac))]
    return float(np.mean(w)), float(np.std(w) + 1e-9)


def threshold_detector(x: np.ndarray, z: float = 4.0, persist: int = 3, baseline_frac: float = 0.2) -> tuple[int | None, int]:
    """Alarm when |z-score| exceeds ``z`` for ``persist`` consecutive samples."""
    mu, sd = _baseline(x, baseline_frac)
    zs = (x - mu) / sd
    run_up = run_dn = 0
    for i, v in enumerate(zs):
        run_up = run_up + 1 if v > z else 0
        run_dn = run_dn + 1 if v < -z else 0
        if run_up >= persist:
            return i, +1
        if run_dn >= persist:
            return i, -1
    return None, 0


def cusum_detector(x: np.ndarray, k: float = 0.5, h: float = 25.0, baseline_frac: float = 0.2) -> tuple[int | None, int]:
    """Two-sided CUSUM on standardised residuals (Page, 1954): accumulates small persistent shifts."""
    mu, sd = _baseline(x, baseline_frac)
    zs = (x - mu) / sd
    hi = lo = 0.0
    for i, v in enumerate(zs):
        hi = max(0.0, hi + v - k)
        lo = max(0.0, lo - v - k)
        if hi > h:
            return i, +1
        if lo > h:
            return i, -1
    return None, 0


def detrend_daily(x: np.ndarray, period: int = 480) -> np.ndarray:
    """Remove the known duty-cycle sinusoid (least squares on sin/cos at that period)."""
    t = np.arange(len(x))
    A = np.c_[np.sin(2 * np.pi * t / period), np.cos(2 * np.pi * t / period), np.ones_like(t, dtype=float)]
    n0 = max(period, int(len(x) * 0.2))
    coef, *_ = np.linalg.lstsq(A[:n0], x[:n0], rcond=None)
    return x - A[:, :2] @ coef[:2]


def scan(telemetry: dict[str, np.ndarray], method: str = "cusum", upto: int | None = None) -> list[Alarm]:
    out = []
    for sig, x in telemetry.items():
        x = x[:upto] if upto else x
        if sig == "motor_temp":
            x = detrend_daily(x)
        idx, d = cusum_detector(x) if method == "cusum" else threshold_detector(x)
        out.append(Alarm(sig, idx, d, 0.0 if idx is None else float(len(x) - idx)))
    return out


def evaluate_detectors(fleet, methods=("threshold", "cusum")) -> dict[str, dict[str, float]]:
    """Per method: detection rate on faulty robots, false-alarm rate on healthy ones, and how early
    faults are caught (median delay after onset, and the share of the degradation period still ahead)."""
    from .fleet import FAULTS

    res = {}
    for m in methods:
        delays, ahead, detected, faulty, false_alarms, healthy = [], [], 0, 0, 0, 0
        for r in fleet:
            alarms = {a.signal: a for a in scan(r.telemetry, m)}
            if r.fault is None:
                healthy += 1
                false_alarms += any(a.index is not None for a in alarms.values())
                continue
            faulty += 1
            a = alarms[FAULTS[r.fault][1]]
            if a.index is not None and a.index >= r.onset:
                detected += 1
                n = len(r.telemetry[FAULTS[r.fault][1]])
                delays.append(a.index - r.onset)
                ahead.append(1 - (a.index - r.onset) / (n - r.onset))
        res[m] = {"detection_rate": detected / max(faulty, 1),
                  "false_alarm_rate": false_alarms / max(healthy, 1),
                  "median_delay": float(np.median(delays)) if delays else float("nan"),
                  "median_degradation_ahead": float(np.median(ahead)) if ahead else float("nan")}
    return res
