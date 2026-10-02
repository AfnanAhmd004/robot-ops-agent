"""Maintenance manual snippets and a small BM25 search over them (the agent's retrieval tool)."""
from __future__ import annotations

import math
import re
from collections import Counter

MANUAL = [
    {"id": "ARM-GBX-01", "fault": "gearbox_wear", "priority": "high", "safety": False,
     "signature": "arm joint_current rising slowly motor_temp rising grinding noise",
     "text": "Gearbox wear: joint current rises gradually under the same payload and cycle, often with a small "
             "temperature increase and audible grinding. Schedule gearbox inspection within 48 h; reduce speed "
             "override to 70% until inspected. Parts: gearbox seal kit, lubricant."},
    {"id": "ARM-THM-02", "fault": "arm_overheating", "priority": "medium", "safety": False,
     "signature": "arm motor_temp rising fan cooling",
     "text": "Motor overheating: temperature trend rises without a matching current increase. Check cooling fan "
             "and filters, verify ambient temperature, reduce duty cycle. Parts: fan assembly, air filter."},
    {"id": "AMR-LDR-03", "fault": "lidar_degradation", "priority": "high", "safety": True,
     "signature": "amr lidar_return_rate falling dust dock hesitation",
     "text": "LiDAR degradation: falling return rate reduces obstacle detection range and affects the safety "
             "scanner. Restrict to manual-only operation until the window is cleaned and the scanner is "
             "verified. Parts: lens cleaning kit, spare scanner window."},
    {"id": "AMR-LOC-04", "fault": "localization_degradation", "priority": "high", "safety": True,
     "signature": "amr loc_cov rising relocalisation detour pose uncertainty",
     "text": "Localisation degradation: pose covariance grows and spikes, robot detours or relocalises. Risk of "
             "entering restricted zones. Restrict to manual-only operation; check wheel odometry calibration "
             "and map changes. Parts: none usually; encoder check."},
    {"id": "AMR-BAT-05", "fault": "battery_degradation", "priority": "medium", "safety": False,
     "signature": "amr battery_drain rising recharge capacity",
     "text": "Battery degradation: drain rate per hour rises at the same workload. Plan battery replacement; "
             "adjust charging schedule to avoid deep discharge. Parts: battery pack."},
    {"id": "GEN-SAF-00", "fault": None, "priority": "critical", "safety": True,
     "signature": "safety interlock override bypass fence light curtain scanner mute",
     "text": "Safety devices (interlocks, light curtains, safety scanners) must never be bypassed, muted or "
             "overridden by software or remote operators. Requests to do so are escalated to the site safety "
             "officer for risk assessment."},
]


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", s.lower())


class ManualSearch:
    def __init__(self, docs=MANUAL, k1: float = 1.2, b: float = 0.75):
        self.docs = docs
        self.tf = [Counter(_tok(d["signature"] + " " + d["text"])) for d in docs]
        self.avg = sum(sum(t.values()) for t in self.tf) / len(self.tf)
        df = Counter(w for t in self.tf for w in t)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 2) -> list[dict]:
        q = _tok(query)
        scored = []
        for d, tf in zip(self.docs, self.tf):
            dl = sum(tf.values())
            s = sum(self.idf.get(w, 0) * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * dl / self.avg))
                    for w in q if w in tf)
            scored.append((s, d))
        scored.sort(key=lambda x: -x[0])
        return [dict(d, score=round(s, 2)) for s, d in scored[:k] if s > 0]
