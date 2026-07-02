"""
Exercise 4 - Scaling decision KV cache driven autoscaling The setup: given live
metrics for a NIM deployment - current replicas and average KV-cache utilization
- compute the desired replica count to hold utilization near a target, with
bounds, a deadband to stop flapping, rate-limited scale-up, and a scale-down
stabilization window. This is the autoscaler's brain.
"""

from __future__ import annotations
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class ScalePolicy:
    target_utilization: float = 0.7
    min_replicas: int = 1
    max_replicas: int = 20
    tolerance: float = 0.1
    max_scale_up: int | None = None
    scale_down_stabilization_s: float = 300.0

@dataclass(frozen=True)
class Metrics:
    replicas: int
    avg_kv_utilization: float
    seconds_since_last_scale_down: float = float("inf")

@dataclass(frozen=True)
class ScaleDecision:
    desired_replicas: int
    reason: str

def decide(m: Metrics, p: ScalePolicy) -> ScaleDecision:
    if p.target_utilization <= 0:
        raise ValueError("target_utilization must be > 0")

    # Scale-from-zero can't be inferred from utilization — needs a request/queue
    # trigger. Seed to min and let the next signal take over.
    if m.replicas <= 0:
        return ScaleDecision(max(p.min_replicas, 1), "no running replicas; seeding to min")

    ratio = m.avg_kv_utilization / p.target_utilization

    # Deadband: near target => hold, so we don't flap around the setpoint.
    if abs(ratio - 1.0) <= p.tolerance:
        return ScaleDecision(m.replicas, f"within tolerance (util={m.avg_kv_utilization:.2f})")

    # HPA-style proportional target, then clamp to bounds.
    raw = math.ceil(m.replicas * ratio)
    desired = max(p.min_replicas, min(p.max_replicas, raw))

    # Rate-limit scale-UP (protect the GPU budget / cluster).
    if p.max_scale_up is not None and desired > m.replicas:
        desired = min(desired, m.replicas + p.max_scale_up)

    # Scale-DOWN stabilization: don't shrink until we've been stable long enough.
    if desired < m.replicas and m.seconds_since_last_scale_down < p.scale_down_stabilization_s:
        return ScaleDecision(m.replicas,
            f"scale-down suppressed ({m.seconds_since_last_scale_down:.0f}s "
            f"< {p.scale_down_stabilization_s:.0f}s)")

    direction = "up" if desired > m.replicas else "down"
    return ScaleDecision(desired,
        f"scale {direction}: util={m.avg_kv_utilization:.2f} "
        f"target={p.target_utilization:.2f} ratio={ratio:.2f} {m.replicas}->{desired}")
