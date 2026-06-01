"""
Workload-aware autoscaling controller for LLM serving.

Switches between TP-favored (low-latency) and DP-favored (high-throughput)
parallelism regimes based on live fleet metrics. The crossover sits in the
mid-hundreds of concurrent requests; a hysteresis band and cooldown timer
prevent the controller from flapping when load oscillates near it.

Reading guide:
  - Metrics              : five signals sampled each control tick
  - RegimeController     : the decision logic; hysteresis + cooldown live here
  - simulate()           : drives the controller against an arrival trace
  - compare_with_and_without_hysteresis() : reproduces the flapping failure mode
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class Regime(Enum):
    TP_FAVORED = "tp_favored"   # latency-bound; lowest single-request latency
    DP_FAVORED = "dp_favored"   # throughput-bound; max aggregate QPS


@dataclass
class Metrics:
    """Signals from the serving fleet, sampled each control tick."""
    queue_depth: int
    active_concurrency: int
    p95_latency_ms: float
    ttft_ms: float
    gpu_util: float  # 0.0 - 1.0


CROSSOVER_CONCURRENCY = 256    # where TP and DP throughput curves intersect
HYSTERESIS_BAND = 64           # must clear center ± band to switch regimes
COOLDOWN_SECONDS = 30          # min dwell time before another switch is allowed
EWMA_ALPHA = 0.2               # smoothing factor for noisy load signals


class EWMA:
    """Exponentially-weighted moving average; absorbs short bursts."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value


@dataclass
class RegimeController:
    """Decides the active parallelism regime with anti-flap guards."""

    band: int = HYSTERESIS_BAND
    cooldown_s: float = COOLDOWN_SECONDS
    current: Regime = Regime.TP_FAVORED
    last_switch_t: float = 0.0
    smoothed_load: EWMA = field(default_factory=lambda: EWMA(EWMA_ALPHA))

    def update(self, t: float, m: Metrics) -> Regime:
        # Primary signals are queue depth + active concurrency. P95, TTFT,
        # and GPU utilization act as secondary tiebreakers in production;
        # the load aggregate alone is sufficient to show the control logic.
        load = self.smoothed_load.update(m.active_concurrency + m.queue_depth)

        # Cooldown converts a sub-dwell-time burst from a thrash into a no-op.
        if t - self.last_switch_t < self.cooldown_s:
            return self.current

        upper = CROSSOVER_CONCURRENCY + self.band
        lower = CROSSOVER_CONCURRENCY - self.band

        # Hysteresis: switch only when load clears the far side of the band.
        # Inside the band, hold the current regime regardless of direction.
        if self.current is Regime.TP_FAVORED and load > upper:
            self._switch(t, Regime.DP_FAVORED)
        elif self.current is Regime.DP_FAVORED and load < lower:
            self._switch(t, Regime.TP_FAVORED)

        return self.current

    def _switch(self, t: float, target: Regime) -> None:
        self.current = target
        self.last_switch_t = t


# --- Workload + serving response model ------------------------------------

@dataclass
class ArrivalTick:
    t: float
    concurrency: int
    queue_depth: int


def tp_throughput(concurrency: int) -> float:
    """TP regime: low per-request latency; saturates early as comms dominate."""
    return min(concurrency, 200) * 1.0


def dp_throughput(concurrency: int) -> float:
    """DP regime: scales linearly with replicas; weak at low concurrency."""
    return concurrency * 0.7


def achieved_throughput(regime: Regime, concurrency: int) -> float:
    return tp_throughput(concurrency) if regime is Regime.TP_FAVORED else dp_throughput(concurrency)


# --- Simulation harness ---------------------------------------------------

@dataclass
class TickResult:
    t: float
    regime: Regime
    throughput: float
    load: int


def simulate(trace: Iterable[ArrivalTick], controller: RegimeController) -> list[TickResult]:
    results: list[TickResult] = []
    for tick in trace:
        m = Metrics(
            queue_depth=tick.queue_depth,
            active_concurrency=tick.concurrency,
            p95_latency_ms=0.0,
            ttft_ms=0.0,
            gpu_util=0.0,
        )
        regime = controller.update(tick.t, m)
        load = tick.concurrency + tick.queue_depth
        results.append(TickResult(tick.t, regime, achieved_throughput(regime, load), load))
    return results


def compare_with_and_without_hysteresis(
    trace: list[ArrivalTick],
) -> tuple[list[TickResult], list[TickResult]]:
    """Same trace, two controllers. band=0 reproduces the flapping failure mode."""
    with_guards = simulate(trace, RegimeController(band=HYSTERESIS_BAND, cooldown_s=COOLDOWN_SECONDS))
    without_guards = simulate(trace, RegimeController(band=0, cooldown_s=0))
    return with_guards, without_guards
