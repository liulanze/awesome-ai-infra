"""
PD-aware routing layer.

In disaggregated serving, the router sits between clients and the worker fleet.
For each request it (a) dispatches to a prefill pod by a configurable policy,
then (b) routes to a decode pod after the KV handoff completes. It is the
orchestration layer that makes distributed PD disaggregation actually work.

A naive round-robin load balancer is blind to pod roles and to the
prefill→decode handoff: it cannot orchestrate the cross-pod handoff, and it
is oblivious to queue depth — so under skewed prompt lengths it piles long
prefills onto arbitrary pods while nearby pods sit idle.

On AKS/K8s, the router discovers pods via label selectors and endpoint watches.
When a pod fails its health check, the router evicts it from the pool and new
requests route around the gap automatically.

Reading guide:
  - PodRole, Pod, PodRegistry   : worker pool + K8s-style discovery/health
  - RoutingPolicy, PDRouter     : two-phase dispatch with pluggable prefill policy
  - Request, RequestTrace       : per-request input and recorded timestamps
  - simulate()                  : tick-based harness; drives requests end-to-end
  - compare_policies()          : queue-aware vs round-robin under skewed load
  - make_skewed_workload()      : test workload with mixed short/long prompts
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum


PREFILL_TICKS_PER_TOKEN = 1
DECODE_TICKS_PER_STEP = 1
KV_TRANSFER_TICKS = 2          # cross-pod KV handoff cost (RDMA/NCCL)


# --- Pod abstractions -------------------------------------------------------

class PodRole(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class PodStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


@dataclass
class Pod:
    pod_id: str
    role: PodRole
    status: PodStatus = PodStatus.HEALTHY
    queue_depth: int = 0        # in-flight work items; drives queue-aware routing
    total_processed: int = 0

    def is_available(self) -> bool:
        return self.status == PodStatus.HEALTHY


@dataclass
class PodRegistry:
    """Simulates K8s label-selector discovery and readiness-probe health checks.

    Production: backed by endpoint watches; a pod that fails its readiness
    probe stops receiving new traffic. Here: a mutable dict with explicit
    mark_unhealthy() so the simulation can model mid-run pod failures.
    """
    _pods: dict[str, Pod] = field(default_factory=dict)

    def register(self, pod: Pod) -> None:
        self._pods[pod.pod_id] = pod

    def deregister(self, pod_id: str) -> None:
        self._pods.pop(pod_id, None)

    def mark_unhealthy(self, pod_id: str) -> None:
        """Simulate a failed health check. In-flight work completes; new
        requests no longer route to this pod.
        """
        if pod_id in self._pods:
            self._pods[pod_id].status = PodStatus.UNHEALTHY

    def healthy_pods(self, role: PodRole) -> list[Pod]:
        return [p for p in self._pods.values()
                if p.role == role and p.is_available()]


# --- Routing policy ---------------------------------------------------------

class RoutingPolicy(Enum):
    ROUND_ROBIN = "round_robin"    # ignores load; simple; hot-spots under skew
    QUEUE_AWARE = "queue_aware"    # routes to least-loaded pod; absorbs skew


def _select_pod(
    pods: list[Pod],
    policy: RoutingPolicy,
    rr_counter: list[int],     # single-element list so callers share state
) -> Pod | None:
    if not pods:
        return None
    if policy == RoutingPolicy.ROUND_ROBIN:
        pod = pods[rr_counter[0] % len(pods)]
        rr_counter[0] += 1
        return pod
    else:
        return min(pods, key=lambda p: p.queue_depth)


# --- Router -----------------------------------------------------------------

@dataclass
class PDRouter:
    """Two-phase dispatch: prefill pod selection, then decode pod selection.

    Prefill dispatch is parameterized because prefill durations are skewed
    (proportional to prompt length) and load balancing matters most there.
    Decode steps are short and uniform, so round-robin is sufficient there.

    The router increments pod.queue_depth on assignment and the simulation
    decrements it on completion — this is how queue-aware routing stays
    current without a separate RPC to query pod state.
    """
    registry: PodRegistry
    prefill_policy: RoutingPolicy
    _prefill_rr: list[int] = field(default_factory=lambda: [0])
    _decode_rr: list[int] = field(default_factory=lambda: [0])

    def route_prefill(self, req: Request) -> Pod | None:
        pods = self.registry.healthy_pods(PodRole.PREFILL)
        pod = _select_pod(pods, self.prefill_policy, self._prefill_rr)
        if pod:
            pod.queue_depth += 1
        return pod

    def route_decode(self, req: Request) -> Pod | None:
        pods = self.registry.healthy_pods(PodRole.DECODE)
        return _select_pod(pods, RoutingPolicy.ROUND_ROBIN, self._decode_rr)


# --- Request types ----------------------------------------------------------

@dataclass(frozen=True)
class Request:
    req_id: int
    prompt_len: int             # drives prefill cost
    output_len: int             # number of decode steps
    arrival_t: int


@dataclass
class RequestTrace:
    req_id: int
    arrival_t: int
    prefill_pod: str | None = None
    decode_pod: str | None = None
    prefill_done_t: int | None = None
    decode_start_t: int | None = None
    complete_t: int | None = None

    @property
    def completion_latency(self) -> int | None:
        if self.complete_t is None:
            return None
        return self.complete_t - self.arrival_t

    @property
    def ttft(self) -> int | None:
        """Time from arrival to first decode token (prefill + KV transfer)."""
        if self.decode_start_t is None:
            return None
        return self.decode_start_t - self.arrival_t


# --- Tick-level work items --------------------------------------------------

@dataclass
class _PrefillWork:
    req: Request
    remaining_ticks: int


@dataclass
class _DecodeWork:
    req: Request
    steps_remaining: int


# --- Simulation -------------------------------------------------------------

def simulate(
    requests: list[Request],
    router: PDRouter,
    horizon: int,
) -> list[RequestTrace]:
    """Tick-based simulation of the full two-phase request lifecycle.

    Each tick:
      1. Arriving requests are dispatched to prefill pods via the router.
      2. Each prefill pod advances its head work item; on completion it
         schedules a KV handoff to a decode pod (delayed by KV_TRANSFER_TICKS).
      3. Matured handoffs are queued on their decode pod.
      4. Each decode pod advances its head work item, round-robining across
         all in-flight requests so no single request monopolises the pod.
    """
    traces: dict[int, RequestTrace] = {}

    prefill_qs: dict[str, deque[_PrefillWork]] = {
        p.pod_id: deque()
        for p in router.registry.healthy_pods(PodRole.PREFILL)
    }
    decode_qs: dict[str, deque[_DecodeWork]] = {
        p.pod_id: deque()
        for p in router.registry.healthy_pods(PodRole.DECODE)
    }
    handoffs: list[tuple[int, Request, str]] = []  # (ready_t, req, decode_pod_id)

    pending = sorted(requests, key=lambda r: r.arrival_t)
    idx = 0

    for t in range(horizon):

        # 1. Dispatch newly arriving requests to prefill pods.
        while idx < len(pending) and pending[idx].arrival_t == t:
            req = pending[idx]
            idx += 1
            traces[req.req_id] = RequestTrace(req.req_id, req.arrival_t)
            pod = router.route_prefill(req)
            if pod:
                traces[req.req_id].prefill_pod = pod.pod_id
                prefill_qs[pod.pod_id].append(
                    _PrefillWork(req, req.prompt_len * PREFILL_TICKS_PER_TOKEN)
                )

        # 2. Advance each prefill pod one tick on its head item.
        for pod_id, q in prefill_qs.items():
            if not q:
                continue
            work = q[0]
            work.remaining_ticks -= 1
            if work.remaining_ticks == 0:
                q.popleft()
                pod = router.registry._pods[pod_id]
                pod.queue_depth -= 1
                pod.total_processed += 1
                traces[work.req.req_id].prefill_done_t = t
                # Schedule KV transfer to a decode pod.
                dpod = router.route_decode(work.req)
                if dpod:
                    traces[work.req.req_id].decode_pod = dpod.pod_id
                    handoffs.append((t + KV_TRANSFER_TICKS, work.req, dpod.pod_id))

        # 3. Promote matured KV handoffs into decode queues.
        ready_now = [h for h in handoffs if h[0] == t]
        handoffs = [h for h in handoffs if h[0] != t]
        for _, req, dpod_id in ready_now:
            decode_qs[dpod_id].append(
                _DecodeWork(req, req.output_len * DECODE_TICKS_PER_STEP)
            )
            traces[req.req_id].decode_start_t = t

        # 4. Advance each decode pod; round-robin so long decodes don't starve others.
        for pod_id, q in decode_qs.items():
            if not q:
                continue
            work = q[0]
            work.steps_remaining -= 1
            if work.steps_remaining == 0:
                q.popleft()
                router.registry._pods[pod_id].total_processed += 1
                traces[work.req.req_id].complete_t = t
            else:
                q.rotate(-1)

    return list(traces.values())


# --- Metrics and comparison -------------------------------------------------

def avg_completion_latency(traces: list[RequestTrace]) -> float:
    lats = [t.completion_latency for t in traces if t.completion_latency is not None]
    return sum(lats) / len(lats) if lats else float("inf")


def avg_ttft(traces: list[RequestTrace]) -> float:
    vals = [t.ttft for t in traces if t.ttft is not None]
    return sum(vals) / len(vals) if vals else float("inf")


def make_skewed_workload(
    num_requests: int = 40,
    short_prompt_len: int = 10,
    long_prompt_len: int = 80,
    long_fraction: float = 0.3,
    output_len: int = 20,
    arrival_spacing: int = 5,
    seed: int = 42,
) -> list[Request]:
    """Mix of short and long prompts, arriving at fixed intervals.

    Long prompts take 8× more prefill ticks than short ones. Under
    round-robin, runs of long prompts can pile up on the same pod. Under
    queue-aware routing, they spread across pods because the router sees the
    accumulating queue depth and steers away from loaded pods.
    """
    import random
    rng = random.Random(seed)
    return [
        Request(
            req_id=i,
            prompt_len=long_prompt_len if rng.random() < long_fraction else short_prompt_len,
            output_len=output_len,
            arrival_t=i * arrival_spacing,
        )
        for i in range(num_requests)
    ]


def compare_policies(
    requests: list[Request],
    num_prefill_pods: int = 3,
    num_decode_pods: int = 2,
    horizon: int = 2000,
) -> dict[str, list[RequestTrace]]:
    """Run the same workload through two routers that differ only in prefill policy.

    Returns traces keyed by policy name. Compare avg_completion_latency()
    across the two to see the queue-aware advantage under skewed prompt lengths.
    """
    results: dict[str, list[RequestTrace]] = {}
    for policy in (RoutingPolicy.QUEUE_AWARE, RoutingPolicy.ROUND_ROBIN):
        registry = PodRegistry()
        for i in range(num_prefill_pods):
            registry.register(Pod(f"prefill-{i}", PodRole.PREFILL))
        for i in range(num_decode_pods):
            registry.register(Pod(f"decode-{i}", PodRole.DECODE))
        router = PDRouter(registry=registry, prefill_policy=policy)
        results[policy.value] = simulate(requests, router, horizon)
    return results
