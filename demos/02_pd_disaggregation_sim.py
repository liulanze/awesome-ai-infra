"""
Prefill/Decode disaggregation simulator.

Prefill is compute-bound; decode is memory-bandwidth-bound. Collocated on the
same worker, a long prefill blocks in-flight decode steps and produces ITL
spikes (head-of-line blocking). Dedicating workers (or pod groups on AKS) to
each phase isolates decode cadence at the cost of a KV-cache handoff.

Reading guide:
  - Request, RequestTrace      : per-request inputs and recorded timestamps
  - CollocatedScheduler        : single worker; prefill ops block decode ops
  - DisaggScheduler            : separate prefill + decode workers + KV handoff
  - goodput()                  : fraction of requests meeting BOTH TTFT and ITL SLOs
                                 SLO - Service Level Objective 你对系统性能立的"军令状" - 必须达到的硬指标。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable


# Cost model: one tick per unit of work. Real systems use wall-clock; ticks
# are sufficient to expose the *ordering* effect that causes ITL spikes.
PREFILL_TICKS_PER_TOKEN = 1
DECODE_TICKS_PER_STEP = 1
KV_TRANSFER_TICKS = 2          # prefill -> decode handoff cost (RDMA/NCCL, cross-pod)


@dataclass(frozen=True)
class Request:
    req_id: int
    prompt_len: int            # drives prefill cost
    output_len: int            # number of decode steps
    arrival_t: int
    ttft_slo: int              # max acceptable ticks from arrival to first token
    itl_slo: int               # max acceptable ticks between consecutive tokens


@dataclass
class RequestTrace:
    req_id: int
    arrival_t: int
    first_token_t: int | None = None
    decode_token_ts: list[int] = field(default_factory=list)

    @property
    def ttft(self) -> int:
        assert self.first_token_t is not None
        return self.first_token_t - self.arrival_t

    @property
    def max_itl(self) -> int:
        if len(self.decode_token_ts) < 2:
            return 0
        return max(b - a for a, b in zip(self.decode_token_ts, self.decode_token_ts[1:]))


# --- Operation queue primitives -------------------------------------------

@dataclass
class PrefillOp:
    req: Request
    remaining_ticks: int


@dataclass
class DecodeOp:
    req: Request
    steps_remaining: int


# --- Collocated: one worker interleaves prefill and decode ----------------

class CollocatedScheduler:
    """Single worker. A prefill op runs to completion before any decode op
    for the same time slice, so long prompts inflate ITL for in-flight decodes.
    """

    def __init__(self) -> None:
        self.queue: deque[PrefillOp | DecodeOp] = deque()
        self.traces: dict[int, RequestTrace] = {}

    def submit(self, req: Request) -> None:
        self.traces[req.req_id] = RequestTrace(req.req_id, req.arrival_t)
        self.queue.append(PrefillOp(req, req.prompt_len * PREFILL_TICKS_PER_TOKEN))

    def run(self, requests: Iterable[Request], horizon: int) -> list[RequestTrace]:
        pending = sorted(requests, key=lambda r: r.arrival_t)
        idx = 0
        for t in range(horizon):
            while idx < len(pending) and pending[idx].arrival_t == t:
                self.submit(pending[idx])
                idx += 1
            if not self.queue:
                continue
            op = self.queue[0]
            if isinstance(op, PrefillOp):
                op.remaining_ticks -= 1
                if op.remaining_ticks == 0:
                    self.queue.popleft()
                    self.queue.append(DecodeOp(op.req, op.req.output_len))
            else:
                trace = self.traces[op.req.req_id]
                trace.decode_token_ts.append(t)
                if trace.first_token_t is None:
                    trace.first_token_t = t
                op.steps_remaining -= 1
                if op.steps_remaining == 0:
                    self.queue.popleft()
                else:
                    self.queue.rotate(-1)  # round-robin across active decodes
        return list(self.traces.values())


# --- Disaggregated: prefill worker(s) and decode worker(s) are independent

class DisaggScheduler:
    """Prefill and decode each have their own worker and queue. Decode cadence
    is independent of prefill load; the cost is a KV-cache handoff between
    workers when prefill completes.
    """

    def __init__(self) -> None:
        self.prefill_q: deque[PrefillOp] = deque()
        self.decode_q: deque[DecodeOp] = deque()
        self.handoff: list[tuple[int, Request]] = []  # (ready_at_tick, req)
        self.traces: dict[int, RequestTrace] = {}

    def submit(self, req: Request) -> None:
        self.traces[req.req_id] = RequestTrace(req.req_id, req.arrival_t)
        self.prefill_q.append(PrefillOp(req, req.prompt_len * PREFILL_TICKS_PER_TOKEN))

    def run(self, requests: Iterable[Request], horizon: int) -> list[RequestTrace]:
        pending = sorted(requests, key=lambda r: r.arrival_t)
        idx = 0
        for t in range(horizon):
            while idx < len(pending) and pending[idx].arrival_t == t:
                self.submit(pending[idx])
                idx += 1

            # Prefill worker advances independently of decode worker.
            if self.prefill_q:
                op = self.prefill_q[0]
                op.remaining_ticks -= 1
                if op.remaining_ticks == 0:
                    self.prefill_q.popleft()
                    self.handoff.append((t + KV_TRANSFER_TICKS, op.req))

            # KV transfers that complete this tick promote requests to decode.
            ready_now = [r for ready_t, r in self.handoff if ready_t == t]
            self.handoff = [(rt, r) for rt, r in self.handoff if rt != t]
            for req in ready_now:
                self.decode_q.append(DecodeOp(req, req.output_len))

            # Decode worker: one step per tick, round-robin across active reqs.
            if self.decode_q:
                op = self.decode_q[0]
                trace = self.traces[op.req.req_id]
                trace.decode_token_ts.append(t)
                if trace.first_token_t is None:
                    trace.first_token_t = t
                op.steps_remaining -= 1
                if op.steps_remaining == 0:
                    self.decode_q.popleft()
                else:
                    self.decode_q.rotate(-1)

        return list(self.traces.values())


# --- The metric that captures both quality and cost in one number ---------

def goodput(traces: Iterable[RequestTrace], requests: Iterable[Request]) -> float:
    """Fraction of completed requests that satisfy BOTH TTFT and ITL SLOs.

    Throughput alone can rise while ITL silently violates the SLO; goodput
    rejects those tokens and is the right lens for disaggregation tradeoffs.
    """
    slo_by_id = {r.req_id: r for r in requests}
    completed = [t for t in traces if t.first_token_t is not None]
    if not completed:
        return 0.0
    ok = 0
    for trace in completed:
        req = slo_by_id[trace.req_id]
        if trace.ttft <= req.ttft_slo and trace.max_itl <= req.itl_slo:
            ok += 1
    return ok / len(completed)
