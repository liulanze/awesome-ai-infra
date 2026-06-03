"""
TP+EP vs DP+EP tradeoff simulator.

TP+EP: all GPUs form one TP group and serve each request in lockstep.
       Compute is sharded (low single-request latency), but AllReduce runs
       on every forward pass, and KV is *replicated* across all TP ranks —
       each rank holds the same KV for the shared batch, limiting the
       effective KV pool to one GPU's worth of HBM.

DP+EP: GPUs split into independent replicas, each serving different requests.
       No cross-replica communication; single-request latency is higher
       (no TP sharding benefit). KV is *partitioned* — each replica owns its
       own requests, so the total fleet KV pool grows linearly with DP degree.

The crossover: TP+EP wins at low concurrency (latency regime, small batches);
               DP+EP wins at high concurrency (the larger KV pool sustains more
               in-flight requests, and replica-level throughput keeps rising past
               the point where TP+EP is KV-bound).

EP (expert parallelism) is present in both configs for MoE token routing.
It adds AllToAll overhead symmetrically, shifting absolute latency numbers
but not the TP-vs-DP tradeoff that this file models.

Reading guide:
  - ServingConfig              : GPU budget and parallelism dimensions
  - request_latency()          : per-request latency under TP sharding + AllReduce
  - effective_kv_pool()        : total fleet KV capacity; the key divergence
  - throughput_at_concurrency(): sustained RPS; shows the saturation shape
  - crossover_concurrency()    : load level where DP+EP first beats TP+EP
  - sweep()                    : full (concurrency → throughput) table for both
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Abstract cost constants. Absolute ticks are placeholders; the *ratios*
# between compute gain, AllReduce cost, and KV pool size are the point.
BASE_COMPUTE_TICKS = 100       # single-GPU forward-pass cost per request
ALLREDUCE_TICKS_PER_STEP = 8  # AllReduce overhead per log2(tp) reduction step
KV_BYTES_PER_REQUEST = 512    # KV footprint per in-flight request
GPU_HBM = 8192                # total HBM per GPU (abstract units)
WEIGHT_BYTES_TOTAL = 4096     # full model weight footprint across all GPUs


@dataclass(frozen=True)
class ServingConfig:
    """One (TP, DP, EP) assignment on a fixed GPU budget.

    TP * DP == gpu_count. EP reuses the same GPUs for MoE expert placement
    and does not consume additional hardware.
    """
    name: str
    tp: int
    dp: int
    ep: int

    @property
    def gpu_count(self) -> int:
        return self.tp * self.dp


def request_latency(cfg: ServingConfig) -> float:
    """Per-request latency in ticks.

    TP shards each matrix multiply across tp GPUs, cutting compute by 1/tp.
    Every attention and MLP block then requires an AllReduce synchronization
    across the TP group, adding log2(tp) reduction steps worth of comms.
    DP replicas are fully independent — no comms overhead on any single request.
    """
    compute = BASE_COMPUTE_TICKS / cfg.tp
    comms = ALLREDUCE_TICKS_PER_STEP * math.log2(cfg.tp) if cfg.tp > 1 else 0.0
    return compute + comms


def effective_kv_pool(cfg: ServingConfig) -> int:
    """Total KV cache capacity available to the fleet, in abstract bytes.

    TP+EP: TP shards model weights (freeing HBM per GPU for KV), but the KV
    for a shared batch is replicated across all tp ranks — every rank holds
    the same keys and values for the same set of requests. Effective pool
    = one GPU's KV budget, regardless of how many GPUs are in the TP group.

    DP+EP: each DP replica is self-contained. KV is never replicated across
    replicas; each holds the KV only for its own in-flight requests. The
    fleet's total KV pool scales linearly with dp.
    """
    weight_per_gpu = WEIGHT_BYTES_TOTAL // cfg.tp
    kv_per_gpu = GPU_HBM - weight_per_gpu

    if cfg.tp > 1 and cfg.dp == 1:
        # TP group replicates KV — only one copy is uniquely held
        return kv_per_gpu
    else:
        # DP partitions KV — all dp replicas contribute independent KV space
        return kv_per_gpu * cfg.dp


def max_concurrent_requests(cfg: ServingConfig) -> int:
    """Maximum in-flight requests the fleet can hold in KV memory."""
    return effective_kv_pool(cfg) // KV_BYTES_PER_REQUEST


def throughput_at_concurrency(cfg: ServingConfig, concurrency: int) -> float:
    """Sustained throughput (requests per tick) at a given active concurrency.

    Throughput rises with concurrency while requests fit in KV memory.
    At the KV cap, no further requests can be admitted and throughput
    plateaus — the fleet is memory-bound, not compute-bound.
    """
    lat = request_latency(cfg)
    cap = max_concurrent_requests(cfg)
    return min(concurrency, cap) / lat


@dataclass(frozen=True)
class SweepPoint:
    concurrency: int
    throughput_a: float
    throughput_b: float
    leader: str  # name of the config that is ahead at this concurrency


def sweep(
    cfg_a: ServingConfig,
    cfg_b: ServingConfig,
    max_concurrency: int = 500,
) -> list[SweepPoint]:
    """Paired throughput curves across a concurrency sweep.

    Read the table to see: where does the leader flip, how steep is the
    rise before saturation, and what are the plateau values for each config.
    """
    points = []
    for c in range(1, max_concurrency + 1):
        ta = throughput_at_concurrency(cfg_a, c)
        tb = throughput_at_concurrency(cfg_b, c)
        points.append(SweepPoint(c, ta, tb, cfg_a.name if ta >= tb else cfg_b.name))
    return points


def crossover_concurrency(cfg_a: ServingConfig, cfg_b: ServingConfig) -> int | None:
    """Lowest concurrency at which cfg_b first beats cfg_a on throughput.

    This is the operating point where a static config choice flips: below it,
    prefer cfg_a; above it, prefer cfg_b.
    Returns None if cfg_b never overtakes within a 2000-request sweep.
    """
    for c in range(1, 2001):
        if throughput_at_concurrency(cfg_b, c) > throughput_at_concurrency(cfg_a, c):
            return c
    return None
