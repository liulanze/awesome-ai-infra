"""
Parallelism planner: enumerate valid TP/DP/EP configurations.

Given a model shape and a GPU budget, returns the (TP, DP, EP) combinations
that satisfy the constraints attention sharding and expert routing actually
impose. The constraints are encoded as named predicates so a reader can map
each one to the architectural reason it exists.

Reading guide:
  - ModelSpec, ParallelConfig  : inputs and outputs
  - constraint predicates      : one function per architectural rule
  - enumerate_configs()        : applies all predicates, yields valid combos
  - estimate_memory_per_gpu()  : first-order weight memory under each combo
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


DTYPE_BYTES_BF16 = 2


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_attention_heads: int
    num_kv_heads: int
    hidden_size: int
    num_layers: int
    num_experts: int           # 0 for dense models
    param_count: int           # approximate, in elements (not bytes)


@dataclass(frozen=True)
class ParallelConfig:
    tp: int
    dp: int
    ep: int

    @property
    def world_size(self) -> int:
        return self.tp * self.dp


# --- Constraint predicates: each rule encodes one architectural fact ------

def head_divisible(tp: int, model: ModelSpec) -> bool:
    """TP shards attention along the head dimension. Uneven shards are not
    supported by standard implementations; TP must divide the head count.
    """
    return model.num_attention_heads % tp == 0


def world_size_fits(cfg: ParallelConfig, gpu_count: int) -> bool:
    """TP * DP must equal the available GPU budget exactly. EP is orthogonal:
    it reuses the same GPUs to host different experts.
    """
    return cfg.world_size == gpu_count


def ep_requires_moe(ep: int, model: ModelSpec) -> bool:
    """EP only has meaning for MoE models; EP > 1 on a dense model is invalid."""
    if ep == 1:
        return True
    return model.num_experts > 1


def ep_requires_distribution(cfg: ParallelConfig) -> bool:
    """EP only changes the comms pattern (AllReduce -> AllToAll) when
    TP*DP > 1. On a single rank, EP > 1 is inert.
    """
    if cfg.ep == 1:
        return True
    return cfg.world_size > 1


def ep_divides_experts(ep: int, model: ModelSpec) -> bool:
    """Experts must distribute evenly across the EP dimension."""
    if model.num_experts == 0:
        return ep == 1
    return model.num_experts % ep == 0


# --- Enumeration ----------------------------------------------------------

def enumerate_configs(model: ModelSpec, gpu_count: int) -> Iterator[ParallelConfig]:
    """Yield every (TP, DP, EP) combination that satisfies all predicates.

    The returned set is the operator's option space; choosing among them is a
    workload decision (latency vs throughput target, prompt-length distribution,
    memory headroom for KV cache).
    """
    ep_candidates = [1]
    if model.num_experts > 0:
        ep_candidates += [e for e in _divisors(model.num_experts) if e > 1]

    for tp in _divisors(gpu_count):
        if not head_divisible(tp, model):
            continue
        dp = gpu_count // tp
        for ep in ep_candidates:
            cfg = ParallelConfig(tp=tp, dp=dp, ep=ep)
            if not world_size_fits(cfg, gpu_count):
                continue
            if not ep_requires_moe(ep, model):
                continue
            if not ep_requires_distribution(cfg):
                continue
            if not ep_divides_experts(ep, model):
                continue
            yield cfg


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


# --- Memory estimate (first-order) ----------------------------------------

def estimate_memory_per_gpu(model: ModelSpec, cfg: ParallelConfig) -> int:
    """Approximate per-GPU weight memory in bytes.

    TP shards weights across the TP group; DP replicates the full model; EP
    distributes experts across the EP group. Activations and KV cache are not
    included — those depend on batch shape and live in a separate budget.
    """
    total_bytes = model.param_count * DTYPE_BYTES_BF16

    if model.num_experts > 0 and cfg.ep > 1:
        # Expert weights distribute across EP; non-expert weights still shard by TP.
        expert_share = total_bytes // cfg.ep
        per_gpu = expert_share // cfg.tp
    else:
        per_gpu = total_bytes // cfg.tp

    return per_gpu


# --- Collective characterization (the question interviewers ask) ----------

def collective_pattern(cfg: ParallelConfig, model: ModelSpec) -> str:
    """Which collective dominates the forward pass under this config.

    - TP > 1                    : AllReduce after each attention/MLP block
    - TP*DP > 1 with EP > 1     : AllToAll (token -> expert routing)
    - DP only                   : no cross-GPU collective in forward
    """
    if cfg.ep > 1 and cfg.world_size > 1 and model.num_experts > 0:
        return "AllToAll (expert routing)"
    if cfg.tp > 1:
        return "AllReduce (TP sync)"
    return "none (pure DP)"
