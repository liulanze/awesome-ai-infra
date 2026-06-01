# Inference Infra Demos

Compact, self-contained Python demonstrations of core LLM serving concepts.
Each file is a concept artifact — readable end-to-end in a few minutes — not
a benchmark or production component. The goal is to make the mechanics
concrete in code, with the decision points named explicitly.

## Contents

| File | Concept |
|------|---------|
| [`01_parallelism_planner.py`](01_parallelism_planner.py) | Enumerate valid TP/DP/EP combinations under attention and MoE constraints |
| [`02_rmsnorm_numerics.py`](02_rmsnorm_numerics.py) | How an FP32-vs-native-dtype final multiply causes RMSNorm drift to compound across layers |
| [`03_autoscaler_sim.py`](03_autoscaler_sim.py) | Workload-aware regime switching with hysteresis and cooldown to prevent flapping |
| [`04_pd_disaggregation_sim.py`](04_pd_disaggregation_sim.py) | Prefill/decode interference, ITL spikes, and goodput as the SLO-aware metric |
| [`05_kv_handoff_push_pull.py`](05_kv_handoff_push_pull.py) | KV-cache transfer modes between prefill and decode workers, and their TTFT timelines |
| [`06_gqa_kv_calculator.py`](06_gqa_kv_calculator.py) | Per-GPU KV memory under MHA / GQA / MQA across tensor-parallel degrees, including replication when TP > num_kv_heads |
| [`07_prefix_cache_router.py`](07_prefix_cache_router.py) | Prefix-aware routing over a shared offload tier; TTFT win on long shared prefixes |

## Reading order

The files are independent and any order works. If you want a guided path:

1. **`06`** — establishes the head/TP arithmetic the rest of the system has to respect.
2. **`01`** — shows where those constraints live in the operator's option space.
3. **`04`** — introduces the metric (goodput) and the failure mode (ITL spikes) that motivate disaggregation.
4. **`05`** — fills in the missing piece: how KV moves between disaggregated workers.
5. **`07`** — extends the cache hierarchy from intra-node KV to a cross-replica shared tier.
6. **`03`** — closes the loop: a controller that keeps the fleet on the right side of the latency/throughput crossover.
7. **`02`** — a kernel-numerics aside; small in scope, large in compounding effect.

## Conventions

- Pure Python where possible; PyTorch only where the concept needs tensors
  (`02`, `05`).
- Cost models use abstract ticks rather than wall-clock seconds. The point is
  the *ratio* between operations (e.g. prefill cost vs. blob load), not the
  absolute number.
- No CLI runners, no plotting code, no test suites — these are reading
  artifacts. Each module exposes the relevant types and functions for use in
  a notebook or REPL.
