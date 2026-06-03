# Inference Infra Demos

Compact, self-contained Python demonstrations of core LLM serving concepts.
Each file is a concept artifact — readable end-to-end in a few minutes — not
a benchmark or production component. The goal is to make the mechanics
concrete in code, with the decision points named explicitly.

## Contents

| File | Concept |
|------|---------|
| [`01_tpep_vs_dpep_sim.py`](01_tpep_vs_dpep_sim.py) | TP+EP vs DP+EP throughput/latency tradeoff across concurrency; the crossover and why KV replication vs partitioning drives it |
| [`02_rmsnorm_numerics.py`](02_rmsnorm_numerics.py) | How an FP32-vs-native-dtype final multiply causes RMSNorm drift to compound across layers |
| [`04_pd_disaggregation_sim.py`](04_pd_disaggregation_sim.py) | Prefill/decode interference, ITL spikes, and goodput as the SLO-aware metric |
| [`05_kv_handoff_push_pull.py`](05_kv_handoff_push_pull.py) | KV-cache transfer modes between prefill and decode workers, and their TTFT timelines |
| [`07_prefix_cache_router.py`](07_prefix_cache_router.py) | Prefix-aware routing over a shared offload tier; TTFT win on long shared prefixes |

## Reading order

The files are independent and any order works. If you want a guided path:

1. **`01`** — establishes the TP-vs-DP tradeoff and why KV memory is the binding constraint.
2. **`04`** — shows the failure mode (ITL spikes, head-of-line blocking) and goodput as the honest metric.
3. **`05`** — fills in the missing piece: how KV moves between disaggregated workers (push vs pull).
4. **`07`** — extends the cache hierarchy to a shared cross-replica tier and shows the TTFT routing win.
5. **`02`** — a kernel-numerics aside; small in scope, large in compounding effect.

## Conventions

- Pure Python where possible; PyTorch only where the concept needs tensors (`02`, `05`).
- Cost models use abstract ticks rather than wall-clock seconds. The point is
  the *ratio* between operations (e.g. prefill cost vs. blob load), not the
  absolute number.
- No CLI runners, no plotting code, no test suites — these are reading
  artifacts. Each module exposes the relevant types and functions for use in
  a notebook or REPL.
