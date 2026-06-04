# Inference Infra Demos

Compact, self-contained Python demonstrations of core LLM serving concepts.
Each file is a concept artifact — readable end-to-end in a few minutes — not
a benchmark or production component. The goal is to make the mechanics
concrete in code, with the decision points named explicitly.

## Contents

| File | Concept |
|------|---------|
| [`01_tpep_vs_dpep_sim.py`](01_tpep_vs_dpep_sim.py) | TP+EP vs DP+EP throughput/latency tradeoff across concurrency; the crossover and why KV replication vs partitioning drives it |
| [`02_pd_disaggregation_sim.py`](02_pd_disaggregation_sim.py) | Prefill/decode interference, ITL spikes, and goodput as the SLO-aware metric |
| [`03_pd_aware_router.py`](03_pd_aware_router.py) | PD-aware two-phase dispatch (prefill pod → KV handoff → decode pod); pod discovery, health eviction, queue-aware vs round-robin policy |
| [`04_kv_offload_blob.py`](04_kv_offload_blob.py) | KV-cache offload to Azure Blob Storage; cross-replica prefix reuse via a shared namespace; two-tier lookup (local → blob → prefill) |
| [`05_kv_handoff_push_pull.py`](05_kv_handoff_push_pull.py) | KV-cache transfer modes between prefill and decode pods, and their TTFT timelines |
| [`06_rmsnorm_numerics.py`](06_rmsnorm_numerics.py) | How an FP32-vs-native-dtype final multiply causes RMSNorm drift to compound across layers |

## Reading order

The files are independent and any order works. If you want a guided path that
follows the system from fleet configuration down to kernel numerics:

1. **`01`** — establishes the TP-vs-DP tradeoff and why KV memory is the binding constraint at the fleet level.
2. **`02`** — shows the failure mode (ITL spikes) when prefill and decode share a worker, and introduces goodput as the honest metric.
3. **`03`** — introduces the routing layer that makes distributed PD disaggregation work: two-phase dispatch, pod discovery, and the load-balancing policy that matters under skewed prompts.
4. **`05`** — fills in how KV moves between prefill and decode pods (push vs pull), and the TTFT timeline each mode produces.
5. **`04`** — extends the cache hierarchy across replicas: a shared Blob tier so a prefix computed once is reusable fleet-wide.
6. **`06`** — a kernel-numerics aside: a one-line dtype bug that compounds silently across layers.

## Conventions

- Pure Python where possible; PyTorch only where the concept needs tensors (`05`, `06`).
- Cost models use abstract ticks rather than wall-clock seconds. The point is
  the *ratio* between operations (e.g. prefill cost vs. blob load), not the
  absolute number.
- No CLI runners, no plotting code, no test suites — these are reading
  artifacts. Each module exposes the relevant types and functions for use in
  a notebook or REPL.
