# llm-d

<https://llm-d.ai/docs/architecture>

## Motivation

Distributed inference is essential for cost-effective generative AI at scale.

LLM inference workloads break traditional Kubernetes scaling assumptions due to the resource-heavy and hardware-affinity nature of requests.

Key challenges include:

- **SLO guarantees** — Ensuring Service Level Objectives (throughput, TTFT — Time to First Token, latency) while minimizing resource utilization and operational complexity.
- **Heterogeneous hardware** — Leveraging and managing heterogeneous hardware for better cost-efficiency.
- **Distributed KV cache management** — Managing KV caches across nodes is a critical part of inference efficiency.

## LLM Inference vs. Traditional Services

| Characteristic | Traditional Services | LLM Inference |
|---|---|---|
| **Resource Consumption** | Small (hundreds of MB memory) | **Massive** (tens of GB GPU VRAM) |
| **Hardware Dependency** | Can run on CPU | **Must** run on specific GPUs (e.g., A100, H100) |
| **State** | Stateless | **Stateful** (KV cache needs to be maintained) |
| **Latency Sensitivity** | Moderate | **Extremely high** (directly impacts UX) |

## Scaling MoE with llm-d

The dense MoE kernel uses a single kernel called **FusedMoE**.

**Expert Parallelism** distributes different experts across devices so that each device only holds a subset of experts, reducing per-device memory requirements.

### Autoscaling Challenges

Autoscaling LLM inference in production is far harder than scaling traditional web services. Unlike conventional workloads where scaling is driven by a single factor (e.g., CPU utilization), LLM inference involves **nonlinear effects** — for instance, scaling up prefill instances can saturate decode workers as more traffic flows through. Addressing these autoscaling challenges is one of the main goals of llm-d.

### Key Projects

- Intelligent inference scheduling
- Prefill/Decode (P/D) disaggregation
- Wide expert parallelism
