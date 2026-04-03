# vLLM Production Stack

https://docs.vllm.ai/projects/production-stack/en/latest/

vLLM Production Stack provides a set of features built around vLLM, including routing, fault tolerance, autoscaling, LoRA management, observability, and KV cache optimizations.

It is cloud-native, supporting the Kubernetes ecosystem and popular cloud providers.

## Motivation

Serving vLLM at scale remains challenging:

- **Cloud-native deployment** — Complex setup and operational overhead.
- **High latency & high cost** — Inefficient resource utilization at scale.
- **Load balancing and request routing** — Lack of intelligent routing strategies.
- **Slow fault recovery and slow autoscaling** — Long downtime during failures or traffic spikes.

## Router

The router is the central component that decides:

- **Where to send requests** — Route each request to the appropriate vLLM instance.
- **How to balance loads** — Distribute traffic evenly or based on custom policies.
- **How to handle failures** — Detect and recover when a vLLM instance goes down.
- **Custom routing logic** — Support user-defined routing rules and strategies.

## LoRA

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that adapts pre-trained models to specific tasks without modifying the original model weights. Instead of retraining all parameters, LoRA injects small trainable low-rank matrices, making fine-tuning lightweight and fast.

## Semantic Cache

Unlike traditional string-matching caches, semantic cache checks **semantic similarity** between queries, enabling cache hits even when the exact wording differs.

## KV Cache in Production Stack

KV cache support is built on top of [LMCache](https://github.com/LMCache/LMCache), which provides:

1. **Cross-instance KV cache sharing** — An interface for sharing KV caches across multiple vLLM instances.
2. **KV cache optimization backend** — Efficient storage and retrieval of cached key-value pairs.
3. **Fast KV cache processing** — Low-latency read/write operations.

KV caches can be stored on local disk or in remote storage, and retrieved on demand when needed.

### KV Cache Offload

Offloads the least-frequently-used (LFU) KV cache entries out of GPU memory to free up space. By default, entries are offloaded to CPU memory, but other storage backends are also supported.
