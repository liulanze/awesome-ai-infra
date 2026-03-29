# Disaggregated Prefill & Decode

## Motivation

Collecting functionalities onto the same node creates contentions:

- High **inter-token-latency (ITL)** when the prefill job is batched with the decoding job.
- Prefill & decode jobs share the same parallelism strategy — cannot tune **time-to-first-token (TTFT)** and **ITL** independently.

## Core Idea

In the disaggregated approach, each node only works on **one job type**. This can be:

- Prefill
- Decode
- Speculative decoding
- Image encoder (multi-modal)

## What Gets Disaggregated

Two things are disaggregated: **tokens** and **KV cache**.

- **Tokens** are cheap to transfer since they are very small.
- **KV cache** is the main challenge due to its large size, so the discussion is primarily about how to transfer KV cache efficiently.

The approach is related to **KV cache sharing** via [LMCache](https://github.com/LMCache/LMCache) + vLLM, researched by UChicago, Stanford, and Microsoft Research.

## Reference

- [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/pdf/2310.07240)
