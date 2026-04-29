# Awesome AI Infra

A collection of technical deep-dives on modern LLM inference infrastructure — covering vLLM internals, distributed serving, quantization, and production deployment.

## Contents

| Topic | Files | What's Covered |
|-------|-------|----------------|
| **[vllm/](vllm/)** | 5 docs | Core vLLM architecture (paged KV cache, scheduler, model runner), speculative decoding, disaggregated prefill/decode, Triton kernels |
| **[batching/](batching/)** | 1 doc | Continuous batching vs. dynamic batching, PagedAttention memory management |
| **[inference/](inference/)** | 2 docs | Tensor parallelism, pipeline parallelism, online vs. offline inference strategies |
| **[quantization/](quantization/)** | 2 docs | FP8 formats (E4M3/E5M2), vLLM quantization plugin system, scaling strategies |
| **[prod-stack/](prod-stack/)** | 1 doc | vLLM Production Stack: routing, autoscaling, fault tolerance, LoRA, KV cache with LMCache |
| **[llm-d/](llm-d/)** | 1 doc | LLM-D inference scheduling, MoE expert parallelism, prefill/decode disaggregation |
| **[ms-foundry/](ms-foundry/)** | 1 doc | Microsoft Foundry on Azure, NVIDIA Vera Rubin NVL72 specs |
| **[api/](api/)** | 1 doc | OpenAI-compatible API layer, request/response formats |

## License

[Apache 2.0](LICENSE)
