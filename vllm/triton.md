# Triton in vLLM

The vLLM repository supports both **CUDA kernels** and **OpenAI Triton** for attention computation.

- **NVIDIA GPUs** — CUDA (FlashAttention) is the default backend, but Triton can also be selected as an alternative.
- **AMD / Intel GPUs** — Triton is the default choice, since CUDA is not available on these platforms.
