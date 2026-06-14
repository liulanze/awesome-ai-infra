# vLLM CUDA Kernels: What Runs Where

## FlashAttention is NOT a GEMM kernel

FlashAttention is a **fused attention kernel** — it handles the entire attention
computation in one CUDA kernel:

```
scores = Q @ K^T          # matmul
scores = scores / sqrt(d) # scale
scores = softmax(scores)  # softmax  ← the hard part
output = scores @ V       # matmul
```

The key insight: softmax requires seeing the full row before normalizing, which
normally forces you to materialize the full N×N attention matrix in HBM.
FlashAttention avoids this with **online softmax + tiling** — it never writes
the full attention matrix to memory. This is a **memory bandwidth** bottleneck
fix, not a compute throughput fix.

FlashAttention is specifically for the QKV attention score computation.
It does not handle projection layers or FFN.

---

## What Each Layer Uses

| Layer | Operation | Kernel |
|---|---|---|
| Q/K/V projection | `X @ W` | cuBLAS GEMM |
| Attention scores + softmax + context | `QK^T`, softmax, `scores @ V` | **FlashAttention** |
| Output projection | `X @ W` | cuBLAS GEMM |
| FFN / MLP | `X @ W1`, `X @ W2` | cuBLAS GEMM |
| MoE gating + expert matmuls | routing + `X @ W_expert` | cuBLAS or custom `fused_moe` |
| RMSNorm | elementwise scale | Custom CUDA (`layernorm_kernels.cu`) |

---

## KV Cache Dtype and FlashAttention Backend Selection

The KV cache dtype (`--kv-cache-dtype`) controls how K and V tensors are stored
**between forward passes** (in the KV cache). FlashAttention backends differ in
which FP8 variants they can read/write from the KV cache:

| dtype | FA2 | FA3 | FlashInfer |
|---|---|---|---|
| `fp8` / `fp8_e4m3` | ✓ | ✓ | ✓ |
| `fp8_e5m2` | ✗ | ✗ | varies |
| `fp8_ds_mla`, `fp8_inc`, `nvfp4` | ✗ | ✗ | ✗ |

vLLM picks the highest-priority backend whose `supports_kv_cache_dtype()`
returns True. If that check is too permissive (bug in PR #42685), the engine
crashes at startup when the selected backend can't actually handle the dtype.

The GEMM layers (projections, FFN) have their own separate quantization paths
(AWQ, GPTQ, etc.) — entirely independent of the KV cache dtype.

---

## vLLM Kernel Dispatch: Independent Per Layer Type

vLLM has kernel choices at multiple levels, but they're independent of each other:

- **Attention kernel**: FlashAttention v2, FA3, FlashInfer, xFormers
  - Chosen based on: hardware capability + kv-cache dtype
- **Linear/GEMM kernel**: cuBLAS, CUTLASS, custom fused kernels
  - Chosen based on: weight quantization method (AWQ, GPTQ, FP8, etc.)
- **Norm kernel**: custom CUDA (`layernorm_kernels.cu`)
  - No swapping — one implementation, must match weight dtype exactly (see PR #42379)
- **MoE routing**: custom `fused_moe` kernel
  - DeepSeek-specific optimizations; separate from attention backend selection

Same mathematical goal, multiple kernel implementations — but organized by
layer type, not as a single unified dispatch.
