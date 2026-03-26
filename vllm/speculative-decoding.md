# Speculative Decoding

## Overview

Speculative decoding accelerates LLM inference by using a small **draft model** to propose K tokens, then having the large **target model** verify all K tokens in a **single forward pass**. The output distribution is mathematically identical to standard autoregressive decoding.

Both models must share the **same vocabulary (tokenizer)**, since rejection sampling compares probabilities over the same token space.

## How It Works

### Why One Forward Pass Is Enough

A causal transformer uses a **causal attention mask** (lower-triangular matrix) so that each position can only attend to itself and earlier positions. This means a single forward pass over `[c₁, ..., cₘ, d₁, d₂, ..., dK]` simultaneously computes:

- `P(· | c₁..cₘ)` — verifies `d₁`
- `P(· | c₁..cₘ, d₁)` — verifies `d₂`
- `P(· | c₁..cₘ, d₁, d₂)` — verifies `d₃`
- ...and so on

Each position's output is identical to what the target model would produce autoregressively, because the mask prevents it from seeing future tokens.

### Verification via Rejection Sampling

For each draft token `dᵢ`, compare target probability `p(dᵢ)` against draft probability `q(dᵢ)`:

- **Accept** with probability `min(1, p(dᵢ) / q(dᵢ))`
- On **first rejection**, resample from `norm(max(0, p(x) - q(x)))` and discard all subsequent draft tokens
- If **all K accepted**, sample a bonus token from the target model's last position for free

If `d₃` is rejected, all computation for `d₄, d₅, ...` is wasted since they were conditioned on the wrong `d₃`:

```
d₁ ✓  accepted
d₂ ✓  accepted
d₃ ✗  rejected → resample corrected token
d₄ 🗑️  wasted
d₅ 🗑️  wasted
```

- **Best case**: All K tokens accepted → advance K+1 tokens for cost of one target forward pass
- **Worst case**: First token rejected → advance 1 token, no worse than normal decoding

### Draft-Verify Loop

The draft model does **not** generate all tokens at once. It works in rounds with a fixed window size K:

1. Draft model generates **K tokens** autoregressively
2. Target model verifies all K in one forward pass
3. Accept tokens up to the first rejection, resample, discard the rest
4. Repeat from the new position until EOS or max length

## Why It Saves Latency

1. **Draft model is fast** — it's small, so generating K tokens autoregressively has low latency
2. **Target model verification is essentially prefill** — processing K tokens in parallel in one forward pass. At low batch sizes (memory-bandwidth bound), processing K tokens costs roughly the same as processing 1, because the bottleneck is loading model weights, not compute

Instead of K sequential decode steps on the target (read all weights K times), you get 1 prefill-like pass (read all weights once for K tokens).

## Production Considerations

### When NOT to Use Speculative Decoding

Speculative decoding helps most when the target model is **memory-bandwidth bound** — typical for autoregressive decoding at low batch sizes, where each forward pass reads all weights but the GPU is underutilized. The extra wasted positions cost almost nothing because they fill idle compute.

Under **high QPS**, requests get batched together, making the GPU **compute-bound**. In this regime:

- Wasted computation from rejected tokens is genuinely costly
- The draft model itself consumes GPU cycles that could serve real requests

**Rule of thumb**: speculative decoding is a **latency optimization** for low-to-moderate load, not a throughput optimization for high load.

### Choosing K and Dynamic Speculative Decoding (DSD)

**K is a hyperparameter** — too large means more wasted compute on rejections, too small means invoking the expensive target model too frequently.

DSD dynamically adjusts K based on **system load** and **speculative accuracy**:

- **Higher load** → GPU more compute-bound → wasted compute is expensive → **shorter K**
- **Lower load** → GPU has idle capacity → wasted compute is cheap → **longer K**
- **Low acceptance rate** → draft model struggles in this context → **shorter K** regardless of load

| | High accuracy | Low accuracy |
|---|---|---|
| **Low load** | Large K | Moderate K |
| **High load** | Moderate K | Small K |
