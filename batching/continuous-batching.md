# Continuous Batching

## Background: Why Not Dynamic Batching?

Dynamic batching collects N requests, processes them as one atomic batch, and returns all results together. This works for single-shot models (e.g., BERT) but breaks down for autoregressive LLM decoding because:

1. **All sequences wait for the slowest one** — the batch has no mechanism to eject a finished sequence mid-batch, so short outputs sit idle until the longest sequence hits EOS.
2. **New requests can't join a running batch** — incoming sequences need prefill (compute-bound, many tokens) while in-flight sequences are in decode (memory-bound, one token). Dynamic batching can't mix these two phases.

## How Continuous Batching Works

Continuous batching operates at the **token step** level instead of the request level.

At **every** decode step, the **scheduler** can:

- **Evict** completed sequences immediately and return their results
- **Admit** new sequences by running their prefill and inserting them into the batch

```
Step 1:  [A B C D]      ← all decoding
Step 2:  [A B C D]      ← B hits EOS → evict B, return result
Step 3:  [A _ C D]      ← slot open, admit new Seq E (prefill + join)
Step 4:  [A E C D]      ← all decoding again
...
```

### Why This Works for LLMs

#### Prefill and decode can be interleaved

Attention processes each sequence **independently** — each sequence only attends to its own KV cache. So nothing forces all sequences to be in the same phase. In a single forward pass:

```
Seq A (decode):   query = 1 token,   attends to its existing 50-token KV cache
Seq B (decode):   query = 1 token,   attends to its existing 120-token KV cache
Seq E (prefill):  query = 30 tokens, builds KV cache from scratch
```

The framework flattens all tokens into one list, uses metadata to tell the attention kernel which tokens belong to which sequence, and the GPU processes them in **one kernel launch**.

In practice, long prefills can starve decode sequences (prefill is compute-heavy). So frameworks like vLLM use **chunked prefill** — break the prefill into smaller chunks and interleave them with decode steps:

```
Step 1: [A_decode, B_decode, E_prefill_chunk1 (tokens 1-10)]
Step 2: [A_decode, B_decode, E_prefill_chunk2 (tokens 11-20)]
Step 3: [A_decode, B_decode, E_prefill_chunk3 (tokens 21-30)]
Step 4: [A_decode, B_decode, E_decode]  ← E fully prefilled, joins decode
```

This keeps decode latency stable — existing users don't feel a stall when a new request arrives.

#### PagedAttention: the memory layer that enables continuous batching

KV cache is the memory bottleneck — it grows with every token, and continuous batching means sequences constantly enter and leave, so the memory layout is always changing.

**Traditional approach** — each sequence gets a **contiguous** block pre-allocated for `max_seq_len`:

```
GPU Memory:
[===Seq A (2048 reserved, 50 used)===][===Seq B (2048 reserved, 120 used)===]
             ↑ 1998 slots wasted                  ↑ 1928 slots wasted
```

This causes internal fragmentation (reserved but unused space) and external fragmentation (holes left by finished sequences), making it hard to admit new sequences.

**PagedAttention** (inspired by OS virtual memory) divides KV cache into **fixed-size pages** (e.g., 16 tokens per page). Pages don't need to be contiguous — a page table maps each sequence's logical positions to physical pages:

```
Physical GPU Memory:
[Page0][Page1][Page2][Page3][Page4][Page5][Page6][Page7]...

Page Table:
  Seq A: logical [0,1,2]    → physical [Page0, Page3, Page5]
  Seq B: logical [0,1,...,7] → physical [Page1, Page2, Page4, Page6, ...]
```

| | Without PagedAttention | With PagedAttention |
|---|---|---|
| New sequence arrives | Need contiguous max-length block | Allocate pages on demand |
| Sequence finishes | Frees block → fragmentation hole | Frees pages → immediately reusable |
| Memory waste | ~60-80% (reserved but unused) | Near zero (pay for actual tokens) |
| Max concurrent sequences | Low | High |

This lets continuous batching pack many more sequences into GPU memory — exactly the fluid, high-utilization batch it needs.

#### Latency is decoupled

A 3-token answer returns in ~3 steps regardless of whether another sequence in the batch needs 200 steps.
