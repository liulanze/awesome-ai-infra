# Simplified vLLM Architecture

## What "OpenAI-Compatible" Means

OpenAI provides a hosted API (`api.openai.com`) for calling their models (GPT-4, etc.) — you pay per token and have no control over the serving infrastructure. Their REST interface (endpoints, JSON schema) became the de facto standard.

vLLM implements the **same interface**, so it's a drop-in replacement — same client, different URL:

```python
# OpenAI's hosted API
client = OpenAI(base_url="https://api.openai.com/v1", api_key="sk-...")

# Your own vLLM server — same code, just change the URL
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
```

The difference: vLLM runs on **your** GPUs with **your** open-source model (Llama, Mistral, etc.), giving you control over cost, latency, data privacy, and model choice.

## Full Request Flow

```
Client (HTTP)
    │
    ▼
┌─────────────────────────────────┐
│  API Server (FastAPI / uvicorn) │
│  - OpenAI-compatible endpoints  │
│  - /v1/completions              │
│  - /v1/chat/completions         │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  AsyncLLMEngine                 │
│  (bridges async API ↔ sync engine)│
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  LLMEngine (the loop)           │
│  Scheduler → Executor → Output  │
└─────────────────────────────────┘
```

## Engine Loop

The CPU is the **control plane**, the GPU is the **data plane**. The GPU only does one thing: the model forward pass. Everything else is orchestrated by three CPU-side components in a loop:

```
┌──────────────────────────────────────────────────────┐
│                  CPU (control plane)                  │
│                                                       │
│  1. Scheduler                                         │
│     - Decides which sequences to prefill/decode       │
│     - Manages KV cache block table (PagedAttention)   │
│     - Handles preemption if memory is tight           │
│                                                       │
│  2. Model Executor                                    │
│     - Prepares input tensors, block tables            │
│     - Dispatches forward pass to GPU  ──────────┐     │
│     - Waits for GPU to return logits  ◄─────────┘     │
│                                                       │
│  3. Output Processor                                  │
│     - Samples next token from logits                  │
│     - Detokenizes, checks stop conditions (EOS/max)   │
│     - Streams finished results back to clients        │
│                                                       │
│  └──→ loop back to Scheduler for next step            │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│                  GPU (data plane)                     │
│                                                       │
│  model.forward(tokens, kv_cache, block_tables)        │
│  → returns logits                                     │
└──────────────────────────────────────────────────────┘
```

### Scheduler (CPU)

Pure logic and bookkeeping — no tensor math. Each iteration it:

1. Checks which sequences have finished (hit EOS or max length) and marks them for eviction
2. Checks available KV cache pages and decides how many new sequences can be admitted
3. Builds the batch metadata: which sequences are prefilling, which are decoding, and their block table mappings
4. If memory is tight, **preempts** lower-priority sequences (swaps their KV pages to CPU or recomputes later)

### Model Executor (CPU → GPU)

Lives on CPU, launches work on GPU. It:

1. Takes the batch metadata from the scheduler
2. Assembles input tensors (token IDs, position IDs, block tables)
3. Calls `model.forward()` on the GPU
4. Gets logits back from the GPU

### Output Processor (CPU)

Post-processing after the GPU returns logits:

1. Samples the next token for each sequence (sampling can also be offloaded to GPU)
2. Detokenizes incrementally for streaming responses
3. Checks stopping conditions (EOS token, max length, stop strings)
4. Returns completed sequences to clients

## Scaling Bottleneck: GPU vs CPU

- **GPU time** is impacted very little by batch size — GPU execution is
  **memory-bound**, meaning increasing batch size does not hurt latency.
- **CPU time** grows ~linearly with batch size — CPU operations are O(N), and
  everything runs in a single Python process (GIL (Global Interpreter Lock)
  prevents concurrency).
