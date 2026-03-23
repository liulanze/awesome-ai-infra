# Inference

## Online vs Offline Inference

**Online Inference (Real-time serving):**
- Interactive requests — user sends a prompt, waits for response
- **Low latency** is critical — users expect responses in seconds
- Data movement is the bottleneck, while compute is cheap comparing to offline.
- Examples: ChatGPT, Claude, coding assistants, chatbots
- **Weight-only quantization scheme:**
  - Weights → INT8 or INT4
  - Activations → Full precision (FP16/BF16)
  - Computation flow:
    ```
    Embedding (FP16/BF16)
      ↓ (may cast in kernel)
    Activation → FP8 (temporary)
      × Weight → INT8/INT4
      = Accumulate → FP16/FP32
      → Q/K/V (FP16/BF16)
    ```
  - In GEMM: multiplication (mul) can be low precision, but accumulation must be high precision
  - **Why:**
    - Weights are loaded from memory (HBM → GPU) for every forward pass. Reducing weight size (INT8/INT4) directly reduces memory bandwidth consumption.
    - Activations are computed on-the-fly and don't need to be loaded from memory (ephemeral). Keeping them in FP16 maintains accuracy without significant bandwidth cost.

**Offline Inference (Batch processing):**
- Bulk processing — process thousands/millions of inputs together
- **High throughput** prioritized — total time matters more than per-request latency
- No user waiting — results can be processed hours/days later
- matrix multiplications are the bottleneck, data movement is cheap comparing to online.
- Examples: Dataset labeling, document analysis, content generation, embeddings generation, evaluation
- **Weight and activation quantization scheme:**
  - Weights → INT8 / FP8
  - Activations → INT8 / FP8
  - **Why:**
    - Offline inference processes large batches, so maximizing throughput (tokens/second across many requests) matters more than individual request latency.
    - With large batch sizes, compute utilization increases. INT8/FP8 compute (using Tensor Cores) can be faster than FP16
