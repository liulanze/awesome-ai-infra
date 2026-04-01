# Distributed Inference

## Motivation

Distributing LLM inference is necessary when:

- **Model cannot fit into a single GPU** (even after quantization to reduce weights) — vLLM supports **tensor parallelism** to shard models across multiple GPUs.
- **Model cannot fit into a single node** — vLLM supports **pipeline parallelism** to shard models across multiple GPUs across nodes.
- **Control plane overhead from CPU harms GPU utilization** — vLLM provides an optimized control plane architecture.

## 1. Tensor Parallelism

Tensor parallelism shards weights **horizontally** and computes concurrently across GPUs.

The shard can be either **column parallel** or **row parallel** (applied to the weight matrices):

- **Column parallel** — results from different GPUs are simply concatenated together (**all-gather**).
- **Row parallel** — results from each GPU must be summed (**all-reduce**).

### Benefits

- Total model weight per GPU is reduced to **1/N** (where N = number of shards), saving HBM.
- More free memory means more **KV cache** can be stored.
- Sharded weights reduce per-GPU computation time.

### Tradeoff

- High **communication overhead**, especially for prefill-heavy workloads.

### Implementation

1. Attention and MLP layers use tensor parallelism.
2. Simple distributed execution model: the executor sends data to all workers, and they run the same code.

## 2. Pipeline Parallelism

For models that cannot fit in a single node (e.g., DeepSeek R1 without FP8), layers must be split across multiple nodes. Tensor parallelism has high inter-node network overhead, so **pipeline parallelism** is preferred across nodes.

### Tradeoff

Since layers execute sequentially, pipeline parallelism can cause **idle bubbles**. One mitigation is to use **request groups** (layer groups): each group contains a batch of sequence requests sent sequentially to different GPUs, each with their own KV cache, with locks between them to avoid interference. *(Needs more investigation.)*

## 3. Combining Both

A common strategy is to combine both approaches:

- **Pipeline parallelism** across nodes.
- **Tensor parallelism** within a node.

---

## Appendix: Related Concepts

### Multi-Modal vs Mixture-of-Experts (MoE)

**Multimodality** expands the types of data a model can process by introducing new token representations (e.g., vision or audio embeddings). **Mixture-of-Experts (MoE)** is a sparse architectural technique that improves scaling efficiency by routing tokens through different internal feed-forward subnetworks — without changing the model's input modality.

### FFN vs MLP

In Transformer architectures, the **Feed-Forward Network (FFN)** refers to the position-wise non-linear transformation applied after the self-attention layer, and it is typically implemented as a two-layer **Multi-Layer Perceptron (MLP)**. While MLP describes the general neural network structure of stacked linear layers and activations, FFN denotes its specific functional role within the Transformer block.
