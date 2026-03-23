# vLLM Quantization System

## Architecture

vLLM is a quantization **consumer**, not producer. It loads pre-quantized checkpoints and runs inference with optimized CUDA kernels. The actual quantization process is done by tools like **llm-compressor**.

## Plugin System

vLLM needs to support lots of quantization methods, plugin system can
encapsulate every quantization method's differences into an unit API.

Two abstract base classes in `base_config.py` form the core:

- **`QuantizationConfig`** — reads `quantize_config.json` from the checkpoint to understand how the model was quantized
- **`QuantizeMethodBase`** — the per-layer adapter that handles three things:
  1. `create_weights()` — allocates / register the right parameter containers (e.g., packed int32 `qweight` + `scales` + `qzeros` + `g_idx` instead of a plain float16 `weight`)
  2. `process_weights_after_loading()` — post-load transforms (shuffling to turn
     kernel-friendly layout)
  3. `apply()` — forward pass dispatching to the correct CUDA kernel

- **Packing** = a storage concern: fit more values into less memory
- **Shuffling** = a performance concern: rearrange the packed values so the CUDA kernel can access them with optimal memory access patterns

## Checkpoint Structure

For a quantized model, a typical checkpoint directory looks like:

```
my-quantized-llama/
├── model-00001-of-00003.safetensors   # packed weight tensors (qweight, scales, qzeros, ...)
├── model-00002-of-00003.safetensors
├── model-00003-of-00003.safetensors
├── config.json                        # model architecture (num_layers, hidden_size, ...)
├── quantize_config.json               # quantization metadata (bits=4, group_size=128, method="gptq")
└── tokenizer.json                     # tokenizer
```

## scale / zero-point

Quantization approximates real numbers with integers. A common affine mapping is:

$$q = \text{round}(x / s) + z$$

$$x \approx s \cdot (q - z)$$

Where:

- **x** = original float value (FP16/BF16/FP32)
- **q** = quantized integer (INT8/INT4)
- **s** = scale (how big 1 integer step is)
- **z** = zero-point (integer value that represents real zero)

### Why do we need them?

Because integers have a fixed limited range:

- INT8 is typically [-128, 127]
- INT4 is typically [-8, 7] (or [0..15] depending on packing)

To represent a float range like [-0.73, 0.92], you must decide:

- How to “stretch/squeeze” float range into integer range → **scale**
- Whether zero maps exactly to integer zero or not → **zero-point**

### Dequantization

Quantized models need metadata to interpret the integers correctly. They often do not literally “convert back” to full precision weights — instead:

- Kernels do dequantization on-the-fly inside the matmul, or
- Accumulate in higher precision (FP16/FP32), because that’s required for accuracy
